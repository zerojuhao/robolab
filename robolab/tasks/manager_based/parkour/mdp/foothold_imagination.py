"""Training-only imagined foothold prediction and support/edge penalties.

The predictor consumes an SSR-style privileged observation group plus the
current action and predicts the next valid contact for both feet. This module only
coordinates environment interaction; model, replay, optimization, and support
evaluation live in the foothold_prediction package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.sensors.ray_caster.ray_caster import RayCaster

from robolab.sensors.volume_points import VolumePoints
from robolab.sensors.volume_points.points_generator import grid3d_points_generator

from .foothold_prediction import (
    FootholdPredictorCfg,
    FootholdPredictorTrainer,
    FootholdSupportCfg,
    FootholdSupportEvaluator,
    normalize_foothold_predictor_cfg,
    normalize_foothold_support_cfg,
)
from .terrain_family import get_terrain_family_ids

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


__all__ = [
    "FootholdImaginationManager",
    "FootholdSupportCfg",
    "imagined_foothold_edge_penetration",
    "imagined_foothold_guidance",
]


class FootholdImaginationManager:
    """Owns prediction, delayed contact labels, training buffers, and reward queries."""

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        predictor_cfg: FootholdPredictorCfg | dict,
        support_cfg: FootholdSupportCfg | dict,
    ) -> None:
        self.env = env
        self.cfg = normalize_foothold_predictor_cfg(predictor_cfg)
        self.support_cfg = normalize_foothold_support_cfg(support_cfg)
        self.device = env.device
        self.num_envs = env.num_envs

        self.trainer: FootholdPredictorTrainer | None = None
        self.input_dim: int | None = None

        self._robot = None
        self._contact_sensor: ContactSensor | None = None
        self._foot_body_ids: list[int] | None = None
        self._contact_body_ids: list[int] | None = None
        self._sole_offsets: list[torch.Tensor] | None = None
        self._terrain_mesh = None
        self._support_evaluator: FootholdSupportEvaluator | None = None

        self._pending_inputs: torch.Tensor | None = None
        self._pending_frames: torch.Tensor | None = None
        self._pending_counts = torch.zeros(
            self.num_envs, 2, dtype=torch.long, device=self.device
        )
        self._contact_support_stats = torch.zeros(
            2, dtype=torch.float64, device=self.device
        )
        self._guidance_stance_deficiency_stats = torch.zeros(
            2, dtype=torch.float64, device=self.device
        )
        self._guidance_swing_deficiency_stats = torch.zeros(
            2, dtype=torch.float64, device=self.device
        )
        self._guidance_reward_stats = torch.zeros(
            2, dtype=torch.float64, device=self.device
        )
        self._terrain_ready_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._prediction_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._latest_mu_b: torch.Tensor | None = None
        self._latest_sigma: torch.Tensor | None = None
        self._latest_base_frame: torch.Tensor | None = None
        self._latest_foot_yaw: torch.Tensor | None = None
        self._prepared = False
        self._step_counter = 0

        self._last_metrics: dict[str, float] = {}

    @staticmethod
    def _yaw_from_quat(quat_w: torch.Tensor) -> torch.Tensor:
        """Return yaw for scalar-first quaternions."""
        w, x, y, z = quat_w.unbind(dim=-1)
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))

    def _ensure_scene_handles(self) -> None:
        if self._robot is not None:
            return

        self._robot = self.env.scene["robot"]
        body_ids, body_names = self._robot.find_bodies(
            self.support_cfg.foot_body_names, preserve_order=True
        )
        if len(body_ids) != 2:
            raise RuntimeError(
                "Foothold imagination requires exactly two ordered foot bodies; "
                f"resolved {body_names}."
            )
        self._foot_body_ids = list(body_ids)

        self._contact_sensor = self.env.scene.sensors[
            self.support_cfg.contact_sensor_name
        ]
        contact_ids, contact_names = self._contact_sensor.find_bodies(
            self.support_cfg.foot_body_names, preserve_order=True
        )
        if len(contact_ids) != 2:
            raise RuntimeError(
                "Foothold imagination contact sensor must resolve exactly two feet; "
                f"resolved {contact_names}."
            )
        self._contact_body_ids = list(contact_ids)

        sole_offsets: list[torch.Tensor] = []
        mesh_path = None
        for scanner_name in self.support_cfg.height_scanner_names:
            scanner = self.env.scene.sensors[scanner_name]
            # Accessing data initializes the ray caster before reusing its sole pattern.
            _ = scanner.data.ray_hits_w
            offsets = scanner.ray_starts[0, :, :2].detach().clone()
            sole_offsets.append(offsets)
            current_mesh_path = scanner.cfg.mesh_prim_paths[0]
            if mesh_path is None:
                mesh_path = current_mesh_path
            elif mesh_path != current_mesh_path:
                raise RuntimeError(
                    "Left and right foothold height scanners must raycast the same terrain mesh."
                )
        if sole_offsets[0].shape != sole_offsets[1].shape:
            raise RuntimeError(
                "Left and right foothold scan patterns must contain the same number of points."
            )
        self._sole_offsets = sole_offsets
        self._terrain_mesh = RayCaster.meshes[mesh_path]
        self._support_evaluator = FootholdSupportEvaluator(
            self.support_cfg, sole_offsets, self._terrain_mesh, self.device
        )

    def _build_trainer(self, input_dim: int) -> None:
        if self.trainer is not None:
            if input_dim != self.input_dim:
                raise RuntimeError(
                    f"Foothold predictor input changed from {self.input_dim} to {input_dim}."
                )
            return

        self.input_dim = input_dim
        # ``prepare_step`` runs inside the rollout's ``torch.inference_mode()``.
        # Training state created there would otherwise become inference tensors and
        # fail on the first backward pass after iteration 0.
        with torch.inference_mode(False):
            self.trainer = FootholdPredictorTrainer(input_dim, self.cfg, self.device)
            pending_shape = (self.num_envs, 2, self.cfg.max_pending_steps, input_dim)
            self._pending_inputs = torch.zeros(
                pending_shape, dtype=torch.float16, device=self.device
            )
            self._pending_frames = torch.zeros(
                self.num_envs,
                2,
                self.cfg.max_pending_steps,
                5,
                dtype=torch.float32,
                device=self.device,
            )

    def _base_frame_snapshot(self) -> torch.Tensor:
        base_pos = self._robot.data.root_pos_w
        yaw = self._yaw_from_quat(self._robot.data.root_quat_w)
        return torch.cat(
            (base_pos, torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)),
            dim=-1,
        )

    def _terrain_ready(self) -> torch.Tensor:
        """Return the globally gated mask shared by every terrain."""
        return self._terrain_ready_mask

    def enable_inference_guidance(self) -> None:
        """Enable guidance for evaluation after loading a trained predictor."""
        self._terrain_ready_mask.fill_(True)

    def prepare_step(
        self, privileged_state: torch.Tensor, action: torch.Tensor
    ) -> None:
        """Capture one policy step and predict both future contacts before simulation advances."""
        self._ensure_scene_handles()
        privileged_state = privileged_state.detach().to(
            device=self.device, dtype=torch.float32
        )
        action = action.detach().to(device=self.device, dtype=torch.float32)
        predictor_input = torch.cat((privileged_state, action), dim=-1)
        self._build_trainer(predictor_input.shape[-1])

        base_frame = self._base_frame_snapshot()
        foot_yaw = self._yaw_from_quat(
            self._robot.data.body_quat_w[:, self._foot_body_ids]
        )
        contact = (
            self._contact_sensor.data.current_contact_time[:, self._contact_body_ids]
            > 0.0
        )
        swing = ~contact
        terrain_ready = self._terrain_ready()
        training_ready = terrain_ready

        if self._step_counter % max(self.cfg.pending_sample_stride, 1) == 0:
            for foot_id in range(2):
                counts = self._pending_counts[:, foot_id]
                swing_foot = swing[:, foot_id] & training_ready
                overflow = swing_foot & (counts >= self.cfg.max_pending_steps)
                can_store = swing_foot & ~overflow

                env_ids = can_store.nonzero(as_tuple=False).squeeze(-1)
                if env_ids.numel() > 0:
                    write_ids = counts[env_ids]
                    self._pending_inputs[env_ids, foot_id, write_ids] = predictor_input[
                        env_ids
                    ].to(torch.float16)
                    self._pending_frames[env_ids, foot_id, write_ids] = base_frame[
                        env_ids
                    ]
                    self._pending_counts[env_ids, foot_id] += 1

        self._latest_mu_b, self._latest_sigma = self.trainer.predict(predictor_input)
        self._latest_base_frame = base_frame
        self._latest_foot_yaw = foot_yaw
        self._prediction_valid.fill_(True)
        self._prepared = True
        self._step_counter += 1

    def _record_contact_support(
        self, events: torch.Tensor, foot_pos_w: torch.Tensor
    ) -> None:
        """Accumulate real touchdown support metrics outside slope terrains."""
        if events.numel() == 0:
            return

        env_ids = events[:, 0]
        foot_ids = events[:, 1]
        family_ids = get_terrain_family_ids(self.env)[env_ids]
        valid = self._terrain_ready()[env_ids]
        if self.support_cfg.disable_slope_family:
            valid = valid & (family_ids != self.support_cfg.slope_family_id)
        env_ids = env_ids[valid]
        foot_ids = foot_ids[valid]
        if env_ids.numel() == 0:
            return

        foot_yaws = self._yaw_from_quat(
            self._robot.data.body_quat_w[:, self._foot_body_ids]
        )[env_ids, foot_ids]
        support_ratios = self._support_evaluator.contact_support_ratio(
            foot_pos_w[env_ids, foot_ids], foot_yaws, foot_ids
        )
        self._contact_support_stats[0] += support_ratios.double().sum()
        self._contact_support_stats[1] += support_ratios.numel()

    def _finalize_contacts(self) -> None:
        first_contact = self._contact_sensor.compute_first_contact(self.env.step_dt)[
            :, self._contact_body_ids
        ]
        foot_pos_w = self._robot.data.body_pos_w[:, self._foot_body_ids]

        events = first_contact.nonzero(as_tuple=False)
        self._record_contact_support(events, foot_pos_w)
        if events.numel() > 0:
            event_env_ids = events[:, 0]
            event_foot_ids = events[:, 1]
            event_counts = self._pending_counts[event_env_ids, event_foot_ids]
            valid_event = event_counts > 0
            event_env_ids = event_env_ids[valid_event]
            event_foot_ids = event_foot_ids[valid_event]
            event_counts = event_counts[valid_event]

            if event_env_ids.numel() > 0:
                touchdown_xy = foot_pos_w[event_env_ids, event_foot_ids, :2]
                inputs = self._pending_inputs[event_env_ids, event_foot_ids]
                frames = self._pending_frames[event_env_ids, event_foot_ids]
                delta_xy = touchdown_xy[:, None] - frames[..., :2]
                cos_yaw = frames[..., 3]
                sin_yaw = frames[..., 4]
                target_x = cos_yaw * delta_xy[..., 0] + sin_yaw * delta_xy[..., 1]
                target_y = -sin_yaw * delta_xy[..., 0] + cos_yaw * delta_xy[..., 1]
                targets = torch.stack((target_x, target_y), dim=-1)

                step_ids = torch.arange(self.cfg.max_pending_steps, device=self.device)
                sample_mask = step_ids.unsqueeze(0) < event_counts.unsqueeze(1)
                tail_steps = int(getattr(self.cfg, "train_pending_tail_steps", 0))
                if tail_steps > 0:
                    # Keep only the last K swing steps (or the full swing if shorter).
                    sample_mask = sample_mask & (
                        step_ids.unsqueeze(0)
                        >= (event_counts.unsqueeze(1) - tail_steps)
                    )
                replay_foot_ids = event_foot_ids.unsqueeze(1).expand_as(sample_mask)
                self.trainer.add_samples(
                    inputs[sample_mask],
                    targets[sample_mask],
                    replay_foot_ids[sample_mask],
                )

            self._pending_counts[events[:, 0], events[:, 1]] = 0

        contact = (
            self._contact_sensor.data.current_contact_time[:, self._contact_body_ids]
            > 0.0
        )
        stale = contact & (~first_contact) & (self._pending_counts > 0)
        self._pending_counts[stale] = 0

    def compute_reward(
        self,
        enable_terrain_foot_weights: bool = True,
        stairs_weight_min: float = 0.0,
        stairs_weight_max: float = 1.0,
    ) -> torch.Tensor:
        """SSR foothold reward from stance support and imagined swing support."""
        self._ensure_scene_handles()
        self._finalize_contacts()
        reward = torch.zeros(self.num_envs, device=self.device)
        if (
            not self._prepared
            or self.trainer is None
            or self._latest_mu_b is None
            or self._latest_sigma is None
            or not self.trainer.reward_enabled
        ):
            return reward

        terrain_ready = self._terrain_ready()
        if not terrain_ready.any():
            return reward

        family_ids = get_terrain_family_ids(self.env)
        contact = (
            self._contact_sensor.data.current_contact_time[:, self._contact_body_ids]
            > 0.0
        )
        slope = torch.zeros_like(terrain_ready)
        if self.support_cfg.disable_slope_family:
            slope = family_ids == self.support_cfg.slope_family_id
        eligible_env = terrain_ready & ~slope & self._prediction_valid
        eligible = eligible_env.unsqueeze(-1).expand(-1, 2)
        rho_tilde = torch.zeros(self.num_envs, 2, device=self.device)

        stance_pairs = (contact & eligible).nonzero(as_tuple=False)
        if stance_pairs.numel() > 0:
            env_ids = stance_pairs[:, 0]
            foot_ids = stance_pairs[:, 1]
            foot_pos_w = self._robot.data.body_pos_w[:, self._foot_body_ids]
            stance_deficiency = self._support_evaluator.contact_unsupported_penalty(
                foot_pos_w[env_ids, foot_ids],
                self._latest_foot_yaw[env_ids, foot_ids],
                foot_ids,
                point_weights=None,
            ).clamp(0.0, 1.0)
            rho_tilde[env_ids, foot_ids] = stance_deficiency
            self._guidance_stance_deficiency_stats[0] += (
                stance_deficiency.double().sum()
            )
            self._guidance_stance_deficiency_stats[1] += float(
                stance_deficiency.numel()
            )

        swing_pairs = ((~contact) & eligible).nonzero(as_tuple=False)
        if swing_pairs.numel() > 0:
            env_ids = swing_pairs[:, 0]
            foot_ids = swing_pairs[:, 1]
            means = self._latest_mu_b[env_ids, foot_ids]
            sigmas = self._latest_sigma[env_ids, foot_ids]
            candidate_xy, candidate_weights = self.trainer.grid.expectation_points(
                means, sigmas
            )
            swing_deficiency = self._support_evaluator.expected_support_deficiency(
                candidate_xy,
                candidate_weights,
                self._latest_base_frame,
                self._latest_foot_yaw,
                env_ids,
                foot_ids,
                family_ids=family_ids,
                enable_terrain_foot_weights=enable_terrain_foot_weights,
                stairs_weight_min=stairs_weight_min,
                stairs_weight_max=stairs_weight_max,
                eval_chunk_size=self.cfg.grid.expectation_eval_chunk_size,
            )
            rho_tilde[env_ids, foot_ids] = swing_deficiency
            self._guidance_swing_deficiency_stats[0] += (
                swing_deficiency.double().sum()
            )
            self._guidance_swing_deficiency_stats[1] += float(
                swing_deficiency.numel()
            )

        summed_deficiency = rho_tilde.sum(dim=-1)
        reward[eligible_env] = torch.exp(
            -summed_deficiency[eligible_env].square() / self.support_cfg.reward_sigma
        )
        # SSR defines slope deficiency as zero, hence r_f=1.
        reward[terrain_ready & slope & self._prediction_valid] = 1.0
        active_reward = reward[terrain_ready & self._prediction_valid]
        self._guidance_reward_stats[0] += active_reward.double().sum()
        self._guidance_reward_stats[1] += float(active_reward.numel())
        return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

    def compute_edge_penetration(
        self,
        volume_sensor_cfg: SceneEntityCfg,
        enable_terrain_foot_weights: bool = True,
        stairs_weight_min: float = 0.0,
        stairs_weight_max: float = 1.0,
        tolerance: float = 0.0,
        sole_layer_only: bool = False,
    ) -> torch.Tensor:
        """Penalize imagined swing footholds that penetrate virtual stair-edge obstacles.

        Does not call ``_finalize_contacts`` (owned by ``compute_reward``). Uses the
        latest predictor outputs from ``prepare_step``.
        """
        self._ensure_scene_handles()
        penalty = torch.zeros(self.num_envs, device=self.device)
        if (
            not self._prepared
            or self.trainer is None
            or not self.trainer.reward_enabled
        ):
            return penalty

        terrain_ready = self._terrain_ready()
        if not terrain_ready.any():
            return penalty

        family_ids = get_terrain_family_ids(self.env)
        contact = (
            self._contact_sensor.data.current_contact_time[:, self._contact_body_ids]
            > 0.0
        )
        eligible = terrain_ready & self._prediction_valid
        if self.support_cfg.disable_slope_family:
            eligible = eligible & (
                family_ids != self.support_cfg.slope_family_id
            )
        swing_pairs = ((~contact) & eligible.unsqueeze(-1).expand(-1, 2)).nonzero(
            as_tuple=False
        )
        if swing_pairs.numel() == 0:
            return penalty

        volume_sensor: VolumePoints = self.env.scene.sensors[volume_sensor_cfg.name]
        points_cfg = volume_sensor.cfg.points_generator
        local_points = grid3d_points_generator(points_cfg).to(self.device)
        if sole_layer_only:
            sole_mask = torch.isclose(
                local_points[:, 2],
                torch.tensor(points_cfg.z_min, device=self.device, dtype=local_points.dtype),
                atol=1e-6,
            )
            local_points = local_points[sole_mask]
        if local_points.numel() == 0:
            return penalty

        terrain = self.env.scene.terrain
        virtual_obstacles = getattr(terrain, "virtual_obstacles", None) or {}
        if not virtual_obstacles:
            return penalty

        if self._latest_mu_b is None:
            return penalty

        env_ids = swing_pairs[:, 0]
        foot_ids = swing_pairs[:, 1]
        imagined_pen = self._support_evaluator.edge_penetration(
            self._latest_mu_b,
            self._latest_base_frame,
            self._latest_foot_yaw,
            env_ids,
            foot_ids,
            family_ids,
            local_points,
            points_cfg.x_min,
            points_cfg.x_max,
            points_cfg.z_min,
            virtual_obstacles,
            enable_terrain_foot_weights=enable_terrain_foot_weights,
            stairs_weight_min=stairs_weight_min,
            stairs_weight_max=stairs_weight_max,
            tolerance=tolerance,
        )
        penalty.index_add_(0, env_ids, imagined_pen)
        return penalty

    def reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() > 0:
            self._pending_counts[env_ids] = 0
            self._prediction_valid[env_ids] = False
            for latest in (
                self._latest_mu_b,
                self._latest_sigma,
                self._latest_base_frame,
                self._latest_foot_yaw,
            ):
                if latest is not None:
                    latest[env_ids] = 0

    def update_predictor(self) -> dict[str, float]:
        """Optimize the predictor and aggregate rollout metrics across all ranks."""
        metrics = dict(self._last_metrics)
        for retired in (
            "Foothold/Accuracy/top1_acc",
            "Foothold/Guidance/expected_support_topk",
            "Foothold/Guidance/forward_window_fallback_ratio",
            "Foothold/Guidance/selected_mass",
            "Foothold/Guidance/mean_deficiency_topk",
            "Foothold/Guidance/point_deficiency",
        ):
            metrics.pop(retired, None)

        if self.trainer is not None:
            metrics.update(self.trainer.update())

        rollout_stats = torch.tensor(
            (
                self.env.scene.terrain.terrain_levels.float().sum().item(),
                self.num_envs,
            ),
            dtype=torch.float64,
            device=self.device,
        )
        rollout_stats = torch.cat(
            (
                rollout_stats,
                self._contact_support_stats,
                self._guidance_stance_deficiency_stats,
                self._guidance_swing_deficiency_stats,
                self._guidance_reward_stats,
            )
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                rollout_stats, op=torch.distributed.ReduceOp.SUM
            )
        (
            terrain_level_sum,
            environment_count,
            contact_support_sum,
            contact_count,
            stance_deficiency_sum,
            stance_deficiency_count,
            swing_deficiency_sum,
            swing_deficiency_count,
            guidance_reward_sum,
            guidance_reward_count,
        ) = rollout_stats.tolist()

        mean_terrain_level = terrain_level_sum / environment_count
        curriculum_ready = mean_terrain_level > self.cfg.curriculum_level_threshold
        self._terrain_ready_mask.fill_(curriculum_ready)

        contact_metric_name = "Foothold/Contact/support_ratio"
        metrics.pop(contact_metric_name, None)
        if contact_count > 0.0:
            metrics[contact_metric_name] = contact_support_sum / contact_count

        stance_metric_name = "Foothold/Guidance/stance_deficiency"
        metrics.pop(stance_metric_name, None)
        if stance_deficiency_count > 0.0:
            metrics[stance_metric_name] = (
                stance_deficiency_sum / stance_deficiency_count
            )

        swing_metric_name = "Foothold/Guidance/swing_expected_deficiency"
        metrics.pop(swing_metric_name, None)
        if swing_deficiency_count > 0.0:
            metrics[swing_metric_name] = swing_deficiency_sum / swing_deficiency_count

        reward_metric_name = "Foothold/Guidance/reward"
        metrics.pop(reward_metric_name, None)
        if guidance_reward_count > 0.0:
            metrics[reward_metric_name] = guidance_reward_sum / guidance_reward_count

        self._contact_support_stats.zero_()
        self._guidance_stance_deficiency_stats.zero_()
        self._guidance_swing_deficiency_stats.zero_()
        self._guidance_reward_stats.zero_()
        self._last_metrics = metrics
        return metrics

    def state_dict(self) -> dict:
        if self.trainer is None:
            return {"input_dim": self.input_dim}
        return {"input_dim": self.input_dim, "trainer": self.trainer.state_dict()}

    def load_state_dict(self, state: dict, load_optimizer: bool = True) -> None:
        input_dim = state.get("input_dim")
        if input_dim is None:
            return
        self._build_trainer(int(input_dim))
        # Pre-refactor checkpoints stored model state directly in this mapping.
        trainer_state = state.get("trainer", state)
        self.trainer.load_state_dict(trainer_state, load_optimizer=load_optimizer)
        if bool(getattr(self.cfg, "clear_train_buffer_on_resume", True)):
            self.trainer.clear_train_buffer()


def imagined_foothold_guidance(
    env: ManagerBasedRLEnv,
    enable_terrain_foot_weights: bool = True,
    stairs_weight_min: float = 0.0,
    stairs_weight_max: float = 1.0,
) -> torch.Tensor:
    """Return the positive SSR stance/swing foothold-support reward."""
    manager = getattr(env, "foothold_guidance", None)
    if manager is None:
        return torch.zeros(env.num_envs, device=env.device)
    return manager.compute_reward(
        enable_terrain_foot_weights=enable_terrain_foot_weights,
        stairs_weight_min=stairs_weight_min,
        stairs_weight_max=stairs_weight_max,
    )


def imagined_foothold_edge_penetration(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("feet_volume_points"),
    enable_terrain_foot_weights: bool = True,
    stairs_weight_min: float = 0.0,
    stairs_weight_max: float = 1.0,
    tolerance: float = 0.0,
    sole_layer_only: bool = False,
    scale: float = 1.0,
) -> torch.Tensor:
    """Penalize predicted swing footholds that intersect virtual stair-edge obstacles.

    ``scale`` multiplies the raw penetration term so its effective strength can be
    reduced without changing the shared curriculum weight schedule.
    """
    manager = getattr(env, "foothold_guidance", None)
    if manager is None:
        return torch.zeros(env.num_envs, device=env.device)
    penalty = manager.compute_edge_penetration(
        volume_sensor_cfg=sensor_cfg,
        enable_terrain_foot_weights=enable_terrain_foot_weights,
        stairs_weight_min=stairs_weight_min,
        stairs_weight_max=stairs_weight_max,
        tolerance=tolerance,
        sole_layer_only=sole_layer_only,
    )
    return penalty * scale
