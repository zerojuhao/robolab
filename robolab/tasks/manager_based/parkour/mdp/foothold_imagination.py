"""Imagined foothold prediction and swing-support guidance.

The privileged teacher consumes critic observations plus the current action and
predicts the next contact XY and relative landing yaw for both feet. Training,
prediction, reward, and contact logs are stairs-only.

Locomotion ``feet_at_plane`` penalizes currently unsupported stance. This module
pre-penalizes the teacher-expected unsupported fraction of feet that were swinging
when the action was applied, using the predicted landing yaw to rotate the sole.
Pending swing samples use the same force-history contact mask as ``feet_at_plane``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.sensors import ContactSensor
from isaaclab.sensors.ray_caster.ray_caster import RayCaster
from isaaclab.utils.math import wrap_to_pi

from .foothold_prediction import (
    FootholdPredictorCfg,
    FootholdPredictorTrainer,
    FootholdSupportCfg,
    FootholdSupportEvaluator,
    empty_predictor_logs,
    normalize_foothold_predictor_cfg,
    normalize_foothold_support_cfg,
)
from .terrain_family import (
    FOOTHOLD_GUIDANCE_FAMILY_IDS,
    get_terrain_family_ids,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["FootholdImaginationManager", "FootholdSupportCfg", "imagined_foothold_guidance"]

# Flattened teacher pose: ``[lx, ly, lyaw, rx, ry, ryaw]``.
_TEACHER_FLAT_DIM = 6


def _family_is_foothold_active(family_ids: torch.Tensor) -> torch.Tensor:
    active = torch.zeros_like(family_ids, dtype=torch.bool)
    for family_id in FOOTHOLD_GUIDANCE_FAMILY_IDS:
        active |= family_ids == family_id
    return active


def _yaw_from_quat(quat_w: torch.Tensor) -> torch.Tensor:
    """Yaw from scalar-first quaternions with arbitrary leading dims."""
    w, x, y, z = quat_w.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))


class FootholdImaginationManager:
    """Owns prediction, delayed contact labels, training, and reward queries."""

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
        self._support_evaluator: FootholdSupportEvaluator | None = None

        self._pending_inputs: torch.Tensor | None = None
        self._pending_frames: torch.Tensor | None = None
        self._pending_counts = torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)
        self._pending_write = torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)
        self._contact_support_stats = torch.zeros(2, dtype=torch.float64, device=self.device)
        self._guidance_reward_stats = torch.zeros(2, dtype=torch.float64, device=self.device)
        self._terrain_ready_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prediction_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._swing_at_prepare = torch.zeros(
            self.num_envs, 2, dtype=torch.bool, device=self.device
        )
        self._latest_mu_b: torch.Tensor | None = None
        self._latest_yaw_b: torch.Tensor | None = None
        self._latest_yaw_w: torch.Tensor | None = None
        self._latest_sigma: torch.Tensor | None = None
        self._latest_base_frame: torch.Tensor | None = None
        self._prepared = False
        self._step_counter = 0
        self._latest_mu_flat = torch.full(
            (self.num_envs, _TEACHER_FLAT_DIM),
            float("nan"),
            dtype=torch.float32,
            device=self.device,
        )

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

        self._contact_sensor = self.env.scene.sensors[self.support_cfg.contact_sensor_name]
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
            _ = scanner.data.ray_hits_w
            sole_offsets.append(scanner.ray_starts[0, :, :2].detach().clone())
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
        self._support_evaluator = FootholdSupportEvaluator(
            self.support_cfg, sole_offsets, RayCaster.meshes[mesh_path], self.device
        )

    def _build_trainer(self, input_dim: int) -> None:
        if self.trainer is not None:
            if input_dim != self.input_dim:
                raise RuntimeError(
                    f"Foothold predictor input changed from {self.input_dim} to {input_dim}."
                )
            return

        self.input_dim = input_dim
        with torch.inference_mode(False):
            self.trainer = FootholdPredictorTrainer(input_dim, self.cfg, self.device)
            self._pending_inputs = torch.zeros(
                (self.num_envs, 2, self.cfg.max_pending_steps, input_dim),
                dtype=torch.float16,
                device=self.device,
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
        yaw = _yaw_from_quat(self._robot.data.root_quat_w)
        return torch.cat(
            (base_pos, torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)),
            dim=-1,
        )

    def _stairs_active(self) -> torch.Tensor:
        return self._terrain_ready_mask & _family_is_foothold_active(
            get_terrain_family_ids(self.env)
        )

    def stairs_active_mask(self) -> torch.Tensor:
        return self._stairs_active()

    def enable_inference_guidance(self) -> None:
        self._terrain_ready_mask.fill_(True)

    def latest_mu_flat(self) -> torch.Tensor:
        """Teacher pose ``[N, 6]`` in the current base frame. Inactive envs are NaN."""
        return self._latest_mu_flat

    def _store_prediction(
        self,
        mean_xy: torch.Tensor,
        mean_yaw: torch.Tensor,
        sigma_xy: torch.Tensor,
        base_frame: torch.Tensor,
        active: torch.Tensor,
    ) -> None:
        self._latest_mu_b = mean_xy
        self._latest_yaw_b = mean_yaw
        self._latest_sigma = sigma_xy
        self._latest_base_frame = base_frame
        base_yaw = torch.atan2(base_frame[:, 4], base_frame[:, 3])
        self._latest_yaw_w = wrap_to_pi(base_yaw.unsqueeze(-1) + mean_yaw)
        self._prediction_valid.copy_(active)
        pose = torch.cat((mean_xy, mean_yaw.unsqueeze(-1)), dim=-1)
        self._latest_mu_flat.copy_(pose.reshape(self.num_envs, _TEACHER_FLAT_DIM))
        inactive = ~active
        self._latest_mu_b[inactive] = 0
        self._latest_yaw_b[inactive] = 0
        self._latest_yaw_w[inactive] = 0
        self._latest_sigma[inactive] = 0
        self._latest_mu_flat[inactive] = float("nan")

    def prepare_step(self, privileged_state: torch.Tensor, action: torch.Tensor) -> None:
        """Cache a swing sample and predict both future contacts before physics."""
        self._ensure_scene_handles()
        privileged_state = privileged_state.detach().to(device=self.device, dtype=torch.float32)
        action = action.detach().to(device=self.device, dtype=torch.float32)
        predictor_input = torch.cat((privileged_state, action), dim=-1)
        self._build_trainer(predictor_input.shape[-1])

        base_frame = self._base_frame_snapshot()
        self._swing_at_prepare = ~self._in_contact_like_feet_at_plane()
        active = self._stairs_active()
        self._pending_counts[~active] = 0
        self._pending_write[~active] = 0

        if self._step_counter % max(self.cfg.pending_sample_stride, 1) == 0:
            max_pending = int(self.cfg.max_pending_steps)
            for foot_id in range(2):
                env_ids = (self._swing_at_prepare[:, foot_id] & active).nonzero(
                    as_tuple=False
                ).squeeze(-1)
                if env_ids.numel() == 0:
                    continue
                write_ids = self._pending_write[env_ids, foot_id]
                self._pending_inputs[env_ids, foot_id, write_ids] = predictor_input[env_ids].to(
                    torch.float16
                )
                self._pending_frames[env_ids, foot_id, write_ids] = base_frame[env_ids]
                self._pending_write[env_ids, foot_id] = (write_ids + 1) % max_pending
                self._pending_counts[env_ids, foot_id] = torch.clamp(
                    self._pending_counts[env_ids, foot_id] + 1, max=max_pending
                )

        mean_xy, mean_yaw, sigma_xy = self.trainer.predict(predictor_input)
        self._store_prediction(mean_xy, mean_yaw, sigma_xy, base_frame, active)
        self._prepared = True
        self._step_counter += 1

    def _record_contact_support(self, events: torch.Tensor, foot_pos_w: torch.Tensor) -> None:
        if events.numel() == 0:
            return
        env_ids = events[:, 0]
        foot_ids = events[:, 1]
        valid = self._stairs_active()[env_ids]
        env_ids = env_ids[valid]
        foot_ids = foot_ids[valid]
        if env_ids.numel() == 0:
            return
        foot_yaws = _yaw_from_quat(self._robot.data.body_quat_w[:, self._foot_body_ids])[
            env_ids, foot_ids
        ]
        support_ratios = self._support_evaluator.contact_support_ratio(
            foot_pos_w[env_ids, foot_ids],
            foot_yaws,
            foot_ids,
            family_ids=get_terrain_family_ids(self.env)[env_ids],
        )
        self._contact_support_stats[0] += support_ratios.double().sum()
        self._contact_support_stats[1] += support_ratios.numel()

    def _label_touchdowns(self, events: torch.Tensor, foot_pos_w: torch.Tensor) -> None:
        event_env_ids = events[:, 0]
        event_foot_ids = events[:, 1]
        event_counts = self._pending_counts[event_env_ids, event_foot_ids]
        valid_event = event_counts > 0
        event_env_ids = event_env_ids[valid_event]
        event_foot_ids = event_foot_ids[valid_event]
        event_counts = event_counts[valid_event]
        if event_env_ids.numel() == 0:
            return

        touchdown_xy = foot_pos_w[event_env_ids, event_foot_ids, :2]
        max_pending = int(self.cfg.max_pending_steps)
        inputs = self._pending_inputs[event_env_ids, event_foot_ids]
        frames = self._pending_frames[event_env_ids, event_foot_ids]
        write = self._pending_write[event_env_ids, event_foot_ids]
        oldest = torch.where(event_counts >= max_pending, write, torch.zeros_like(write))
        step_ids = torch.arange(max_pending, device=self.device)
        gather_idx = (oldest.unsqueeze(1) + step_ids) % max_pending
        inputs = torch.gather(inputs, 1, gather_idx.unsqueeze(-1).expand_as(inputs))
        frames = torch.gather(frames, 1, gather_idx.unsqueeze(-1).expand_as(frames))

        delta_xy = touchdown_xy[:, None] - frames[..., :2]
        cos_yaw = frames[..., 3]
        sin_yaw = frames[..., 4]
        target_x = cos_yaw * delta_xy[..., 0] + sin_yaw * delta_xy[..., 1]
        target_y = -sin_yaw * delta_xy[..., 0] + cos_yaw * delta_xy[..., 1]
        touchdown_yaw = _yaw_from_quat(self._robot.data.body_quat_w[:, self._foot_body_ids])[
            event_env_ids, event_foot_ids
        ]
        target_yaw = wrap_to_pi(touchdown_yaw.unsqueeze(1) - torch.atan2(sin_yaw, cos_yaw))
        targets = torch.stack((target_x, target_y, target_yaw), dim=-1)

        sample_mask = step_ids.unsqueeze(0) < event_counts.unsqueeze(1)
        tail_steps = int(self.cfg.train_pending_tail_steps)
        if tail_steps > 0:
            sample_mask = sample_mask & (
                step_ids.unsqueeze(0) >= (event_counts.unsqueeze(1) - tail_steps)
            )
        replay_foot_ids = event_foot_ids.unsqueeze(1).expand_as(sample_mask)
        self.trainer.add_samples(
            inputs[sample_mask],
            targets[sample_mask],
            replay_foot_ids[sample_mask],
        )

    def _finalize_contacts(self) -> None:
        first_contact = self._contact_sensor.compute_first_contact(self.env.step_dt)[
            :, self._contact_body_ids
        ]
        foot_pos_w = self._robot.data.body_pos_w[:, self._foot_body_ids]
        events = first_contact.nonzero(as_tuple=False)
        self._record_contact_support(events, foot_pos_w)
        if events.numel() > 0:
            self._label_touchdowns(events, foot_pos_w)
            self._pending_counts[events[:, 0], events[:, 1]] = 0
            self._pending_write[events[:, 0], events[:, 1]] = 0

        in_contact = self._in_contact_like_feet_at_plane()
        stale = in_contact & (~first_contact) & (self._pending_counts > 0)
        self._pending_counts[stale] = 0
        self._pending_write[stale] = 0

    def _in_contact_like_feet_at_plane(self) -> torch.Tensor:
        """Contact mask shared with ``feet_at_plane`` (history peak force > 1 N)."""
        net_forces = self._contact_sensor.data.net_forces_w_history[:, :, self._contact_body_ids]
        return torch.max(torch.norm(net_forces, dim=-1), dim=1)[0] > 1.0

    def compute_reward(
        self,
        enable_terrain_foot_weights: bool = True,
        stairs_weight_min: float = 0.0,
        stairs_weight_max: float = 1.0,
    ) -> torch.Tensor:
        """Sum of per-swing-foot expected unsupported fractions in ``[0, 2]``."""
        self._ensure_scene_handles()
        self._finalize_contacts()
        reward = torch.zeros(self.num_envs, device=self.device)
        if (
            not self._prepared
            or self.trainer is None
            or self._latest_mu_b is None
            or self._latest_yaw_w is None
            or self._latest_sigma is None
            or not self.trainer.reward_enabled
        ):
            return reward
        if not self._terrain_ready_mask.any():
            return reward

        family_ids = get_terrain_family_ids(self.env)
        eligible_env = self._stairs_active() & self._prediction_valid
        eligible = eligible_env.unsqueeze(-1).expand(-1, 2)

        swing_deficiency = torch.zeros(self.num_envs, 2, device=self.device)
        swing_pairs = (self._swing_at_prepare & eligible).nonzero(as_tuple=False)
        if swing_pairs.numel() > 0:
            env_ids = swing_pairs[:, 0]
            foot_ids = swing_pairs[:, 1]
            candidate_xy, candidate_weights = self.trainer.grid.expectation_points(
                self._latest_mu_b[env_ids, foot_ids],
                self._latest_sigma[env_ids, foot_ids],
            )
            predicted = self._support_evaluator.expected_support_deficiency(
                candidate_xy,
                candidate_weights,
                self._latest_base_frame,
                self._latest_yaw_w,
                env_ids,
                foot_ids,
                family_ids=family_ids,
                enable_terrain_foot_weights=enable_terrain_foot_weights,
                stairs_weight_min=stairs_weight_min,
                stairs_weight_max=stairs_weight_max,
                eval_chunk_size=self.cfg.grid.expectation_eval_chunk_size,
            )
            swing_deficiency[env_ids, foot_ids] = predicted

        reward[eligible_env] = swing_deficiency[eligible_env].sum(dim=-1)
        active_reward = reward[eligible_env]
        self._guidance_reward_stats[0] += active_reward.double().sum()
        self._guidance_reward_stats[1] += float(active_reward.numel())
        return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

    def reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self._pending_counts[env_ids] = 0
        self._pending_write[env_ids] = 0
        self._prediction_valid[env_ids] = False
        self._swing_at_prepare[env_ids] = False
        for latest in (
            self._latest_mu_b,
            self._latest_yaw_b,
            self._latest_yaw_w,
            self._latest_sigma,
            self._latest_base_frame,
        ):
            if latest is not None:
                latest[env_ids] = 0
        self._latest_mu_flat[env_ids] = float("nan")

    def update_predictor(self) -> dict[str, float]:
        if self.trainer is not None:
            metrics = self.trainer.update()
        else:
            metrics = empty_predictor_logs()

        rollout_stats = torch.tensor(
            (
                self.env.scene.terrain.terrain_levels.float().sum().item(),
                self.num_envs,
            ),
            dtype=torch.float64,
            device=self.device,
        )
        rollout_stats = torch.cat((rollout_stats, self._contact_support_stats, self._guidance_reward_stats))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(rollout_stats, op=torch.distributed.ReduceOp.SUM)
        (
            terrain_level_sum,
            environment_count,
            contact_support_sum,
            contact_count,
            guidance_reward_sum,
            guidance_reward_count,
        ) = rollout_stats.tolist()

        if terrain_level_sum / environment_count > self.cfg.curriculum_level_threshold:
            self._terrain_ready_mask.fill_(True)

        metrics["Foothold/Contact/support_ratio"] = (
            contact_support_sum / contact_count if contact_count > 0.0 else 0.0
        )
        metrics["Foothold/Guidance/reward"] = (
            guidance_reward_sum / guidance_reward_count if guidance_reward_count > 0.0 else 0.0
        )

        self._contact_support_stats.zero_()
        self._guidance_reward_stats.zero_()
        return metrics

    def state_dict(self) -> dict:
        state = {
            "input_dim": self.input_dim,
            "terrain_ready": bool(self._terrain_ready_mask.any().item()),
        }
        if self.trainer is not None:
            state["trainer"] = self.trainer.state_dict()
        return state

    def load_state_dict(self, state: dict, load_optimizer: bool = True) -> None:
        input_dim = state.get("input_dim")
        if input_dim is None:
            return
        self._build_trainer(int(input_dim))
        trainer_state = state.get("trainer", state)
        self.trainer.load_state_dict(trainer_state, load_optimizer=load_optimizer)
        if self.cfg.clear_train_buffer_on_resume:
            self.trainer.clear_train_buffer()
        if bool(state.get("terrain_ready", False)):
            self._terrain_ready_mask.fill_(True)


def imagined_foothold_guidance(
    env: ManagerBasedRLEnv,
    enable_terrain_foot_weights: bool = True,
    stairs_weight_min: float = 0.0,
    stairs_weight_max: float = 1.0,
) -> torch.Tensor:
    """Swing-foot expected unsupported fraction. Pair with a negative reward weight."""
    manager = getattr(env, "foothold_guidance", None)
    if manager is None:
        return torch.zeros(env.num_envs, device=env.device)
    return manager.compute_reward(
        enable_terrain_foot_weights=enable_terrain_foot_weights,
        stairs_weight_min=stairs_weight_min,
        stairs_weight_max=stairs_weight_max,
    )
