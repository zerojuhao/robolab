"""Imagined foothold prediction and swing-support guidance.

The privileged teacher consumes the critic observation group plus the current
action and predicts the next contact XY and relative landing yaw for both feet.
Labels use the same full-root-quaternion body frame as the foothold targets.
Training, prediction, reward, and contact logs cover discrete footholds and stairs.

Locomotion ``feet_at_plane`` penalizes currently unsupported stance. This module
pre-penalizes the teacher-expected unsupported fraction of feet that were swinging
when the action was applied, using the predicted landing yaw to rotate the sole.
Guidance weights early swing more (the action can still change the landing) and
near-contact less. Teacher NLL is uniform across remaining steps-to-contact.
Pending swing samples use the same force-history contact mask as
``feet_at_plane``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.sensors import ContactSensor
from isaaclab.sensors.ray_caster.ray_caster import RayCaster
from isaaclab.utils.math import (
    quat_apply_inverse,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
    wrap_to_pi,
)

from .contact_metrics import vertical_contact_mask
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
    DISCRETE_FAMILY_ID,
    FOOTHOLD_GUIDANCE_FAMILY_IDS,
    STAIRS_DOWN_FAMILY_ID,
    STAIRS_UP_FAMILY_ID,
    get_terrain_family_ids,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["FootholdImaginationManager", "FootholdSupportCfg", "imagined_foothold_guidance"]

# Flattened teacher pose: ``[lx, ly, lyaw, rx, ry, ryaw]``.
_TEACHER_FLAT_DIM = 6
# Root pos (3) + quat (4) + left/right foot body-frame z (2).
_BASE_FRAME_DIM = 9
_HORIZON_BIN_EDGES = (4, 8, 16)
_HORIZON_BIN_NAMES = ("h0_3", "h4_7", "h8_15", "h16p")
_FAMILY_BIN_NAMES = ("discrete", "stairs_down", "stairs_up")
# Per-bin: count, xy_se, x_sum, y_sum, x_se, y_se.
_DIAG_STAT_DIM = 6
_SUPPORT_HIST_BINS = 20
_TILT_HIST_BINS = 18
# Count, final/area/terrain sums, below-threshold count, tilt sum, then histograms.
_CONTACT_STAT_DIM = 6 + _SUPPORT_HIST_BINS + _TILT_HIST_BINS


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
        self._pending_contact_seen = torch.zeros(
            self.num_envs, 2, dtype=torch.bool, device=self.device
        )
        self._contact_support_stats = torch.zeros(
            len(_FAMILY_BIN_NAMES), _CONTACT_STAT_DIM, dtype=torch.float64, device=self.device
        )
        # Pending, rejected vertical force, rejected support, accepted.
        self._touchdown_filter_stats = torch.zeros(
            len(_FAMILY_BIN_NAMES), 4, dtype=torch.float64, device=self.device
        )
        self._guidance_reward_stats = torch.zeros(2, dtype=torch.float64, device=self.device)
        # deficiency sum/count, eval-sigma sum/count, horizon-weight sum.
        self._swing_guidance_stats = torch.zeros(5, dtype=torch.float64, device=self.device)
        self._horizon_diag_stats = torch.zeros(
            len(_HORIZON_BIN_NAMES), _DIAG_STAT_DIM, dtype=torch.float64, device=self.device
        )
        self._family_diag_stats = torch.zeros(
            len(_FAMILY_BIN_NAMES), _DIAG_STAT_DIM, dtype=torch.float64, device=self.device
        )
        self._terrain_ready_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prediction_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._swing_at_prepare = torch.zeros(
            self.num_envs, 2, dtype=torch.bool, device=self.device
        )
        self._swing_age_at_prepare = torch.zeros(
            self.num_envs, 2, dtype=torch.long, device=self.device
        )
        self._latest_mu_b: torch.Tensor | None = None
        self._latest_yaw_b: torch.Tensor | None = None
        self._latest_yaw_w: torch.Tensor | None = None
        self._latest_sigma: torch.Tensor | None = None
        self._latest_yaw_sigma: torch.Tensor | None = None
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
                _BASE_FRAME_DIM,
                dtype=torch.float32,
                device=self.device,
            )

    def _base_frame_snapshot(self) -> torch.Tensor:
        """Root pose plus per-foot body-frame z for labeling touchdowns."""
        root_pos = self._robot.data.root_pos_w
        root_quat = self._robot.data.root_quat_w
        foot_pos_w = self._robot.data.body_pos_w[:, self._foot_body_ids]
        foot_pos_b = quat_apply_inverse(
            root_quat.unsqueeze(1).expand(-1, 2, -1),
            foot_pos_w - root_pos.unsqueeze(1),
        )
        return torch.cat((root_pos, root_quat, foot_pos_b[..., 2]), dim=-1)

    def _touchdown_pose_in_frames(
        self,
        touchdown_pos_w: torch.Tensor,
        touchdown_quat_w: torch.Tensor,
        frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Body-frame XY and relative yaw of a contact pose in each pending root frame."""
        root_pos = frames[..., :3]
        root_quat = frames[..., 3:7]
        pos_b = quat_apply_inverse(root_quat, touchdown_pos_w.unsqueeze(1) - root_pos)
        rel_quat = quat_mul(
            quat_conjugate(root_quat),
            touchdown_quat_w.unsqueeze(1).expand_as(root_quat),
        )
        return pos_b[..., :2], wrap_to_pi(_yaw_from_quat(rel_quat))

    def _foothold_active(self) -> torch.Tensor:
        return self._terrain_ready_mask & _family_is_foothold_active(
            get_terrain_family_ids(self.env)
        )

    def foothold_active_mask(self) -> torch.Tensor:
        return self._foothold_active()

    def _guidance_horizon_weight(self, elapsed_steps: torch.Tensor) -> torch.Tensor:
        """Early-swing weight: action can still change the landing.

        Uses elapsed swing as a causal proxy because remaining time is unknown.
        Independent of teacher NLL sample weights.
        """
        tau = float(self.cfg.guidance_horizon_tau)
        if tau <= 0.0:
            return torch.ones(elapsed_steps.shape[0], device=self.device, dtype=torch.float32)
        floor = max(float(self.cfg.guidance_horizon_weight_min), 0.0)
        return torch.exp(-elapsed_steps.to(dtype=torch.float32) / tau).clamp(min=floor)

    def enable_inference_guidance(self) -> None:
        self._terrain_ready_mask.fill_(True)

    def latest_mu_flat(self) -> torch.Tensor:
        """Teacher pose ``[N, 6]`` in the current body frame. Inactive envs are NaN."""
        return self._latest_mu_flat

    def _store_prediction(
        self,
        mean_xy: torch.Tensor,
        mean_yaw: torch.Tensor,
        sigma_xy: torch.Tensor,
        sigma_yaw: torch.Tensor,
        base_frame: torch.Tensor,
        active: torch.Tensor,
    ) -> None:
        self._latest_mu_b = mean_xy
        self._latest_yaw_b = mean_yaw
        self._latest_sigma = sigma_xy
        self._latest_yaw_sigma = sigma_yaw
        self._latest_base_frame = base_frame
        zeros = torch.zeros_like(mean_yaw)
        rel_quat = quat_from_euler_xyz(zeros, zeros, mean_yaw)
        root_quat = base_frame[:, 3:7].unsqueeze(1).expand(-1, 2, -1)
        self._latest_yaw_w = wrap_to_pi(_yaw_from_quat(quat_mul(root_quat, rel_quat)))
        self._prediction_valid.copy_(active)
        pose = torch.cat((mean_xy, mean_yaw.unsqueeze(-1)), dim=-1)
        self._latest_mu_flat.copy_(pose.reshape(self.num_envs, _TEACHER_FLAT_DIM))
        inactive = ~active
        self._latest_mu_b[inactive] = 0
        self._latest_yaw_b[inactive] = 0
        self._latest_yaw_w[inactive] = 0
        self._latest_sigma[inactive] = 0
        self._latest_yaw_sigma[inactive] = 0
        self._latest_mu_flat[inactive] = float("nan")

    def prepare_step(self, privileged_state: torch.Tensor, action: torch.Tensor) -> None:
        """Cache a swing sample and predict both future contacts before physics."""
        self._ensure_scene_handles()
        privileged_state = privileged_state.detach().to(device=self.device, dtype=torch.float32)
        action = action.detach().to(device=self.device, dtype=torch.float32)
        predictor_input = torch.cat((privileged_state, action), dim=-1)
        self._build_trainer(predictor_input.shape[-1])

        base_frame = self._base_frame_snapshot()
        in_contact = self._in_contact_like_feet_at_plane()
        self._clear_finished_invalid_contacts(in_contact)
        self._swing_at_prepare = ~in_contact
        active = self._foothold_active()
        self._pending_counts[~active] = 0
        self._pending_write[~active] = 0
        self._pending_contact_seen[~active] = False

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

        mean_xy, mean_yaw, sigma_xy, sigma_yaw = self.trainer.predict(predictor_input)
        self._store_prediction(mean_xy, mean_yaw, sigma_xy, sigma_yaw, base_frame, active)
        self._swing_age_at_prepare.copy_(self._pending_counts)
        self._prepared = True
        self._step_counter += 1

    def _clear_finished_invalid_contacts(self, in_contact: torch.Tensor) -> None:
        """Discard swing samples after a rejected contact ends."""
        finished = self._pending_contact_seen & ~in_contact
        self._pending_counts[finished] = 0
        self._pending_write[finished] = 0
        self._pending_contact_seen[finished] = False

    def _contact_support_components(
        self,
        events: torch.Tensor,
        foot_pos_w: torch.Tensor,
        enable_terrain_foot_weights: bool = True,
        stairs_weight_min: float = 0.1,
        stairs_weight_max: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate support for environment-foot contact pairs."""
        if events.numel() == 0:
            empty = torch.zeros(0, device=self.device)
            return empty, empty, empty
        env_ids = events[:, 0]
        foot_ids = events[:, 1]
        foot_yaws = _yaw_from_quat(self._robot.data.body_quat_w[:, self._foot_body_ids])[
            env_ids, foot_ids
        ]
        return self._support_evaluator.contact_support_components(
            foot_pos_w[env_ids, foot_ids],
            foot_yaws,
            foot_ids,
            family_ids=get_terrain_family_ids(self.env)[env_ids],
            enable_terrain_foot_weights=enable_terrain_foot_weights,
            stairs_weight_min=stairs_weight_min,
            stairs_weight_max=stairs_weight_max,
        )

    @staticmethod
    def _family_bin_ids(family_ids: torch.Tensor) -> torch.Tensor:
        bins = torch.full_like(family_ids, -1)
        bins = torch.where(family_ids == DISCRETE_FAMILY_ID, 0, bins)
        bins = torch.where(family_ids == STAIRS_DOWN_FAMILY_ID, 1, bins)
        return torch.where(family_ids == STAIRS_UP_FAMILY_ID, 2, bins)

    def _accumulate_contact_support_stats(
        self,
        family_ids: torch.Tensor,
        support: torch.Tensor,
        area_support: torch.Tensor,
        terrain_support: torch.Tensor,
        tilt: torch.Tensor,
    ) -> None:
        family_bins = self._family_bin_ids(family_ids)
        for bin_id in range(len(_FAMILY_BIN_NAMES)):
            mask = family_bins == bin_id
            if not mask.any():
                continue
            values = support[mask]
            tilts = tilt[mask]
            stats = self._contact_support_stats[bin_id]
            stats[0] += values.numel()
            stats[1] += values.double().sum()
            stats[2] += area_support[mask].double().sum()
            stats[3] += terrain_support[mask].double().sum()
            stats[4] += (values < self.support_cfg.touchdown_support_ratio_min).double().sum()
            stats[5] += tilts.double().sum()
            support_bins = (values.clamp(0.0, 1.0) * _SUPPORT_HIST_BINS).long().clamp_max(
                _SUPPORT_HIST_BINS - 1
            )
            tilt_bins = (
                tilts.clamp(0.0, 0.5 * torch.pi) / (0.5 * torch.pi) * _TILT_HIST_BINS
            ).long().clamp_max(_TILT_HIST_BINS - 1)
            stats[6 : 6 + _SUPPORT_HIST_BINS] += torch.bincount(
                support_bins, minlength=_SUPPORT_HIST_BINS
            ).double()
            stats[6 + _SUPPORT_HIST_BINS :] += torch.bincount(
                tilt_bins, minlength=_TILT_HIST_BINS
            ).double()

    def _accumulate_touchdown_filter_stats(
        self,
        family_ids: torch.Tensor,
        vertical_contact: torch.Tensor,
        supported: torch.Tensor,
    ) -> None:
        family_bins = self._family_bin_ids(family_ids)
        accepted = vertical_contact & supported
        for bin_id in range(len(_FAMILY_BIN_NAMES)):
            mask = family_bins == bin_id
            if not mask.any():
                continue
            stats = self._touchdown_filter_stats[bin_id]
            stats[0] += mask.sum()
            stats[1] += (mask & ~vertical_contact).sum()
            stats[2] += (mask & vertical_contact & ~supported).sum()
            stats[3] += (mask & accepted).sum()

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

        touchdown_pos_w = foot_pos_w[event_env_ids, event_foot_ids]
        touchdown_quat_w = self._robot.data.body_quat_w[:, self._foot_body_ids][
            event_env_ids, event_foot_ids
        ]
        max_pending = int(self.cfg.max_pending_steps)
        inputs = self._pending_inputs[event_env_ids, event_foot_ids]
        frames = self._pending_frames[event_env_ids, event_foot_ids]
        write = self._pending_write[event_env_ids, event_foot_ids]
        oldest = torch.where(event_counts >= max_pending, write, torch.zeros_like(write))
        step_ids = torch.arange(max_pending, device=self.device)
        gather_idx = (oldest.unsqueeze(1) + step_ids) % max_pending
        inputs = torch.gather(inputs, 1, gather_idx.unsqueeze(-1).expand_as(inputs))
        frames = torch.gather(frames, 1, gather_idx.unsqueeze(-1).expand_as(frames))

        target_xy, target_yaw = self._touchdown_pose_in_frames(
            touchdown_pos_w, touchdown_quat_w, frames
        )
        targets = torch.cat((target_xy, target_yaw.unsqueeze(-1)), dim=-1)

        sample_mask = step_ids.unsqueeze(0) < event_counts.unsqueeze(1)
        tail_steps = int(self.cfg.train_pending_tail_steps)
        if tail_steps > 0:
            sample_mask = sample_mask & (
                step_ids.unsqueeze(0) >= (event_counts.unsqueeze(1) - tail_steps)
            )
        replay_foot_ids = event_foot_ids.unsqueeze(1).expand_as(sample_mask)
        steps_to_contact = event_counts.unsqueeze(1) - 1 - step_ids
        family_ids = get_terrain_family_ids(self.env)[event_env_ids]
        self._accumulate_label_diagnostics(
            inputs[sample_mask],
            targets[sample_mask],
            replay_foot_ids[sample_mask],
            steps_to_contact.expand_as(sample_mask)[sample_mask],
            family_ids.unsqueeze(1).expand_as(sample_mask)[sample_mask],
        )
        self.trainer.add_samples(
            inputs[sample_mask],
            targets[sample_mask],
            replay_foot_ids[sample_mask],
            steps_to_contact.expand_as(sample_mask)[sample_mask],
            family_ids.unsqueeze(1).expand_as(sample_mask)[sample_mask],
        )

    def _horizon_bin_ids(self, steps_to_contact: torch.Tensor) -> torch.Tensor:
        edges = torch.tensor(_HORIZON_BIN_EDGES, device=self.device, dtype=steps_to_contact.dtype)
        return torch.bucketize(steps_to_contact, edges)

    def _accumulate_diag_stats(
        self,
        stats: torch.Tensor,
        bin_ids: torch.Tensor,
        xy_err: torch.Tensor,
    ) -> None:
        if bin_ids.numel() == 0:
            return
        num_bins = stats.shape[0]
        valid = (bin_ids >= 0) & (bin_ids < num_bins)
        if not valid.any():
            return
        bin_ids = bin_ids[valid].long()
        xy_err = xy_err[valid]
        ones = torch.ones(bin_ids.shape[0], dtype=torch.float64, device=self.device)
        stats[:, 0].scatter_add_(0, bin_ids, ones)
        stats[:, 1].scatter_add_(0, bin_ids, xy_err.square().sum(dim=-1).double())
        stats[:, 2].scatter_add_(0, bin_ids, xy_err[:, 0].double())
        stats[:, 3].scatter_add_(0, bin_ids, xy_err[:, 1].double())
        stats[:, 4].scatter_add_(0, bin_ids, xy_err[:, 0].square().double())
        stats[:, 5].scatter_add_(0, bin_ids, xy_err[:, 1].square().double())

    def _accumulate_label_diagnostics(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        foot_ids: torch.Tensor,
        steps_to_contact: torch.Tensor,
        family_ids: torch.Tensor,
    ) -> None:
        if inputs.shape[0] == 0 or self.trainer is None:
            return
        mean_xy, _mean_yaw, _sigma, _yaw_sigma = self.trainer.predict(
            inputs.to(dtype=torch.float32)
        )
        batch_ids = torch.arange(inputs.shape[0], device=self.device)
        xy_err = mean_xy[batch_ids, foot_ids] - targets[..., :2]
        self._accumulate_diag_stats(
            self._horizon_diag_stats, self._horizon_bin_ids(steps_to_contact), xy_err
        )
        family_bins = torch.full_like(family_ids, -1)
        family_bins = torch.where(family_ids == DISCRETE_FAMILY_ID, 0, family_bins)
        family_bins = torch.where(family_ids == STAIRS_DOWN_FAMILY_ID, 1, family_bins)
        family_bins = torch.where(family_ids == STAIRS_UP_FAMILY_ID, 2, family_bins)
        self._accumulate_diag_stats(self._family_diag_stats, family_bins, xy_err)

    def _diag_logs_from_stats(
        self, prefix: str, names: tuple[str, ...], stats: torch.Tensor
    ) -> dict[str, float]:
        logs: dict[str, float] = {}
        for bin_id, name in enumerate(names):
            count = float(stats[bin_id, 0].item())
            if count <= 0.0:
                continue
            logs[f"{prefix}/{name}/xy_rmse_m"] = (stats[bin_id, 1].item() / count) ** 0.5
            logs[f"{prefix}/{name}/x_bias_m"] = stats[bin_id, 2].item() / count
            logs[f"{prefix}/{name}/y_bias_m"] = stats[bin_id, 3].item() / count
            logs[f"{prefix}/{name}/x_rmse_m"] = (stats[bin_id, 4].item() / count) ** 0.5
            logs[f"{prefix}/{name}/y_rmse_m"] = (stats[bin_id, 5].item() / count) ** 0.5
        return logs

    @staticmethod
    def _histogram_quantile(histogram: torch.Tensor, quantile: float, maximum: float) -> float:
        count = histogram.sum()
        if count <= 0:
            return 0.0
        bin_id = int(torch.searchsorted(histogram.cumsum(0), count * quantile).item())
        return (bin_id + 0.5) * maximum / histogram.numel()

    def _contact_logs_from_stats(
        self, contact_stats: torch.Tensor, filter_stats: torch.Tensor
    ) -> dict[str, float]:
        logs: dict[str, float] = {}
        total_count = float(contact_stats[:, 0].sum().item())
        if total_count > 0.0:
            logs["Foothold/Contact/support_ratio"] = (
                contact_stats[:, 1].sum().item() / total_count
            )
            logs["Foothold/Contact/area_support_ratio"] = (
                contact_stats[:, 2].sum().item() / total_count
            )
            logs["Foothold/Contact/terrain_support_ratio"] = (
                contact_stats[:, 3].sum().item() / total_count
            )
            logs["Foothold/Contact/below_threshold_fraction"] = (
                contact_stats[:, 4].sum().item() / total_count
            )
        for bin_id, name in enumerate(_FAMILY_BIN_NAMES):
            count = float(contact_stats[bin_id, 0].item())
            if count > 0.0:
                prefix = f"Foothold/Contact/{name}"
                logs[f"{prefix}/support_ratio"] = contact_stats[bin_id, 1].item() / count
                logs[f"{prefix}/area_support_ratio"] = contact_stats[bin_id, 2].item() / count
                logs[f"{prefix}/terrain_support_ratio"] = contact_stats[bin_id, 3].item() / count
                logs[f"{prefix}/below_threshold_fraction"] = (
                    contact_stats[bin_id, 4].item() / count
                )
                logs[f"{prefix}/tilt_mean_rad"] = contact_stats[bin_id, 5].item() / count
                support_hist = contact_stats[
                    bin_id, 6 : 6 + _SUPPORT_HIST_BINS
                ]
                tilt_hist = contact_stats[bin_id, 6 + _SUPPORT_HIST_BINS :]
                logs[f"{prefix}/support_p10"] = self._histogram_quantile(
                    support_hist, 0.1, 1.0
                )
                logs[f"{prefix}/tilt_p90_rad"] = self._histogram_quantile(
                    tilt_hist, 0.9, 0.5 * torch.pi
                )
            pending = float(filter_stats[bin_id, 0].item())
            if pending > 0.0:
                prefix = f"Foothold/Touchdown/{name}"
                logs[f"{prefix}/rejected_vertical_fraction"] = (
                    filter_stats[bin_id, 1].item() / pending
                )
                logs[f"{prefix}/rejected_support_fraction"] = (
                    filter_stats[bin_id, 2].item() / pending
                )
                logs[f"{prefix}/accepted_fraction"] = filter_stats[bin_id, 3].item() / pending
        return logs

    def _finalize_contacts(
        self,
        enable_terrain_foot_weights: bool = True,
        stairs_weight_min: float = 0.1,
        stairs_weight_max: float = 1.0,
    ) -> None:
        first_contact = self._contact_sensor.compute_first_contact(self.env.step_dt)[
            :, self._contact_body_ids
        ]
        foot_pos_w = self._robot.data.body_pos_w[:, self._foot_body_ids]
        first_events = first_contact.nonzero(as_tuple=False)
        if first_events.numel() > 0:
            first_events = first_events[self._foothold_active()[first_events[:, 0]]]

        in_contact = self._in_contact_like_feet_at_plane()
        pending_events = ((self._pending_counts > 0) & in_contact).nonzero(
            as_tuple=False
        )
        if pending_events.numel() > 0:
            self._pending_contact_seen[
                pending_events[:, 0], pending_events[:, 1]
            ] = True
        if first_events.numel() == 0 and pending_events.numel() == 0:
            return

        support_events = torch.unique(
            torch.cat((first_events, pending_events), dim=0), dim=0
        )
        support_ratios, area_support, terrain_support = self._contact_support_components(
            support_events,
            foot_pos_w,
            enable_terrain_foot_weights=enable_terrain_foot_weights,
            stairs_weight_min=stairs_weight_min,
            stairs_weight_max=stairs_weight_max,
        )
        support_by_foot = torch.full(
            (self.num_envs, 2), float("nan"), device=self.device
        )
        support_by_foot[support_events[:, 0], support_events[:, 1]] = support_ratios
        area_by_foot = torch.full_like(support_by_foot, float("nan"))
        terrain_by_foot = torch.full_like(support_by_foot, float("nan"))
        area_by_foot[support_events[:, 0], support_events[:, 1]] = area_support
        terrain_by_foot[support_events[:, 0], support_events[:, 1]] = terrain_support

        if first_events.numel() > 0:
            first_env_ids = first_events[:, 0]
            first_foot_ids = first_events[:, 1]
            foot_quat = self._robot.data.body_quat_w[:, self._foot_body_ids][
                first_env_ids, first_foot_ids
            ]
            gravity_w = self._robot.data.GRAVITY_VEC_W
            gravity = (
                gravity_w.expand_as(foot_quat[..., :3])
                if gravity_w.ndim == 1
                else gravity_w[first_env_ids]
            )
            projected_gravity = quat_apply_inverse(foot_quat, gravity)
            tilt = torch.atan2(
                torch.linalg.vector_norm(projected_gravity[..., :2], dim=-1),
                -projected_gravity[..., 2],
            )
            family_ids = get_terrain_family_ids(self.env)[first_env_ids]
            self._accumulate_contact_support_stats(
                family_ids,
                support_by_foot[first_env_ids, first_foot_ids],
                area_by_foot[first_env_ids, first_foot_ids],
                terrain_by_foot[first_env_ids, first_foot_ids],
                tilt,
            )

        touchdown_events = pending_events
        if touchdown_events.numel() > 0:
            env_ids = touchdown_events[:, 0]
            foot_ids = touchdown_events[:, 1]
            forces = self._contact_sensor.data.net_forces_w[:, self._contact_body_ids]
            vertical_contact = vertical_contact_mask(
                forces[env_ids, foot_ids],
                float(self.support_cfg.touchdown_vertical_force_ratio),
                float(self.support_cfg.touchdown_vertical_force_min),
            )
            supported = support_by_foot[env_ids, foot_ids] >= float(
                self.support_cfg.touchdown_support_ratio_min
            )
            self._accumulate_touchdown_filter_stats(
                get_terrain_family_ids(self.env)[env_ids], vertical_contact, supported
            )
            touchdown_events = touchdown_events[vertical_contact & supported]
        if touchdown_events.numel() > 0:
            self._label_touchdowns(touchdown_events, foot_pos_w)
            self._pending_counts[touchdown_events[:, 0], touchdown_events[:, 1]] = 0
            self._pending_write[touchdown_events[:, 0], touchdown_events[:, 1]] = 0
            self._pending_contact_seen[
                touchdown_events[:, 0], touchdown_events[:, 1]
            ] = False

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
        """Sum of horizon-weighted per-swing-foot expected unsupported fractions."""
        self._ensure_scene_handles()
        self._finalize_contacts(
            enable_terrain_foot_weights=enable_terrain_foot_weights,
            stairs_weight_min=stairs_weight_min,
            stairs_weight_max=stairs_weight_max,
        )
        reward = torch.zeros(self.num_envs, device=self.device)
        if (
            not self._prepared
            or self.trainer is None
            or self._latest_mu_b is None
            or self._latest_yaw_w is None
            or self._latest_sigma is None
            or self._latest_yaw_sigma is None
            or not self.trainer.reward_enabled
        ):
            return reward
        if not self._terrain_ready_mask.any():
            return reward

        family_ids = get_terrain_family_ids(self.env)
        eligible_env = self._foothold_active() & self._prediction_valid
        weighted_deficiency = torch.zeros(self.num_envs, 2, device=self.device)
        swing_pairs = (self._swing_at_prepare & eligible_env.unsqueeze(-1)).nonzero(
            as_tuple=False
        )
        if swing_pairs.numel() > 0:
            env_ids = swing_pairs[:, 0]
            foot_ids = swing_pairs[:, 1]
            eval_sigma = self.trainer.grid.reward_sigma(self._latest_sigma[env_ids, foot_ids])
            candidate_xy, candidate_yaws, candidate_weights = self.trainer.grid.expectation_poses(
                self._latest_mu_b[env_ids, foot_ids],
                eval_sigma,
                self._latest_yaw_w[env_ids, foot_ids],
                self._latest_yaw_sigma[env_ids, foot_ids],
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
                reduction=self.trainer.grid.expectation_weighting,
                candidate_yaws_w=candidate_yaws,
            )
            horizon_weight = self._guidance_horizon_weight(
                self._swing_age_at_prepare[env_ids, foot_ids]
            )
            weighted_deficiency[env_ids, foot_ids] = predicted * horizon_weight
            self._swing_guidance_stats[0] += predicted.double().sum()
            self._swing_guidance_stats[1] += float(predicted.numel())
            self._swing_guidance_stats[2] += eval_sigma.double().sum()
            self._swing_guidance_stats[3] += float(eval_sigma.numel())
            self._swing_guidance_stats[4] += horizon_weight.double().sum()

        reward[eligible_env] = weighted_deficiency[eligible_env].sum(dim=-1)
        active_reward = reward[eligible_env]
        self._guidance_reward_stats[0] += active_reward.double().sum()
        self._guidance_reward_stats[1] += float(active_reward.numel())
        return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

    def reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self._pending_counts[env_ids] = 0
        self._pending_write[env_ids] = 0
        self._pending_contact_seen[env_ids] = False
        self._prediction_valid[env_ids] = False
        self._swing_at_prepare[env_ids] = False
        self._swing_age_at_prepare[env_ids] = 0
        for latest in (
            self._latest_mu_b,
            self._latest_yaw_b,
            self._latest_yaw_w,
            self._latest_sigma,
            self._latest_yaw_sigma,
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

        family_ids = get_terrain_family_ids(self.env)
        inv_stairs = family_ids == STAIRS_UP_FAMILY_ID
        inv_levels = self.env.scene.terrain.terrain_levels.float()
        rollout_stats = torch.tensor(
            (
                inv_levels[inv_stairs].sum().item() if bool(inv_stairs.any()) else 0.0,
                float(inv_stairs.sum().item()),
            ),
            dtype=torch.float64,
            device=self.device,
        )
        rollout_stats = torch.cat(
            (
                rollout_stats,
                self._guidance_reward_stats,
                self._swing_guidance_stats,
                self._contact_support_stats.reshape(-1),
                self._touchdown_filter_stats.reshape(-1),
                self._horizon_diag_stats.reshape(-1),
                self._family_diag_stats.reshape(-1),
            )
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(rollout_stats, op=torch.distributed.ReduceOp.SUM)
        (
            terrain_level_sum,
            environment_count,
            guidance_reward_sum,
            guidance_reward_count,
            swing_deficiency_sum,
            swing_deficiency_count,
            eval_sigma_sum,
            eval_sigma_count,
            horizon_weight_sum,
        ) = rollout_stats[:9].tolist()
        offset = 9
        contact_n = self._contact_support_stats.numel()
        filter_n = self._touchdown_filter_stats.numel()
        horizon_n = self._horizon_diag_stats.numel()
        family_n = self._family_diag_stats.numel()
        contact_stats = rollout_stats[offset : offset + contact_n].view_as(
            self._contact_support_stats
        )
        offset += contact_n
        filter_stats = rollout_stats[offset : offset + filter_n].view_as(
            self._touchdown_filter_stats
        )
        offset += filter_n
        horizon_stats = rollout_stats[offset : offset + horizon_n].view_as(self._horizon_diag_stats)
        family_stats = rollout_stats[offset + horizon_n : offset + horizon_n + family_n].view_as(
            self._family_diag_stats
        )

        inv_stairs_mean = (
            terrain_level_sum / environment_count if environment_count > 0.0 else 0.0
        )
        if inv_stairs_mean > self.cfg.curriculum_level_threshold:
            self._terrain_ready_mask.fill_(True)
        metrics["Foothold/Guidance/inv_stairs_level"] = inv_stairs_mean

        metrics.update(self._contact_logs_from_stats(contact_stats, filter_stats))
        # Eligible-terrain mean of the reward tensor (stance zeros included).
        # Swing-only signal is ``swing_deficiency``; Episode_Reward is further diluted.
        metrics["Foothold/Guidance/reward"] = (
            guidance_reward_sum / guidance_reward_count if guidance_reward_count > 0.0 else 0.0
        )
        if swing_deficiency_count > 0.0:
            metrics["Foothold/Guidance/swing_deficiency"] = (
                swing_deficiency_sum / swing_deficiency_count
            )
            metrics["Foothold/Guidance/horizon_weight"] = (
                horizon_weight_sum / swing_deficiency_count
            )
        if eval_sigma_count > 0.0:
            metrics["Foothold/Guidance/eval_sigma_m"] = eval_sigma_sum / eval_sigma_count
        metrics.update(self._diag_logs_from_stats("Foothold/Horizon", _HORIZON_BIN_NAMES, horizon_stats))
        metrics.update(self._diag_logs_from_stats("Foothold/Terrain", _FAMILY_BIN_NAMES, family_stats))

        self._contact_support_stats.zero_()
        self._touchdown_filter_stats.zero_()
        self._guidance_reward_stats.zero_()
        self._swing_guidance_stats.zero_()
        self._horizon_diag_stats.zero_()
        self._family_diag_stats.zero_()
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
    """Swing-foot expected unsupported fraction, stronger early in swing. Pair with a negative weight."""
    manager = getattr(env, "foothold_guidance", None)
    if manager is None:
        return torch.zeros(env.num_envs, device=env.device)
    return manager.compute_reward(
        enable_terrain_foot_weights=enable_terrain_foot_weights,
        stairs_weight_min=stairs_weight_min,
        stairs_weight_max=stairs_weight_max,
    )
