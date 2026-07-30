"""Foothold predictor optimization and loss-based reward gating."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
import torch.optim as optim

from .config import FootholdPredictorCfg
from .grid import ReachableFootholdGrid
from .model import FootholdPredictor
from .replay_buffer import FootholdReplayBuffer


class FootholdPredictorTrainer:
    def __init__(
        self,
        input_dim: int,
        cfg: FootholdPredictorCfg,
        device: torch.device | str,
    ) -> None:
        self.input_dim = input_dim
        self.cfg = cfg
        self.device = device
        self.grid = ReachableFootholdGrid(cfg.grid, device)
        self.model = FootholdPredictor(
            input_dim, cfg.hidden_dims, self.grid.num_cells
        ).to(device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for parameter in self.model.parameters():
                torch.distributed.broadcast(parameter.data, src=0)
        self.ema_model = copy.deepcopy(self.model).to(device)
        self.ema_model.eval()
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.train_buffer = FootholdReplayBuffer(
            cfg.train_buffer_capacity, input_dim, device
        )
        self.reward_enabled = False

    def add_samples(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        foot_ids: torch.Tensor,
    ) -> None:
        self.train_buffer.append(inputs, targets, foot_ids)

    @torch.no_grad()
    def predict(
        self, predictor_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.ema_model.eval()
        logits, raw_residuals = self.ema_model(predictor_input)
        return (
            self.grid.probabilities(logits),
            self.grid.bounded_residuals(raw_residuals),
        )

    def _neighbor_support_mask(
        self,
        target_indices: torch.Tensor,
        foot_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return neighbor cell ids [B, 5] and valid∩reachable mask."""
        neighbor_ids = self.grid.neighbor_indices[target_indices]
        neighbor_valid = self.grid.neighbor_valid[target_indices]
        reachable = self.grid.reachable_mask[foot_ids].gather(1, neighbor_ids)
        return neighbor_ids, neighbor_valid & reachable

    def _classification_loss(
        self,
        logits: torch.Tensor,
        target_indices: torch.Tensor,
        foot_ids: torch.Tensor,
        target_in_reach: torch.Tensor,
    ) -> torch.Tensor:
        """Hard CE, or soft CE over GT + reachable 4-neighbors."""
        zero = logits.sum() * 0.0
        neighbor_weight = float(getattr(self.cfg, "ce_neighbor_weight", 0.0))
        in_reach_only = bool(getattr(self.cfg, "classify_in_reach_only", True))

        if neighbor_weight <= 0.0:
            per_sample = F.cross_entropy(logits, target_indices, reduction="none")
        else:
            neighbor_ids, support = self._neighbor_support_mask(
                target_indices, foot_ids
            )
            is_gt = neighbor_ids == target_indices.unsqueeze(1)
            soft = torch.where(
                is_gt,
                torch.ones_like(neighbor_ids, dtype=logits.dtype),
                logits.new_full(neighbor_ids.shape, neighbor_weight),
            )
            soft = soft * support.to(logits.dtype)
            soft_sum = soft.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(logits.dtype).eps
            )
            soft = soft / soft_sum
            log_probs = F.log_softmax(logits, dim=-1).gather(1, neighbor_ids)
            per_sample = -(soft * log_probs).sum(dim=-1)

        if in_reach_only:
            if not target_in_reach.any():
                return zero
            return per_sample[target_in_reach].mean()
        return per_sample.mean()

    def _neighbor_residual_loss(
        self,
        residuals: torch.Tensor,
        target: torch.Tensor,
        target_indices: torch.Tensor,
        foot_ids: torch.Tensor,
        target_in_reach: torch.Tensor,
    ) -> torch.Tensor:
        """Supervise residual on the GT cell and its reachable 4-neighbors."""
        delta = self.grid.max_residual
        zero = residuals.sum() * 0.0
        if not target_in_reach.any() or delta <= 0.0:
            return zero

        neighbor_ids, support = self._neighbor_support_mask(target_indices, foot_ids)
        is_gt = neighbor_ids == target_indices.unsqueeze(1)
        neighbor_weight = float(getattr(self.cfg, "residual_neighbor_weight", 0.0))
        weights = torch.where(
            is_gt,
            torch.ones_like(neighbor_ids, dtype=residuals.dtype),
            residuals.new_full(neighbor_ids.shape, neighbor_weight),
        )
        weights = weights * support.to(residuals.dtype)
        weights = weights * target_in_reach.to(residuals.dtype).unsqueeze(1)
        weight_sum = weights.sum()
        if weight_sum <= 0.0:
            return zero

        centers = self.grid.points[neighbor_ids]
        gathered = torch.gather(
            residuals,
            1,
            neighbor_ids.unsqueeze(-1).expand(-1, -1, 2),
        )
        target_residuals = (target.unsqueeze(1) - centers).clamp(-delta, delta)
        per_cell = F.smooth_l1_loss(
            gathered / delta,
            target_residuals / delta,
            reduction="none",
        ).mean(dim=-1)
        return (weights * per_cell).sum() / weight_sum

    def _losses(
        self,
        model: FootholdPredictor,
        predictor_input: torch.Tensor,
        target: torch.Tensor,
        foot_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits_all, raw_residuals_all = model(predictor_input)
        logits_all = self.grid.masked_logits(logits_all)
        residuals_all = self.grid.bounded_residuals(raw_residuals_all)

        batch_ids = torch.arange(predictor_input.shape[0], device=self.device)
        logits = logits_all[batch_ids, foot_ids]
        residuals = residuals_all[batch_ids, foot_ids]
        target_indices, target_in_reach = self.grid.target_indices(target, foot_ids)

        classification_loss = self._classification_loss(
            logits, target_indices, foot_ids, target_in_reach
        )
        residual_loss = self._neighbor_residual_loss(
            residuals, target, target_indices, foot_ids, target_in_reach
        )
        ce_coef = float(getattr(self.cfg, "classification_loss_coef", 1.0))
        prediction_loss = (
            ce_coef * classification_loss + self.cfg.residual_loss_coef * residual_loss
        )

        probabilities = torch.softmax(logits, dim=-1)
        mode_indices = probabilities.argmax(dim=-1)
        mode_xy = self.grid.points[mode_indices] + residuals[batch_ids, mode_indices]
        xy_rmse = (mode_xy - target).square().sum(dim=-1).mean().sqrt()

        metrics = {
            "xy_rmse": xy_rmse.detach(),
            "top1_acc": (mode_indices == target_indices).float().mean().detach(),
            "out_of_reach_ratio": (~target_in_reach).float().mean().detach(),
        }
        return prediction_loss, metrics

    @torch.no_grad()
    def _update_ema_model(self) -> None:
        for ema_parameter, parameter in zip(
            self.ema_model.parameters(), self.model.parameters()
        ):
            ema_parameter.mul_(self.cfg.ema_decay).add_(
                parameter, alpha=1.0 - self.cfg.ema_decay
            )

    def update(self) -> dict[str, float]:
        locally_ready = self.train_buffer.size >= self.cfg.min_train_samples
        globally_ready = locally_ready
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            ready = torch.tensor(int(locally_ready), device=self.device)
            torch.distributed.all_reduce(ready, op=torch.distributed.ReduceOp.MIN)
            globally_ready = bool(ready.item())

        if not globally_ready:
            return {}

        self.model.train()
        optimization_losses: list[float] = []
        for _ in range(self.cfg.updates_per_iteration):
            inputs, targets, foot_ids = self.train_buffer.sample(self.cfg.batch_size)
            loss, _ = self._losses(self.model, inputs, targets, foot_ids)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        torch.distributed.all_reduce(
                            parameter.grad, op=torch.distributed.ReduceOp.SUM
                        )
                        parameter.grad.div_(world_size)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self._update_ema_model()
            optimization_losses.append(float(loss.detach().item()))

        prediction_loss = sum(optimization_losses) / len(optimization_losses)
        with torch.no_grad():
            inputs, targets, foot_ids = self.train_buffer.sample(self.cfg.batch_size)
            _, metrics = self._losses(self.ema_model, inputs, targets, foot_ids)
            metric_values = {
                key: float(value.item()) for key, value in metrics.items()
            }

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            keys = (
                "prediction_loss",
                "xy_rmse",
                "top1_acc",
                "out_of_reach_ratio",
            )
            packed = torch.tensor(
                (
                    prediction_loss,
                    metric_values["xy_rmse"],
                    metric_values["top1_acc"],
                    metric_values["out_of_reach_ratio"],
                ),
                dtype=torch.float64,
                device=self.device,
            )
            torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
            packed /= torch.distributed.get_world_size()
            values = packed.tolist()
            prediction_loss = values[0]
            metric_values = dict(zip(keys[1:], values[1:]))

        xy_rmse = metric_values["xy_rmse"]
        if not self.reward_enabled and xy_rmse < self.cfg.enable_xy_rmse_threshold:
            self.reward_enabled = True

        return {
            "Foothold/Loss/prediction_loss": prediction_loss,
            "Foothold/Accuracy/xy_rmse_m": xy_rmse,
            "Foothold/Accuracy/top1_acc": metric_values["top1_acc"],
            "Foothold/Data/out_of_reach_ratio": metric_values["out_of_reach_ratio"],
        }

    def state_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "grid_signature": self.grid.signature,
            "reward_enabled": self.reward_enabled,
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict, load_optimizer: bool = True) -> None:
        if state.get("grid_signature") != self.grid.signature:
            raise RuntimeError(
                "Foothold predictor checkpoint uses a different output distribution; "
                "the reachable-grid predictor must be trained from scratch."
            )
        if int(state["input_dim"]) != self.input_dim:
            raise RuntimeError(
                f"Foothold predictor input changed from {state['input_dim']} to {self.input_dim}."
            )
        self.model.load_state_dict(state["model"])
        self.ema_model.load_state_dict(state["ema_model"])
        if load_optimizer and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        self.reward_enabled = bool(state.get("reward_enabled", False))
