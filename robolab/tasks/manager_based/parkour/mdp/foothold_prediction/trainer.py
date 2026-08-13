"""Foothold predictor optimization and RMSE reward gating."""

from __future__ import annotations

import copy

import torch
import torch.optim as optim

from isaaclab.utils.math import wrap_to_pi

from .config import FootholdPredictorCfg
from .grid import FootholdGaussianGeometry
from .model import FootholdPredictor
from .replay_buffer import FootholdReplayBuffer


def empty_predictor_logs(reward_enabled: bool = False) -> dict[str, float]:
    """Zeroed foothold logs used before the trainer exists or replay is ready."""
    return {
        "Foothold/Loss/nll": 0.0,
        "Foothold/Accuracy/xy_rmse_m": 0.0,
        "Foothold/Accuracy/yaw_rmse_rad": 0.0,
        "Foothold/Distribution/sigma_mean_m": 0.0,
        "Foothold/Distribution/yaw_sigma_mean_rad": 0.0,
        "Foothold/Guidance/reward_enabled": float(reward_enabled),
    }


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
        self.grid = FootholdGaussianGeometry(cfg.grid, device)
        if not cfg.hidden_dims:
            raise ValueError("FootholdPredictorCfg.hidden_dims is required.")
        self.model = FootholdPredictor(input_dim, list(cfg.hidden_dims)).to(device)
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

    def clear_train_buffer(self) -> None:
        self.train_buffer.clear()

    @torch.no_grad()
    def predict(
        self, predictor_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return mean XY ``[N,2,2]``, mean yaw ``[N,2]``, and XY sigma ``[N,2]``."""
        self.ema_model.eval()
        mean_xy, mean_yaw, sigma_xy, _sigma_yaw = self.grid.decode_distribution(
            self.ema_model(predictor_input)
        )
        return mean_xy, mean_yaw, sigma_xy

    def _losses(
        self,
        model: FootholdPredictor,
        predictor_input: torch.Tensor,
        target: torch.Tensor,
        foot_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred_xy, pred_yaw, pred_sigma_xy, pred_sigma_yaw = self.grid.decode_distribution(
            model(predictor_input)
        )
        batch_ids = torch.arange(predictor_input.shape[0], device=self.device)
        xy_foot = pred_xy[batch_ids, foot_ids]
        yaw_foot = pred_yaw[batch_ids, foot_ids]
        sigma_xy = pred_sigma_xy[batch_ids, foot_ids]
        sigma_yaw = pred_sigma_yaw[batch_ids, foot_ids]

        xy_squared_error = (xy_foot - target[..., :2]).square().sum(dim=-1)
        yaw_squared_error = wrap_to_pi(yaw_foot - target[..., 2]).square()
        per_sample = (
            xy_squared_error / (2.0 * sigma_xy.square())
            + 2.0 * torch.log(sigma_xy)
            + yaw_squared_error / (2.0 * sigma_yaw.square())
            + torch.log(sigma_yaw)
        )
        finite = torch.isfinite(target).all(dim=-1)
        nll = per_sample[finite].mean() if finite.any() else xy_foot.sum() * 0.0

        valid_xy = xy_squared_error[finite]
        valid_yaw = yaw_squared_error[finite]
        valid_sigma_xy = sigma_xy[finite]
        valid_sigma_yaw = sigma_yaw[finite]
        zero = xy_foot.new_zeros(())
        metrics = {
            "nll": nll.detach(),
            "xy_rmse": (valid_xy.mean().sqrt() if valid_xy.numel() else zero).detach(),
            "yaw_rmse": (valid_yaw.mean().sqrt() if valid_yaw.numel() else zero).detach(),
            "sigma_mean": (
                valid_sigma_xy.mean() if valid_sigma_xy.numel() else zero
            ).detach(),
            "yaw_sigma_mean": (
                valid_sigma_yaw.mean() if valid_sigma_yaw.numel() else zero
            ).detach(),
        }
        return float(self.cfg.nll_loss_coef) * nll, metrics

    @torch.no_grad()
    def _update_ema_model(self) -> None:
        for ema_parameter, parameter in zip(
            self.ema_model.parameters(), self.model.parameters()
        ):
            ema_parameter.mul_(self.cfg.ema_decay).add_(
                parameter, alpha=1.0 - self.cfg.ema_decay
            )

    def idle_metrics(self) -> dict[str, float]:
        return empty_predictor_logs(self.reward_enabled)

    def update(self) -> dict[str, float]:
        locally_ready = self.train_buffer.size >= self.cfg.min_train_samples
        globally_ready = locally_ready
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            ready = torch.tensor(int(locally_ready), device=self.device)
            torch.distributed.all_reduce(ready, op=torch.distributed.ReduceOp.MIN)
            globally_ready = bool(ready.item())

        if not globally_ready:
            return self.idle_metrics()

        self.model.train()
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

        with torch.no_grad():
            inputs, targets, foot_ids = self.train_buffer.sample(self.cfg.batch_size)
            _, metrics = self._losses(self.ema_model, inputs, targets, foot_ids)
            metric_values = {key: float(value.item()) for key, value in metrics.items()}

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            packed = torch.tensor(
                (
                    metric_values["nll"],
                    metric_values["xy_rmse"],
                    metric_values["yaw_rmse"],
                    metric_values["sigma_mean"],
                    metric_values["yaw_sigma_mean"],
                ),
                dtype=torch.float64,
                device=self.device,
            )
            torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
            packed /= torch.distributed.get_world_size()
            values = packed.tolist()
            metric_values = {
                "nll": values[0],
                "xy_rmse": values[1],
                "yaw_rmse": values[2],
                "sigma_mean": values[3],
                "yaw_sigma_mean": values[4],
            }

        if (
            not self.reward_enabled
            and metric_values["xy_rmse"] < self.cfg.enable_xy_rmse_threshold
            and metric_values["yaw_rmse"] < self.cfg.enable_yaw_rmse_threshold
        ):
            self.reward_enabled = True

        return {
            "Foothold/Loss/nll": metric_values["nll"],
            "Foothold/Accuracy/xy_rmse_m": metric_values["xy_rmse"],
            "Foothold/Accuracy/yaw_rmse_rad": metric_values["yaw_rmse"],
            "Foothold/Distribution/sigma_mean_m": metric_values["sigma_mean"],
            "Foothold/Distribution/yaw_sigma_mean_rad": metric_values["yaw_sigma_mean"],
            "Foothold/Guidance/reward_enabled": float(self.reward_enabled),
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
                "train the Gaussian predictor from scratch."
            )
        if int(state["input_dim"]) != self.input_dim:
            raise RuntimeError(
                f"Foothold predictor input changed from {state['input_dim']} to {self.input_dim}."
            )
        if bool(state.get("shared", False)):
            raise RuntimeError(
                "Checkpoint used a shared critic foothold head; train the standalone "
                "predictor from scratch."
            )
        self.model.load_state_dict(state["model"])
        self.ema_model.load_state_dict(state["ema_model"])
        if load_optimizer and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        self.reward_enabled = bool(state.get("reward_enabled", False))
