"""Foothold predictor optimization and loss-based reward gating."""

from __future__ import annotations

import copy

import torch
import torch.optim as optim

from .config import FootholdPredictorCfg
from .grid import FootholdGaussianGeometry
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
        self.grid = FootholdGaussianGeometry(cfg.grid, device)
        self.model = FootholdPredictor(input_dim, cfg.hidden_dims).to(device)
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
        """Discard replay samples without touching model weights."""
        self.train_buffer.clear()

    @torch.no_grad()
    def predict(
        self, predictor_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Gaussian means ``[N,2,2]`` and sigmas ``[N,2]``."""
        self.ema_model.eval()
        return self.grid.decode_distribution(self.ema_model(predictor_input))

    def _losses(
        self,
        model: FootholdPredictor,
        predictor_input: torch.Tensor,
        target: torch.Tensor,
        foot_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw_distribution = model(predictor_input)
        pred_xy, pred_sigma = self.grid.decode_distribution(raw_distribution)
        batch_ids = torch.arange(predictor_input.shape[0], device=self.device)
        pred_foot = pred_xy[batch_ids, foot_ids]
        sigma_foot = pred_sigma[batch_ids, foot_ids]

        squared_error = (pred_foot - target).square().sum(dim=-1)
        per_sample = (
            squared_error / (2.0 * sigma_foot.square())
            + 2.0 * torch.log(sigma_foot)
        )
        finite = torch.isfinite(target).all(dim=-1)
        nll = per_sample[finite].mean() if finite.any() else pred_foot.sum() * 0.0

        prediction_loss = float(self.cfg.nll_loss_coef) * nll
        valid_squared_error = squared_error[finite]
        valid_sigma = sigma_foot[finite]
        xy_rmse = valid_squared_error.mean().sqrt()
        normalized_squared_error = valid_squared_error / valid_sigma.square()
        metrics = {
            "nll": nll.detach(),
            "xy_rmse": xy_rmse.detach(),
            "sigma_mean": valid_sigma.mean().detach(),
            "calibration_error": (
                normalized_squared_error.mean() - 2.0
            ).abs().detach(),
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
            packed = torch.tensor(
                (
                    prediction_loss,
                    metric_values["nll"],
                    metric_values["xy_rmse"],
                    metric_values["sigma_mean"],
                    metric_values["calibration_error"],
                ),
                dtype=torch.float64,
                device=self.device,
            )
            torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
            packed /= torch.distributed.get_world_size()
            values = packed.tolist()
            prediction_loss = values[0]
            metric_values = {
                "nll": values[1],
                "xy_rmse": values[2],
                "sigma_mean": values[3],
                "calibration_error": values[4],
            }

        xy_rmse = metric_values["xy_rmse"]
        if (
            not self.reward_enabled
            and xy_rmse < self.cfg.enable_xy_rmse_threshold
        ):
            self.reward_enabled = True

        return {
            "Foothold/Loss/prediction_loss": prediction_loss,
            "Foothold/Loss/nll": metric_values["nll"],
            "Foothold/Accuracy/xy_rmse_m": metric_values["xy_rmse"],
            "Foothold/Distribution/sigma_mean_m": metric_values["sigma_mean"],
            "Foothold/Distribution/calibration_error": metric_values[
                "calibration_error"
            ],
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
                "the Gaussian predictor must be trained from scratch."
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
