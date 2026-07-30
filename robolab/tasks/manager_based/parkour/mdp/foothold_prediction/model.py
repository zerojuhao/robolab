"""Foothold prediction network."""

from __future__ import annotations

import torch
import torch.nn as nn


class FootholdPredictor(nn.Module):
    """MLP with per-foot grid logits and cell-local XY residuals."""

    def __init__(
        self, input_dim: int, hidden_dims: list[int], num_grid_cells: int
    ) -> None:
        super().__init__()
        self.num_grid_cells = num_grid_cells
        dims = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend((nn.Linear(in_dim, out_dim), nn.ELU()))
        layers.append(nn.Linear(dims[-1], 2 * num_grid_cells * 3))
        self.network = nn.Sequential(*layers)

    def forward(
        self, predictor_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.network(predictor_input).view(-1, 2, self.num_grid_cells, 3)
        return output[..., 0], output[..., 1:]
