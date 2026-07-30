"""GPU replay storage for foothold prediction."""

from __future__ import annotations

import torch


class FootholdReplayBuffer:
    def __init__(self, capacity: int, input_dim: int, device: torch.device | str) -> None:
        self.inputs = torch.zeros(capacity, input_dim, dtype=torch.float16, device=device)
        self.targets = torch.zeros(capacity, 2, dtype=torch.float32, device=device)
        self.foot_ids = torch.zeros(capacity, dtype=torch.long, device=device)
        self.capacity = capacity
        self.write_index = 0
        self.size = 0

    def append(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        foot_ids: torch.Tensor,
    ) -> None:
        count = inputs.shape[0]
        if count == 0:
            return
        if count > self.capacity:
            inputs = inputs[-self.capacity :]
            targets = targets[-self.capacity :]
            foot_ids = foot_ids[-self.capacity :]
            count = self.capacity

        first_count = min(count, self.capacity - self.write_index)
        first_slice = slice(self.write_index, self.write_index + first_count)
        self.inputs[first_slice] = inputs[:first_count].to(torch.float16)
        self.targets[first_slice] = targets[:first_count]
        self.foot_ids[first_slice] = foot_ids[:first_count]

        remaining = count - first_count
        if remaining > 0:
            self.inputs[:remaining] = inputs[first_count:].to(torch.float16)
            self.targets[:remaining] = targets[first_count:]
            self.foot_ids[:remaining] = foot_ids[first_count:]

        self.write_index = (self.write_index + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = min(batch_size, self.size)
        indices = torch.randint(self.size, (batch_size,), device=self.inputs.device)
        return (
            self.inputs[indices].to(torch.float32),
            self.targets[indices],
            self.foot_ids[indices],
        )
