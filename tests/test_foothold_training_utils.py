# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


_MDP_DIR = (
    Path(__file__).parents[1]
    / "robolab"
    / "tasks"
    / "manager_based"
    / "parkour"
    / "mdp"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


contact_metrics = _load_module("contact_metrics", _MDP_DIR / "contact_metrics.py")
normalization = _load_module(
    "foothold_normalization", _MDP_DIR / "foothold_prediction" / "normalization.py"
)
replay_buffer = _load_module(
    "foothold_replay_buffer", _MDP_DIR / "foothold_prediction" / "replay_buffer.py"
)
terrain_family = _load_module("terrain_family", _MDP_DIR / "terrain_family.py")


def test_horizontal_force_excess_is_continuous_and_thresholded():
    forces = torch.tensor(
        [
            [10.0, 0.0, 100.0],
            [120.0, 0.0, 20.0],
            [220.0, 0.0, 20.0],
        ]
    )
    excess = contact_metrics.horizontal_force_excess(
        forces, horizontal_ratio=1.0, force_threshold=20.0
    )
    torch.testing.assert_close(excess, torch.tensor([0.0, 80.0, 180.0]))


def test_vertical_contact_rejects_horizontal_impact():
    forces = torch.tensor([[100.0, 0.0, 20.0], [10.0, 0.0, 100.0]])
    valid = contact_metrics.vertical_contact_mask(
        forces, vertical_force_ratio=1.0, vertical_force_min=5.0
    )
    assert valid.tolist() == [False, True]


def test_rp1_support_clearance_separates_perlin_noise_from_stair_drop():
    ankle_height = torch.tensor([[0.045]])
    terrain_heights = torch.tensor([[0.0, -0.03, -0.05]])
    unsupported = terrain_family.soft_absolute_clearance_unsupported(
        ankle_height,
        terrain_heights,
        height_offset=0.045,
        height_tolerance=0.030,
        transition_width=0.005,
        unsupported_only=True,
    )

    # Flat ground and the 3 cm Perlin envelope stay inside the support band;
    # the 5 cm stair drop remains clearly unsupported.
    expected_stair_deficiency = torch.sigmoid(torch.tensor(4.0)) / 3.0
    torch.testing.assert_close(unsupported, expected_stair_deficiency.unsqueeze(0))


def test_highest_plane_support_exposes_stair_drop_despite_low_ankle():
    terrain_heights = torch.tensor([[0.0, 0.0, -0.05, -0.05]])
    final, area, terrain = terrain_family.highest_plane_support_deficiency(
        terrain_heights,
        height_tolerance=0.030,
        transition_width=0.005,
        unsupported_only=True,
    )

    expected = torch.sigmoid(torch.tensor(4.0)) / 2.0
    torch.testing.assert_close(final, expected.unsqueeze(0))
    torch.testing.assert_close(area, final)
    torch.testing.assert_close(terrain, final)


def test_highest_plane_support_rejects_missing_terrain():
    deficiencies = terrain_family.highest_plane_support_deficiency(
        torch.full((1, 4), torch.inf), unsupported_only=True
    )
    for value in deficiencies:
        torch.testing.assert_close(value, torch.ones(1))


def test_running_normalization_and_checkpoint_round_trip():
    normalizer = normalization.RunningMeanStd((2,), "cpu")
    values = torch.tensor([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])
    normalizer.update(values)

    normalized = normalizer(values)
    torch.testing.assert_close(normalized.mean(dim=0), torch.zeros(2), atol=1.0e-6, rtol=0.0)

    restored = normalization.RunningMeanStd((2,), "cpu")
    restored.load_state_dict(normalizer.state_dict())
    torch.testing.assert_close(restored(values), normalized)


def test_balanced_replay_sampling_covers_all_foothold_family_buckets():
    torch.manual_seed(7)
    buffer = replay_buffer.FootholdReplayBuffer(4800, 1, "cpu")
    horizon_steps = (1, 5, 10, 20)
    for family_offset, family_id in enumerate((1, 2, 3)):
        for foot_id in (0, 1):
            for horizon_id, steps in enumerate(horizon_steps):
                bucket_id = (family_offset * 2 + foot_id) * 4 + horizon_id
                count = 200
                buffer.append(
                    torch.full((count, 1), float(bucket_id)),
                    torch.zeros(count, 3),
                    torch.full((count,), foot_id),
                    torch.full((count,), steps),
                    torch.full((count,), family_id),
                )

    inputs, _, _, _ = buffer.sample(
        384, balanced=True, candidate_multiplier=8
    )
    counts = torch.bincount(inputs[:, 0].long(), minlength=24)
    assert torch.all(counts >= 4)
    assert torch.all(counts <= 36)


def test_discrete_terrain_prioritizes_mid_sole_support():
    local_x = torch.tensor([-0.13, 0.0, 0.13])
    family_ids = torch.tensor([terrain_family.DISCRETE_FAMILY_ID])
    weights = terrain_family.terrain_foot_point_weights(
        local_x, family_ids, stairs_weight_min=0.1, stairs_weight_max=1.0
    )
    torch.testing.assert_close(weights, torch.tensor([[0.1, 1.0, 0.1]]))


def test_non_foothold_terrain_uses_uniform_sole_weights():
    local_x = torch.tensor([-0.13, 0.0, 0.13])
    family_ids = torch.tensor(
        [terrain_family.PERLIN_ROUGH_FAMILY_ID, terrain_family.SLOPE_FAMILY_ID]
    )
    weights = terrain_family.terrain_foot_point_weights(
        local_x, family_ids, stairs_weight_min=0.1, stairs_weight_max=1.0
    )
    torch.testing.assert_close(weights, torch.ones(2, 3))


def test_direction_weights_cannot_hide_unsupported_area():
    foot_z = torch.tensor([[0.0]])
    terrain_z = torch.tensor([[-0.05, 0.0]])
    point_weights = torch.tensor([[0.1, 1.0]])
    final, area, terrain = terrain_family.conservative_support_deficiency(
        foot_z,
        terrain_z,
        height_offset=0.0,
        height_tolerance=0.035,
        transition_width=0.005,
        point_weights=point_weights,
        unsupported_only=True,
    )
    assert terrain.item() < area.item()
    torch.testing.assert_close(final, area)
