# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize an OBJ/STL terrain mesh in Isaac Sim.

.. code-block:: bash

    ./isaaclab.sh -p robolab/scripts/tools/visualize_obj_terrain.py
    ./isaaclab.sh -p robolab/scripts/tools/visualize_obj_terrain.py \\
        --terrain_obj robolab/data/motions/rpo_bm/largebox_cleaned_simplified.obj
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize an OBJ/STL terrain mesh in Isaac Sim.")
parser.add_argument(
    "--terrain_obj",
    type=str,
    default="robolab/data/motions/rpo_bm/beyond_reverse_vault_003_aug001_dm_aug8_terrain.obj",
    help="Path to the terrain mesh (.obj / .stl).",
)
parser.add_argument(
    "--prim_path",
    type=str,
    default="/World/ground",
    help="USD prim path for the spawned terrain mesh.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sim import SimulationContext
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from robolab.utils.obj_terrain import import_obj_as_terrain


def _spawn_world_origin_marker(sim: SimulationContext, *, scale: float) -> VisualizationMarkers:
    """Spawn an XYZ frame at the Isaac world origin (0, 0, 0), not the mesh centroid."""
    marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/WorldOrigin")
    marker_cfg.markers["frame"].scale = (scale, scale, scale)
    marker = VisualizationMarkers(marker_cfg)
    origin = torch.zeros(1, 3, device=sim.device)
    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim.device)
    marker.visualize(translations=origin, orientations=identity_quat)
    return marker


def _print_mesh_vs_world_origin(mesh) -> None:
    low, high = mesh.bounds
    print("[INFO] World origin (Isaac): (0.000, 0.000, 0.000)")
    print(f"[INFO] Mesh bounds low : ({low[0]:.3f}, {low[1]:.3f}, {low[2]:.3f})")
    print(f"[INFO] Mesh bounds high: ({high[0]:.3f}, {high[1]:.3f}, {high[2]:.3f})")
    contains_origin = bool(np.all(low <= 0.0) and np.all(high >= 0.0))
    print(f"[INFO] World origin inside mesh AABB: {contains_origin}")


def _frame_camera_on_mesh(sim: SimulationContext, mesh) -> None:
    """Point the viewport at the mesh center with a distance scaled to its size."""
    center = mesh.bounds.mean(axis=0)
    span = float((mesh.bounds[1] - mesh.bounds[0]).max())
    offset = max(span * 1.8, 0.5)
    eye = center + np.array([offset, offset, offset * 0.6])
    sim.set_camera_view(eye, center)


def main() -> None:
    obj_path = os.path.abspath(args_cli.terrain_obj)
    if not os.path.isfile(obj_path):
        raise FileNotFoundError(f"Terrain mesh not found: {obj_path}")

    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))

    dome_light_cfg = sim_utils.DomeLightCfg(
        intensity=750.0,
        texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    )
    dome_light_cfg.func("/World/skyLight", dome_light_cfg)

    mesh = import_obj_as_terrain(args_cli.prim_path, obj_path)
    _print_mesh_vs_world_origin(mesh)
    _frame_camera_on_mesh(sim, mesh)

    sim.reset()

    span = float((mesh.bounds[1] - mesh.bounds[0]).max())
    _spawn_world_origin_marker(sim, scale=max(span * 0.08, 0.03))
    print("[INFO]: Terrain loaded. RGB axes at /Visuals/WorldOrigin mark world (0, 0, 0).")
    print("[INFO]: Close the Isaac window to exit.")

    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
