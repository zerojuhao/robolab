# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize parkour terrains defined in terrain_generator_cfg.py.

Example usage:

.. code-block:: bash

    ./isaaclab.sh -p robolab/robolab/tasks/manager_based/parkour/visualize_terrain.py
    ./isaaclab.sh -p robolab/robolab/tasks/manager_based/parkour/visualize_terrain.py --sub_terrain trapezoid_stairs
    ./isaaclab.sh -p robolab/robolab/tasks/manager_based/parkour/visualize_terrain.py --color_scheme height --use_curriculum
    ./isaaclab.sh -p robolab/robolab/tasks/manager_based/parkour/visualize_terrain.py --tick_spacing 0.5
    ./isaaclab.sh -p robolab/robolab/tasks/manager_based/parkour/visualize_terrain.py --no_debug_overlay
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize parkour terrain_generator_cfg terrains.")
parser.add_argument(
    "--color_scheme",
    type=str,
    default="none",
    choices=["height", "random", "none"],
    help="Color scheme for terrain meshes.",
)
parser.add_argument("--use_curriculum", action="store_true", default=False, help="Enable terrain curriculum.")
parser.add_argument(
    "--sub_terrain",
    type=str,
    default=None,
    help="Visualize only this sub-terrain name (e.g. trapezoid_stairs).",
)
parser.add_argument("--num_rows", type=int, default=2, help="Number of terrain rows.")
parser.add_argument("--num_cols", type=int, default=2, help="Number of terrain columns.")
parser.add_argument("--tick_spacing", type=float, default=1.0, help="Grid tick spacing in meters (local frame).")
parser.add_argument(
    "--no_debug_overlay",
    action="store_true",
    default=False,
    help="Disable XY grid ticks and terrain-name overlay.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
from pxr import Gf, UsdGeom, Vt

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from robolab.terrains import TerrainImporter, TerrainImporterCfg
from robolab.terrains.terrain_generator import FiledTerrainGenerator

from robolab.tasks.manager_based.parkour.terrain_generator_cfg import ROUGH_TERRAINS_CFG

# Grid overlay colors: border, minor lines, +X axis, +Y axis.
_COLOR_BORDER = (0.85, 0.85, 0.85)
_COLOR_MINOR = (0.45, 0.45, 0.45)
_COLOR_AXIS_X = (0.95, 0.25, 0.25)
_COLOR_AXIS_Y = (0.25, 0.85, 0.25)


class _LineBatch:
    """Collects USD basis-curve segments and commits them as one prim."""

    def __init__(self) -> None:
        self.points: list[Gf.Vec3f] = []
        self.counts: list[int] = []
        self.colors: list[Gf.Vec3f] = []
        self.widths: list[float] = []

    def add(
        self,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        color: tuple[float, float, float],
        width: float,
    ) -> None:
        self.points.extend([Gf.Vec3f(*start), Gf.Vec3f(*end)])
        self.counts.append(2)
        self.colors.extend([Gf.Vec3f(*color), Gf.Vec3f(*color)])
        self.widths.extend([width, width])

    def commit(self, stage, prim_path: str) -> None:
        if not self.points:
            return
        curves = UsdGeom.BasisCurves.Define(stage, prim_path)
        curves.CreateTypeAttr(UsdGeom.Tokens.linear)
        curves.CreateBasisAttr(UsdGeom.Tokens.bspline)
        curves.CreatePointsAttr(Vt.Vec3fArray(self.points))
        curves.CreateCurveVertexCountsAttr(self.counts)
        curves.CreateWidthsAttr(Vt.FloatArray(self.widths))
        curves.CreateDisplayColorAttr(Vt.Vec3fArray(self.colors))


def _tick_values(half_extent: float, spacing: float) -> np.ndarray:
    return np.arange(-half_extent, half_extent + spacing * 0.5, spacing)


def _global_grid_z(vertices: np.ndarray, *, wall_height: float = 5.0, clearance: float = 0.05) -> float:
    """Single overlay height for all tiles so adjacent grid lines stay coplanar."""
    z_vals = vertices[:, 2]
    walkable = z_vals[z_vals < wall_height - 0.25]
    if walkable.size == 0:
        walkable = z_vals
    return float(walkable.max()) + clearance


def _spawn_tile_grid(
    stage,
    prim_root: str,
    ox: float,
    oy: float,
    half_x: float,
    half_y: float,
    z: float,
    tick_spacing: float,
) -> None:
    """Draw border, minor grid lines, and axis highlights for one tile."""
    lines = _LineBatch()
    corners = [
        (ox - half_x, oy - half_y, z),
        (ox + half_x, oy - half_y, z),
        (ox + half_x, oy + half_y, z),
        (ox - half_x, oy + half_y, z),
    ]
    for i in range(4):
        lines.add(corners[i], corners[(i + 1) % 4], _COLOR_BORDER, 0.03)

    for local_x in _tick_values(half_x, tick_spacing):
        wx = ox + local_x
        on_axis = abs(local_x) < 1e-6
        color = _COLOR_AXIS_X if on_axis else _COLOR_MINOR
        width = 0.05 if on_axis else 0.02
        lines.add((wx, oy - half_y, z), (wx, oy + half_y, z), color, width)

    for local_y in _tick_values(half_y, tick_spacing):
        wy = oy + local_y
        on_axis = abs(local_y) < 1e-6
        color = _COLOR_AXIS_Y if on_axis else _COLOR_MINOR
        width = 0.05 if on_axis else 0.02
        lines.add((ox - half_x, wy, z), (ox + half_x, wy, z), color, width)

    lines.commit(stage, f"{prim_root}/grid")


def spawn_debug_overlay(terrain_importer, tick_spacing: float = 1.0) -> None:
    """Draw per-tile local XY grids. Local origin = terrain_origins[row, col]."""
    terrain_gen = terrain_importer.terrain_generator
    if not isinstance(terrain_gen, FiledTerrainGenerator) or terrain_gen.subterrain_index_grid is None:
        raise RuntimeError("FiledTerrainGenerator with subterrain_index_grid is required for debug overlay.")

    origins = terrain_importer.terrain_origins
    if origins is None:
        origins = terrain_gen.terrain_origins
    if origins is None:
        raise RuntimeError("terrain_origins is not available.")
    origins = origins.detach().cpu().numpy() if hasattr(origins, "detach") else np.asarray(origins)

    cfg = terrain_gen.cfg
    half_x, half_y = cfg.size[0] * 0.5, cfg.size[1] * 0.5
    sub_names = list(cfg.sub_terrains.keys())
    vertices = np.asarray(terrain_gen.terrain_mesh.vertices)
    stage = sim_utils.get_current_stage()
    grid_z = _global_grid_z(vertices)

    print("[INFO] Terrain debug overlay (local frame origin = terrain_origins[row, col]):")
    print(f"[INFO]   tile size = ({cfg.size[0]:.2f}, {cfg.size[1]:.2f}) m, tick_spacing = {tick_spacing:.2f} m")
    print(f"[INFO]   local x in [-{half_x:.2f}, {half_x:.2f}], local y in [-{half_y:.2f}, {half_y:.2f}]")
    print(f"[INFO]   grid z = {grid_z:.2f} m (shared across all tiles)")
    print("[INFO]   Red = local +X, Green = local +Y")

    for row in range(cfg.num_rows):
        for col in range(cfg.num_cols):
            ox, oy, oz = map(float, origins[row, col])

            sub_index = int(terrain_gen.subterrain_index_grid[row, col])
            terrain_name = sub_names[sub_index]
            print(
                f"[INFO]   ({row:02d},{col:02d}) {terrain_name:28s} "
                f"world_origin=({ox:7.2f}, {oy:7.2f}, {oz:6.2f})"
            )
            _spawn_tile_grid(
                stage,
                f"/Visuals/TerrainDebug/r{row:02d}_c{col:02d}",
                ox,
                oy,
                half_x,
                half_y,
                grid_z,
                tick_spacing,
            )


def design_scene():
    """Design the scene."""
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    terrain_gen_cfg = ROUGH_TERRAINS_CFG.replace(
        curriculum=args_cli.use_curriculum,
        color_scheme=args_cli.color_scheme,
        num_rows=args_cli.num_rows,
        num_cols=args_cli.num_cols,
    )
    if args_cli.sub_terrain is not None:
        if args_cli.sub_terrain not in terrain_gen_cfg.sub_terrains:
            available = ", ".join(sorted(terrain_gen_cfg.sub_terrains.keys()))
            raise ValueError(f"Unknown sub-terrain '{args_cli.sub_terrain}'. Available: {available}")
        sub_cfg = terrain_gen_cfg.sub_terrains[args_cli.sub_terrain]
        terrain_gen_cfg.sub_terrains = {args_cli.sub_terrain: sub_cfg.replace(proportion=1.0)}

    terrain_importer_cfg = TerrainImporterCfg(
        num_envs=args_cli.num_rows * args_cli.num_cols,
        env_spacing=3.0,
        prim_path="/World/ground",
        max_init_terrain_level=None,
        terrain_type="generator",
        terrain_generator=terrain_gen_cfg,
        debug_vis=True,
    )
    if args_cli.color_scheme in ["height", "random"]:
        terrain_importer_cfg.visual_material = None
    else:
        terrain_importer_cfg.visual_material = sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        )

    return TerrainImporter(terrain_importer_cfg)


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[12.0, 12.0, 8.0], target=[0.0, 0.0, 0.0])

    terrain_importer = design_scene()
    sim.reset()
    if not args_cli.no_debug_overlay:
        spawn_debug_overlay(terrain_importer, tick_spacing=args_cli.tick_spacing)
    print("[INFO]: Parkour terrain visualization ready.")
    print(f"[INFO]: Sub-terrains: {list(terrain_importer.terrain_generator.cfg.sub_terrains.keys())}")

    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
