# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Configure TerrainImporter to load a single OBJ/STL mesh terrain."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import robolab.terrains as terrain_gen
import trimesh

from robolab.terrains import TerrainImporterCfg
from robolab.terrains.terrain_generator_cfg import FiledTerrainGeneratorCfg


@dataclass(frozen=True)
class MeshXYFootprint:
    """Rasterized XY footprint of a terrain mesh."""

    origin: tuple[float, float]
    resolution: float
    grid: np.ndarray
    width: int
    height: int


def _erode_binary_mask(mask: np.ndarray, radius_pixels: int) -> np.ndarray:
    """Erode a 2D boolean mask by ``radius_pixels`` using 4-neighbor erosion."""
    if radius_pixels <= 0:
        return mask
    eroded = mask.copy()
    for _ in range(radius_pixels):
        eroded[1:, :] &= eroded[:-1, :]
        eroded[:-1, :] &= eroded[1:, :]
        eroded[:, 1:] &= eroded[:, :-1]
        eroded[:, :-1] &= eroded[:, 1:]
    return eroded


def build_mesh_xy_footprint(
    mesh_path: str,
    *,
    resolution: float = 0.02,
    distance_buffer: float = 0.1,
) -> MeshXYFootprint:
    """Rasterize the terrain mesh XY footprint with downward ray casting.

    Each grid cell is marked valid when a vertical ray from above intersects the mesh.
    ``distance_buffer`` shrinks the valid region inward, matching ``terrain_out_of_bounds``.
    """
    mesh_path = os.path.abspath(mesh_path)
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(f"Terrain mesh not found: {mesh_path}")

    mesh = trimesh.load(mesh_path, force="mesh")
    low = mesh.bounds[0, :2]
    high = mesh.bounds[1, :2]
    width = int(np.ceil((high[0] - low[0]) / resolution)) + 1
    height = int(np.ceil((high[1] - low[1]) / resolution)) + 1
    xs = low[0] + (np.arange(width) + 0.5) * resolution
    ys = low[1] + (np.arange(height) + 0.5) * resolution
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    ray_origin_z = float(mesh.bounds[1, 2] + 1.0)
    origins = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, ray_origin_z)])
    directions = np.tile(np.array([0.0, 0.0, -1.0]), (origins.shape[0], 1))
    _, hit_ray_indices, _ = mesh.ray.intersects_location(origins, directions)

    on_mesh = np.zeros(origins.shape[0], dtype=bool)
    on_mesh[hit_ray_indices] = True
    grid = on_mesh.reshape(height, width)

    erosion_radius = max(0, int(np.ceil(distance_buffer / resolution)))
    grid = _erode_binary_mask(grid, erosion_radius)

    return MeshXYFootprint(
        origin=(float(low[0]), float(low[1])),
        resolution=resolution,
        grid=grid,
        width=width,
        height=height,
    )


def infer_terrain_size_from_mesh(obj_path: str, padding: float = 0.5) -> tuple[float, float]:
    """Infer generator cell size from mesh AABB."""
    mesh = trimesh.load(os.path.abspath(obj_path), force="mesh")
    span = mesh.bounds[1, :2] - mesh.bounds[0, :2]
    return float(span[0] + padding), float(span[1] + padding)


def import_obj_as_terrain(
    prim_path: str,
    obj_path: str,
    *,
    physics_material=None,
    visual_material=None,
) -> trimesh.Trimesh:
    """Spawn a mesh terrain so OBJ vertex coordinates match Isaac world coordinates."""
    import isaaclab.sim as sim_utils
    from isaaclab.terrains.utils import create_prim_from_mesh

    obj_path = os.path.abspath(obj_path)
    if not os.path.isfile(obj_path):
        raise FileNotFoundError(f"Terrain mesh not found: {obj_path}")

    mesh = trimesh.load(obj_path, force="mesh")
    if physics_material is None:
        physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )

    create_prim_from_mesh(
        prim_path,
        mesh,
        physics_material=physics_material,
        visual_material=visual_material,
    )
    low, high = mesh.bounds
    print(f"[INFO] Terrain mesh: {obj_path}")
    print(f"[INFO] Prim path   : {prim_path}")
    print(f"[INFO] Bounds low  : ({low[0]:.3f}, {low[1]:.3f}, {low[2]:.3f})")
    print(f"[INFO] Bounds high : ({high[0]:.3f}, {high[1]:.3f}, {high[2]:.3f})")
    return mesh


def configure_single_obj_terrain(
    terrain_cfg: TerrainImporterCfg,
    obj_path: str,
    *,
    size: tuple[float, float] | None = None,
    direct: bool = True,
) -> None:
    """Configure *terrain_cfg* to load one mesh file via TerrainGenerator.

    Args:
        terrain_cfg: Scene terrain config to modify in-place.
        obj_path: Path to ``.obj`` / ``.stl`` terrain mesh.
        size: Generator cell size. If ``None``, inferred from mesh bounds.
        direct: If ``True``, preserve OBJ world coordinates (StaticMeshTerrainCfg).
            If ``False``, use MotionMatched crop/shift (AMP training convention).
    """
    obj_path = os.path.abspath(obj_path)
    if not os.path.isfile(obj_path):
        raise FileNotFoundError(f"Terrain mesh not found: {obj_path}")

    if size is None:
        size = infer_terrain_size_from_mesh(obj_path)

    terrain_dir = os.path.dirname(obj_path)
    obj_name = os.path.basename(obj_path)

    if direct:
        terrain_cfg.terrain_type = "generator"
        terrain_cfg.terrain_generator = FiledTerrainGeneratorCfg(
            size=size,
            num_rows=1,
            num_cols=1,
            curriculum=False,
            use_cache=False,
            border_width=0.0,
            sub_terrains={
                "obj_terrain": terrain_gen.StaticMeshTerrainCfg(
                    proportion=1.0,
                    path=terrain_dir,
                    mesh_file=obj_name,
                ),
            },
        )
    else:
        metadata_yaml = os.path.join(terrain_dir, "_terrain_metadata.yaml")
        import yaml

        with open(metadata_yaml, "w", encoding="utf-8") as file:
            yaml.safe_dump(
                {"terrains": [{"terrain_id": "terrain", "terrain_file": obj_name}]},
                file,
                sort_keys=False,
            )
        terrain_cfg.terrain_type = "generator"
        terrain_cfg.terrain_generator = FiledTerrainGeneratorCfg(
            size=size,
            num_rows=1,
            num_cols=1,
            curriculum=False,
            use_cache=False,
            border_width=0.0,
            sub_terrains={
                "obj_terrain": terrain_gen.MotionMatchedTerrainCfg(
                    proportion=1.0,
                    path=terrain_dir,
                    metadata_yaml=metadata_yaml,
                ),
            },
        )

    terrain_cfg.use_terrain_origins = False
    terrain_cfg.debug_vis = False
