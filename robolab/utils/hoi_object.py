# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn HOI rigid objects from OBJ/STL mesh files."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import MISSING

import trimesh

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.meshes import meshes
from isaaclab.sim.spawners.meshes.meshes_cfg import MeshCfg
from isaaclab.sim.utils import clone, get_current_stage
from isaaclab.utils import configclass


@clone
def spawn_mesh_from_obj_file(
    prim_path: str,
    cfg: "MeshObjFileCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    obj_path = os.path.abspath(cfg.obj_path)
    if not os.path.isfile(obj_path):
        raise FileNotFoundError(f"Object mesh not found: {obj_path}")
    mesh = trimesh.load(obj_path, force="mesh")
    stage = get_current_stage()
    meshes._spawn_mesh_geom_from_mesh(prim_path, cfg, mesh, translation, orientation, stage=stage, **kwargs)
    return stage.GetPrimAtPath(prim_path)


@configclass
class MeshObjFileCfg(MeshCfg):
    func: Callable = spawn_mesh_from_obj_file
    obj_path: str = MISSING


def make_hoi_object_cfg(
    object_mesh: str,
    *,
    prim_path: str = "{ENV_REGEX_NS}/box_object",
    default_mass: float = 1.0,
    kinematic: bool = False,
    disable_gravity: bool | None = None,
    diffuse_color: tuple[float, float, float] = (0.8, 0.6, 0.4),
) -> RigidObjectCfg:
    """Build a mesh-based HOI rigid object with collision enabled."""
    obj_path = os.path.abspath(object_mesh)
    if not os.path.isfile(obj_path):
        raise FileNotFoundError(f"Object mesh not found: {obj_path}")
    if disable_gravity is None:
        disable_gravity = kinematic
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=MeshObjFileCfg(
            obj_path=obj_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=disable_gravity,
                kinematic_enabled=kinematic,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=default_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=diffuse_color),
        ),
    )
