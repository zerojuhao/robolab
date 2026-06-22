# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# MuJoCo sim2sim for RPO-Perceptive policies (depth_encoder.onnx + actor.onnx).
#
# Policy observations (``perceptive_env_cfg.PolicyCfg``):
#   command, base_ang_vel, projected_gravity, joint_pos, joint_vel, actions
#   each stacked ``robot_config.frame_stack`` times (order matches PolicyCfg).
#   depth_image -> ``delayed_visualizable_image`` (see ``robot_config`` depth fields).

from __future__ import annotations

import argparse
import os
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import cv2
import matplotlib.pyplot as plt
import mujoco
import mujoco_viewer
import numpy as np
from pynput import keyboard
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from robolab.assets import ISAAC_DATA_DIR

try:
    import onnxruntime as ort
except ImportError as e:
    ort = None
    _ORT_IMPORT_ERROR = e
else:
    _ORT_IMPORT_ERROR = None


def proprio_obs_dim(frame_stack: int, num_actions: int) -> int:
    """Flat proprio size: command (2×num_actions) + 5 terms per frame stack."""
    num_command = 2 * num_actions
    return (num_command + 3 + 3 + num_actions + num_actions + num_actions) * frame_stack


def build_motion_command(
    m_joint_pos: np.ndarray,
    m_joint_vel: np.ndarray,
    idx: int,
) -> np.ndarray:
    """Motion command in Isaac Lab order (raw NPZ, same as ``sim2sim_rpo_bm`` ``m_input``)."""
    return np.concatenate((m_joint_pos[idx, :], m_joint_vel[idx, :]), axis=0).astype(np.float32)


def sample_depth_history(
    depth_buf: list[np.ndarray],
    *,
    history_len: int,
    num_output_frames: int,
    history_skip_frames: int,
    encoder_h: int,
    encoder_w: int,
    delay_frames: int = 0,
) -> np.ndarray:
    """Sample depth frames for the policy encoder (matches ``delayed_visualizable_image``)."""
    h, w = encoder_h, encoder_w
    if not depth_buf:
        return np.zeros((num_output_frames, h, w), dtype=np.float32)
    buf = list(depth_buf)
    if len(buf) < history_len:
        buf = [buf[0]] * (history_len - len(buf)) + buf
    stacked = np.stack(buf[-history_len:], axis=0)
    frame_offset = np.arange(0, num_output_frames * history_skip_frames, history_skip_frames)[::-1]
    latest_idx = len(stacked) - 1 - delay_frames
    indices = np.clip(latest_idx - frame_offset, 0, len(stacked) - 1)
    return stacked[indices].astype(np.float32)


def get_obs(data: mujoco.MjData, model: mujoco.MjModel):
    """Articulation observation from MuJoCo."""
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double)
    rot = R.from_quat(quat)
    v = rot.apply(data.qvel[:3], inverse=True).astype(np.double)
    omega = data.sensor("angular-velocity").data.astype(np.double)
    gvec = rot.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    return q, dq, quat, v, omega, gvec


def pd_control(target_q, q, target_dq, dq, kp, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def mj_macro_step(model: mujoco.MjModel, data: mujoco.MjData, *, n_sub: int) -> None:
    for _ in range(max(1, n_sub)):
        mujoco.mj_step(model, data)


def open_interactive_viewer(model: mujoco.MjModel, data: mujoco.MjData):
    viewer = mujoco_viewer.MujocoViewer(model, data, mode="window", width=1920, height=1080)
    viewer.cam.distance = 4.0
    viewer.cam.azimuth = 45.0
    viewer.cam.elevation = -20.0
    viewer.cam.lookat = [0, 0, 1]
    return viewer


def update_follow_camera(cam: mujoco.MjvCamera, data: mujoco.MjData, model: mujoco.MjModel) -> None:
    del model
    cam.lookat[:] = data.qpos[0:3].astype(np.float64)


def configure_depth_rendering(model: mujoco.MjModel, dc) -> None:
    model.vis.map.znear = float(dc.depth_znear_scale)
    if hasattr(model.vis.map, "zfar"):
        model.vis.map.zfar = float(dc.depth_zfar_scale)


def _rot_mat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    return R.from_quat([x, y, z, w]).as_matrix()


def get_depth_camera_pose(data: mujoco.MjData, model: mujoco.MjModel, torso_body_name: str, dc) -> dict[str, np.ndarray]:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, torso_body_name)
    if bid < 0:
        raise RuntimeError(f"Body not found: {torso_body_name}")
    xmat = data.xmat[bid].reshape(3, 3)
    xpos = data.xpos[bid].copy()
    r_off = _rot_mat_wxyz(dc.camera_offset_quat_wxyz)
    r_world = (xmat @ r_off).astype(np.float64)
    eye = (xpos + xmat @ dc.camera_offset_pos_body).astype(np.float64)
    forward = r_world[:, 0].copy()
    fn = np.linalg.norm(forward)
    if fn < 1e-9:
        forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        forward /= fn
    return {"eye": eye, "forward": forward, "r_world": r_world}


def pinhole_ray_dirs_cam(height: int, width: int, fovy_deg: float, fovx_deg: float) -> np.ndarray:
    fovy = np.radians(fovy_deg)
    fovx = np.radians(fovx_deg)
    u = (np.arange(width, dtype=np.float64) + 0.5) / width * 2.0 - 1.0
    v = (np.arange(height, dtype=np.float64) + 0.5) / height * 2.0 - 1.0
    uu, vv = np.meshgrid(u, v)
    x = np.ones_like(uu)
    y = -uu * np.tan(fovx / 2.0)
    z = -vv * np.tan(fovy / 2.0)
    dirs = np.stack([x, y, z], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    return dirs


def capture_depth_mj_ray(
    data: mujoco.MjData,
    model: mujoco.MjModel,
    torso_body_name: str,
    vertical_fov_deg: float,
    horizontal_fov_deg: float,
    dc,
) -> np.ndarray:
    pose = get_depth_camera_pose(data, model, torso_body_name, dc)
    eye = pose["eye"]
    r_world = pose["r_world"]
    dirs_cam = pinhole_ray_dirs_cam(dc.raw_depth_h, dc.raw_depth_w, vertical_fov_deg, horizontal_fov_deg)
    dirs_world = dirs_cam @ r_world.T
    depth = np.full((dc.raw_depth_h, dc.raw_depth_w), dc.depth_ray_max, dtype=np.float64)
    geomid = np.zeros(1, dtype=np.int32)
    for j in range(dc.raw_depth_h):
        for i in range(dc.raw_depth_w):
            vec = dirs_world[j, i]
            dist = mujoco.mj_ray(model, data, eye, vec, None, 1, -1, geomid)
            if dist is not None and dist >= 0.0:
                depth[j, i] = min(float(dist) * float(dirs_cam[j, i, 0]), dc.depth_ray_max)
    return depth


def capture_raycaster_depth(
    depth_renderer: Any | None,
    depth_cam: mujoco.MjvCamera | None,
    data: mujoco.MjData,
    model: mujoco.MjModel,
    torso_body_name: str,
    vertical_fov_deg: float,
    horizontal_fov_deg: float,
    dc,
) -> np.ndarray:
    del depth_renderer, depth_cam
    if dc.depth_backend == "ray":
        return capture_depth_mj_ray(data, model, torso_body_name, vertical_fov_deg, horizontal_fov_deg, dc)
    raise RuntimeError(f"Unsupported depth backend: {dc.depth_backend}")


def raw_depth_to_metric_grid(depth_hw: np.ndarray, model: mujoco.MjModel, dc) -> np.ndarray:
    del model
    z = depth_hw.astype(np.float64)
    if z.shape[0] != dc.raw_depth_h or z.shape[1] != dc.raw_depth_w:
        z = cv2.resize(z, (dc.raw_depth_w, dc.raw_depth_h), interpolation=cv2.INTER_AREA)
    return z


def crop_depth(depth_hw: np.ndarray, crop_region: tuple[int, int, int, int]) -> np.ndarray:
    up, down, left, right = crop_region
    h, w = depth_hw.shape[:2]
    return depth_hw[up : h - down, left : w - right].copy()


def preprocess_depth_frame_from_lowres(z_low: np.ndarray, dc) -> np.ndarray:
    z = np.nan_to_num(z_low, nan=dc.depth_clip[1], posinf=dc.depth_clip[1], neginf=dc.depth_clip[0])
    z = crop_depth(z, dc.crop_region)
    if z.shape != (dc.encoder_h, dc.encoder_w):
        z = cv2.resize(z, (dc.encoder_w, dc.encoder_h), interpolation=cv2.INTER_AREA)
    z = cv2.GaussianBlur(z.astype(np.float64), (3, 3), 1.0)
    z = np.clip(z, dc.depth_clip[0], dc.depth_clip[1])
    z = (z - dc.depth_clip[0]) / (dc.depth_clip[1] - dc.depth_clip[0])
    return z.astype(np.float32)


def _draw_encoder_crop_roi_on_bgr(bgr: np.ndarray, scale: int, dc) -> None:
    up, down, left, right = dc.crop_region
    h, w = dc.raw_depth_h, dc.raw_depth_w
    fac = float(scale)
    x1 = int(round(left * fac))
    y1 = int(round(up * fac))
    x2 = int(round((w - right) * fac)) - 1
    y2 = int(round((h - down) * fac)) - 1
    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 255), max(1, scale // 2))


def show_depth_camera_side_by_side(
    metric_lowres_hw: np.ndarray,
    policy_normalized_hw: np.ndarray,
    window_name: str,
    scale: int,
    dc,
) -> None:
    tw, th = dc.raw_depth_w * scale, dc.raw_depth_h * scale
    p = np.clip(policy_normalized_hw, 0.0, 1.0)
    enc_bgr = cv2.cvtColor((p * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    enc_bgr = cv2.resize(enc_bgr, (tw, th), interpolation=cv2.INTER_NEAREST)
    d = np.clip(metric_lowres_hw, dc.depth_clip[0], dc.depth_clip[1])
    metric_bgr = cv2.cvtColor((d / dc.depth_clip[1] * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    metric_bgr = cv2.resize(metric_bgr, (tw, th), interpolation=cv2.INTER_NEAREST)
    _draw_encoder_crop_roi_on_bgr(metric_bgr, scale, dc)
    cv2.imshow(window_name, np.hstack([metric_bgr, enc_bgr]))
    cv2.waitKey(1)


class TermHistory:
    """Isaac CircularBuffer semantics: maxlen ring, iteration oldest->newest."""

    def __init__(self, max_len: int, term_dim: int):
        self.max_len = max_len
        self.term_dim = term_dim
        self._dq: deque[np.ndarray] = deque(maxlen=max_len)

    def reset(self) -> None:
        self._dq.clear()

    def append(self, x: np.ndarray) -> None:
        self._dq.append(np.asarray(x, dtype=np.float32).reshape(-1))

    def fill_tile(self, x: np.ndarray) -> None:
        self.reset()
        v = np.asarray(x, dtype=np.float32).reshape(-1)
        for _ in range(self.max_len):
            self._dq.append(v.copy())

    def flat(self) -> np.ndarray:
        if len(self._dq) == 0:
            return np.zeros(self.max_len * self.term_dim, dtype=np.float32)
        return np.concatenate(list(self._dq), axis=0)


def build_onnx_sessions(
    depth_encoder_path: str, actor_path: str, providers: list[str] | None = None
) -> tuple[Any, Any]:
    if ort is None:
        raise RuntimeError("onnxruntime is required for this script") from _ORT_IMPORT_ERROR
    so = ort.SessionOptions()
    so.log_severity_level = 3
    prov = providers or ort.get_available_providers()
    enc = ort.InferenceSession(depth_encoder_path, so, providers=prov)
    act = ort.InferenceSession(actor_path, so, providers=prov)
    return enc, act


@dataclass
class TerrainHField:
    """MuJoCo height-field sampled from a training OBJ mesh."""

    png_path: str
    nrow: int
    ncol: int
    size: list[float]  # [x_half, y_half, z_range, z_base]
    center: tuple[float, float, float]


def infer_terrain_mesh_from_motion(motion_file: str) -> str | None:
    """``foo.npz`` -> ``foo_terrain.obj`` when the file exists."""
    motion_file = os.path.abspath(motion_file)
    if motion_file.endswith(".npz"):
        candidate = motion_file[: -len(".npz")] + "_terrain.obj"
    else:
        candidate = f"{motion_file}_terrain.obj"
    return candidate if os.path.isfile(candidate) else None


def obj_mesh_to_hfield(obj_path: str, *, res: float = 0.03, work_dir: str) -> TerrainHField:
    """Ray-cast an OBJ height mesh into a MuJoCo hfield PNG (Isaac world coordinates)."""
    import trimesh
    from PIL import Image

    obj_path = os.path.abspath(obj_path)
    if not os.path.isfile(obj_path):
        raise FileNotFoundError(f"Terrain mesh not found: {obj_path}")

    mesh = trimesh.load(obj_path, force="mesh")
    min_x, min_y, min_z = mesh.bounds[0]
    max_x, max_y, max_z = mesh.bounds[1]

    x_vals = np.arange(min_x, max_x, res)
    y_vals = np.arange(min_y, max_y, res)
    grid_x, grid_y = np.meshgrid(x_vals, y_vals)
    z_high = max_z + 1.0
    ray_origins = np.column_stack([grid_x.flatten(), grid_y.flatten(), np.full(grid_x.size, z_high)])
    ray_directions = np.tile([0.0, 0.0, -1.0], (grid_x.size, 1))
    locations, index_ray, _ = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
    )

    hfield_flat = np.full(grid_x.size, min_z, dtype=np.float64)
    if len(locations) > 0:
        np.maximum.at(hfield_flat, index_ray, locations[:, 2])

    hfield_data = np.flipud(hfield_flat.reshape(grid_y.shape))
    h_min = float(hfield_data.min())
    h_max = float(hfield_data.max())
    h_range = h_max - h_min
    if h_range < 1e-6:
        hfield_norm = np.zeros_like(hfield_data)
    else:
        hfield_norm = (hfield_data - h_min) / h_range

    hfield_uint16 = (hfield_norm * 65535.0).astype(np.uint16)
    nrow, ncol = hfield_uint16.shape
    png_path = os.path.join(work_dir, "motion_terrain_hfield.png")
    Image.fromarray(hfield_uint16).save(png_path)

    x_half = float((max_x - min_x) * 0.5)
    y_half = float((max_y - min_y) * 0.5)
    center = (float(min_x + x_half), float(min_y + y_half), 0.0)
    return TerrainHField(
        png_path=png_path,
        nrow=nrow,
        ncol=ncol,
        size=[x_half, y_half, h_range, abs(h_min)],
        center=center,
    )


def build_mujoco_model(
    base_xml_path: str,
    terrain_mesh: str | None = None,
    *,
    hfield_res: float = 0.03,
) -> tuple[mujoco.MjModel, tempfile.TemporaryDirectory[str] | None]:
    """Load robot MJCF and optionally replace the default ground with a motion-matched hfield."""
    if terrain_mesh is None:
        return mujoco.MjModel.from_xml_path(base_xml_path), None

    work_dir = tempfile.TemporaryDirectory(prefix="sim2sim_terrain_")
    hfield = obj_mesh_to_hfield(terrain_mesh, res=hfield_res, work_dir=work_dir.name)

    spec = mujoco.MjSpec.from_file(base_xml_path)
    for geom in list(spec.geoms):
        if geom.name == "ground":
            geom.delete()

    spec.add_hfield(
        name="motion_terrain",
        file=hfield.png_path,
        nrow=hfield.nrow,
        ncol=hfield.ncol,
        size=hfield.size,
    )
    spec.worldbody.add_geom(
        name="motion_terrain",
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname="motion_terrain",
        pos=list(hfield.center),
        friction=[1.0, 0.2, 0.2],
        condim=3,
        contype=1,
        conaffinity=15,
    )
    print(f"[INFO] Motion terrain: {os.path.abspath(terrain_mesh)}")
    print(
        f"[INFO] HField grid: {hfield.nrow}x{hfield.ncol}, "
        f"size=({hfield.size[0]:.3f}, {hfield.size[1]:.3f}, {hfield.size[2]:.3f}, {hfield.size[3]:.3f}), "
        f"center=({hfield.center[0]:.3f}, {hfield.center[1]:.3f}, {hfield.center[2]:.3f})"
    )
    return spec.compile(), work_dir


class cmd:
    camera_follow = True
    reset_requested = False
    loop_motion = False

    @classmethod
    def toggle_camera_follow(cls) -> None:
        cls.camera_follow = not cls.camera_follow
        print(f"Camera follow: {cls.camera_follow}")

    @classmethod
    def toggle_loop(cls) -> None:
        cls.loop_motion = not cls.loop_motion
        print(f"Motion loop: {cls.loop_motion}")


def print_controls_guide() -> None:
    print("=" * 60)
    print("Keyboard: F = toggle camera follow | 0 = reset | L = loop motion")
    print("=" * 60)


def on_press(key):
    try:
        if not hasattr(key, "char") or key.char is None:
            return
        c = key.char.lower()
        if c == "f":
            cmd.toggle_camera_follow()
        elif c == "0":
            cmd.reset_requested = True
        elif c == "l":
            cmd.toggle_loop()
    except AttributeError:
        pass


def start_keyboard_listener():
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def frame_idx(t: int, num_frames: int, loop: bool) -> int:
    if num_frames <= 0:
        return 0
    if loop:
        return t % num_frames
    return min(t, num_frames - 1)


def compute_root_z_lift(
    body_pos_w: np.ndarray,
    idx: int,
    *,
    min_clearance: float = 0.03,
    terrain_mesh_path: str | None = None,
) -> float:
    """Raise root +Z so motion link frames clear z=0 and the terrain mesh at ``idx``."""
    min_z = float(body_pos_w[..., 2].min())
    z_lift = max(0.0, min_clearance - min_z)

    if terrain_mesh_path is None or not os.path.isfile(terrain_mesh_path):
        return z_lift

    import trimesh

    mesh = trimesh.load(os.path.abspath(terrain_mesh_path), force="mesh")
    ray_z = float(mesh.bounds[1, 2]) + 2.0
    max_deficit = 0.0
    for link_pos in body_pos_w[idx]:
        x, y, z_link = link_pos
        z_link += z_lift
        loc, _, _ = mesh.ray.intersects_location(
            ray_origins=np.array([[x, y, ray_z]], dtype=np.float64),
            ray_directions=np.array([[0.0, 0.0, -1.0]], dtype=np.float64),
        )
        if len(loc) == 0:
            continue
        terrain_z = float(loc[0, 2])
        max_deficit = max(max_deficit, (terrain_z + min_clearance) - z_link)
    return z_lift + max(0.0, max_deficit)


def apply_motion_root_pose(
    data: mujoco.MjData,
    motion_pos: np.ndarray,
    motion_quat: np.ndarray,
    idx: int,
    *,
    z_lift: float = 0.0,
) -> None:
    """Set free-joint pose from motion ``body_pos_w`` / ``body_quat_w`` (body index 0 = base)."""
    data.qpos[0:3] = motion_pos[idx, 0, :]
    data.qpos[2] += z_lift
    data.qpos[3:7] = motion_quat[idx, 0, :]


def apply_motion_initial_state(
    data: mujoco.MjData,
    cfg,
    motion_pos: np.ndarray,
    motion_quat: np.ndarray,
    m_joint_pos: np.ndarray,
    m_joint_vel: np.ndarray,
    m_body_lin_vel: np.ndarray,
    m_body_ang_vel: np.ndarray,
    idx: int,
    *,
    z_lift: float = 0.0,
) -> None:
    """Match Isaac ``MotionCommand._resample_command``: root + joint state from reference motion."""
    apply_motion_root_pose(data, motion_pos, motion_quat, idx, z_lift=z_lift)
    n = cfg.robot_config.num_actions
    # Motion joints are Isaac order; MuJoCo qpos is URDF order (``usd2urdf`` maps Isaac idx -> URDF idx).
    for i, mj_idx in enumerate(cfg.robot_config.usd2urdf):
        data.qpos[-n + mj_idx] = m_joint_pos[idx, i]
        data.qvel[-n + mj_idx] = m_joint_vel[idx, i]
    data.qvel[0:3] = m_body_lin_vel[idx, 0, :]
    data.qvel[3:6] = m_body_ang_vel[idx, 0, :]


def run_mujoco_onnx(
    depth_encoder: Any,
    actor: Any,
    cfg,
    motion_file: str,
    *,
    headless: bool = False,
    loop_motion: bool = False,
    show_depth_vis: bool = True,
    depth_vis_scale: int = 6,
    debug_obs: bool = False,
    quiet: bool = True,
    zero_latent: bool = False,
) -> None:
    enc_in_name = depth_encoder.get_inputs()[0].name
    act_in_name = actor.get_inputs()[0].name

    motion = np.load(motion_file)
    motion_pos = motion["body_pos_w"]
    motion_quat = motion["body_quat_w"]
    m_joint_pos = motion["joint_pos"]
    m_joint_vel = motion["joint_vel"]
    m_body_lin_vel = motion["body_lin_vel_w"]
    m_body_ang_vel = motion["body_ang_vel_w"]
    num_frames = min(
        m_joint_pos.shape[0],
        m_joint_vel.shape[0],
        motion_pos.shape[0],
        motion_quat.shape[0],
        m_body_lin_vel.shape[0],
        m_body_ang_vel.shape[0],
    )

    cmd.loop_motion = loop_motion
    keyboard_listener = start_keyboard_listener()
    print_controls_guide()
    if zero_latent:
        print("[INFO] --zero_latent: actor_in = [proprio, zeros(128)] (depth encoder skipped for actor)")

    dc = cfg.depth_config

    if cfg.sim_config.mujoco_model is not None:
        model = cfg.sim_config.mujoco_model
    else:
        model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    configure_depth_rendering(model, dc)
    xml_dt = float(model.opt.timestep)
    macro_dt = float(cfg.sim_config.dt)
    phys_substeps = max(1, int(round(macro_dt / xml_dt)))
    model.opt.timestep = macro_dt / float(phys_substeps)
    model.opt.iterations = max(int(model.opt.iterations), 150)
    try:
        model.vis.global_.fovy = float(dc.fov_y_deg)
    except Exception:
        pass
    ax = float(dc.raw_depth_w) / float(dc.raw_depth_h)
    fovx_implied = float(np.degrees(2.0 * np.arctan(np.tan(np.radians(dc.fov_y_deg) / 2.0) * ax)))
    vertical_fov_deg = dc.fov_y_deg
    horizontal_fov_deg = fovx_implied

    ground_clearance = float(cfg.robot_config.ground_clearance)
    ground_geom_margin = float(getattr(cfg.robot_config, "ground_geom_margin", 0.0))
    terrain_mesh_path = getattr(cfg.sim_config, "terrain_mesh_path", None)
    root_z_lift = compute_root_z_lift(
        motion_pos,
        0,
        min_clearance=ground_clearance + ground_geom_margin,
        terrain_mesh_path=terrain_mesh_path,
    )
    if root_z_lift > 0.0:
        print(
            f"[INFO] Root z lift: +{root_z_lift:.4f} m "
            f"(motion link min_z={float(motion_pos[..., 2].min()):.4f}, "
            f"target clearance={ground_clearance + ground_geom_margin:.3f} m)"
        )

    data = mujoco.MjData(model)
    apply_motion_initial_state(
        data,
        cfg,
        motion_pos,
        motion_quat,
        m_joint_pos,
        m_joint_vel,
        m_body_lin_vel,
        m_body_ang_vel,
        0,
        z_lift=root_z_lift,
    )
    mujoco.mj_forward(model, data)

    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()

    depth_renderer = None
    depth_cam = None
    if dc.depth_backend == "renderer":
        depth_renderer = mujoco.Renderer(model, height=dc.raw_depth_h, width=dc.raw_depth_w)
        if hasattr(depth_renderer, "enable_depth_rendering"):
            depth_renderer.enable_depth_rendering()
        depth_cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(depth_cam)

    target_pos = np.zeros((cfg.robot_config.num_actions,), dtype=np.double)
    action = np.zeros((cfg.robot_config.num_actions,), dtype=np.double)

    frame_stack = int(cfg.robot_config.frame_stack)
    depth_history_len = int(cfg.robot_config.depth_history_len)
    depth_num_output_frames = int(cfg.robot_config.depth_num_output_frames)
    depth_history_skip_frames = int(cfg.robot_config.depth_history_skip_frames)
    num_command = 2 * cfg.robot_config.num_actions
    num_proprio = proprio_obs_dim(frame_stack, cfg.robot_config.num_actions)
    hist = {
        "command": TermHistory(frame_stack, num_command),
        "base_ang_vel": TermHistory(frame_stack, 3),
        "projected_gravity": TermHistory(frame_stack, 3),
        "joint_pos": TermHistory(frame_stack, cfg.robot_config.num_actions),
        "joint_vel": TermHistory(frame_stack, cfg.robot_config.num_actions),
        "actions": TermHistory(frame_stack, cfg.robot_config.num_actions),
    }
    depth_ring: deque[np.ndarray] = deque(maxlen=depth_history_len)
    is_first_policy = True
    count_lowlevel = 0
    motion_t = 0
    tau = np.zeros((cfg.robot_config.num_actions,), dtype=np.double)

    if headless:
        show_depth_vis = False
        renderer = mujoco.Renderer(model, width=1920, height=1080)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cam_vid = mujoco.MjvCamera()
        cam_vid.distance = 4.0
        cam_vid.azimuth = 45.0
        cam_vid.elevation = -20.0
        cam_vid.lookat = [0, 0, 1]
        out = cv2.VideoWriter(
            "simulation_perceptive.mp4",
            fourcc,
            1.0 / cfg.sim_config.dt / cfg.sim_config.decimation,
            (1920, 1080),
        )
        viewer = None
    else:
        viewer = open_interactive_viewer(model, data)
        renderer = None
        out = None
        cam_vid = None

    win_depth = "Raw Depth | Encoder Input"
    if show_depth_vis:
        cv2.namedWindow(win_depth, cv2.WINDOW_NORMAL)

    time_data: list[float] = []
    commanded_joint_pos_data: list[np.ndarray] = []
    actual_joint_pos_data: list[np.ndarray] = []

    n_step = int(cfg.sim_config.sim_duration / cfg.sim_config.dt)
    for step in tqdm(range(n_step), desc="Simulating..."):
        if cmd.reset_requested:
            data.qpos[:] = initial_qpos
            data.qvel[:] = initial_qvel
            data.ctrl[:] = 0.0
            mujoco.mj_forward(model, data)
            action[:] = 0.0
            tau[:] = 0.0
            target_pos[:] = cfg.robot_config.default_pos
            count_lowlevel = 0
            motion_t = 0
            for h in hist.values():
                h.reset()
            depth_ring.clear()
            is_first_policy = True
            cmd.reset_requested = False

        q, dq, _, _, omega, gvec = get_obs(data, model)
        q = q[-cfg.robot_config.num_actions :]
        dq = dq[-cfg.robot_config.num_actions :]

        if count_lowlevel % cfg.sim_config.decimation == 0:
            idx = frame_idx(motion_t, num_frames, cmd.loop_motion)

            q_ = q - cfg.robot_config.default_pos

            q_obs = np.zeros((cfg.robot_config.num_actions,), dtype=np.double)
            dq_obs = np.zeros((cfg.robot_config.num_actions,), dtype=np.double)
            for i in range(len(cfg.robot_config.usd2urdf)):
                q_obs[i] = q_[cfg.robot_config.usd2urdf[i]]
                dq_obs[i] = dq[cfg.robot_config.usd2urdf[i]]

            motion_command = build_motion_command(m_joint_pos, m_joint_vel, idx)
            vecs_policy = (
                motion_command,
                omega.astype(np.float32),
                gvec.astype(np.float32),
                q_obs.astype(np.float32),
                dq_obs.astype(np.float32),
                action.astype(np.float32),
            )
            if is_first_policy:
                for key, vec in zip(_PERCEPTIVE_OBS_HISTORY_KEYS, vecs_policy):
                    hist[key].fill_tile(vec)
                is_first_policy = False
            else:
                for key, vec in zip(_PERCEPTIVE_OBS_HISTORY_KEYS, vecs_policy):
                    hist[key].append(vec)

            raw_depth = capture_raycaster_depth(
                depth_renderer,
                depth_cam,
                data,
                model,
                cfg.sim_config.depth_camera_body,
                vertical_fov_deg,
                horizontal_fov_deg,
                dc,
            )
            z_depth_low = raw_depth_to_metric_grid(raw_depth, model, dc)
            dproc = preprocess_depth_frame_from_lowres(z_depth_low, dc)
            depth_ring.append(dproc)
            depth_frames = sample_depth_history(
                list(depth_ring),
                history_len=depth_history_len,
                num_output_frames=depth_num_output_frames,
                history_skip_frames=depth_history_skip_frames,
                encoder_h=dc.encoder_h,
                encoder_w=dc.encoder_w,
            )
            depth_bchw = depth_frames[None, ...].astype(np.float32)

            if show_depth_vis:
                show_depth_camera_side_by_side(z_depth_low, dproc, win_depth, depth_vis_scale, dc)

            flat_proprio = np.concatenate([hist[k].flat() for k in _PERCEPTIVE_OBS_HISTORY_KEYS], axis=0)
            assert flat_proprio.shape[0] == num_proprio, flat_proprio.shape

            if zero_latent:
                latent_vec = np.zeros(128, dtype=np.float32)
            else:
                latent = depth_encoder.run(None, {enc_in_name: depth_bchw})[0]
                latent_vec = latent.reshape(-1).astype(np.float32)
            actor_in = np.concatenate([flat_proprio, latent_vec], axis=0)[None, :].astype(np.float32)

            action[:] = actor.run(None, {act_in_name: actor_in})[0].reshape(-1)

            target_q = action * cfg.robot_config.action_scale
            for i in range(len(cfg.robot_config.usd2urdf)):
                target_pos[cfg.robot_config.usd2urdf[i]] = target_q[i]
            target_pos[:] = target_pos + cfg.robot_config.default_pos

            if debug_obs:
                latent_flat = latent_vec.astype(np.float64)
                depth_flat = depth_bchw.reshape(-1).astype(np.float64)
                print(f"[debug] motion_t={motion_t} idx={idx}" + (" [zero_latent]" if zero_latent else ""))
                print("q_obs:", q_obs)
                print("cmd pos:", motion_command[:23])
                print("action:", action)
                print("target_mj:", target_pos)
                print(
                    f"[debug] proprio={flat_proprio.shape} depth={depth_bchw.shape} "
                    f"latent={(128,) if zero_latent else latent_vec.shape} actor_in={actor_in.shape}"
                )
                print(
                    f"[debug] depth in: min={depth_flat.min():.4f} max={depth_flat.max():.4f} "
                    f"mean={depth_flat.mean():.4f} std={depth_flat.std():.4f}"
                )
                print(
                    f"[debug] latent: min={latent_flat.min():.4f} max={latent_flat.max():.4f} "
                    f"mean={latent_flat.mean():.4f} std={latent_flat.std():.4f} "
                    f"norm={np.linalg.norm(latent_flat):.4f}"
                )

            time_data.append(step * cfg.sim_config.dt)
            commanded_joint_pos_data.append(target_pos.copy())
            actual_joint_pos_data.append(q.copy())

            if not quiet:
                print(f"motion_t={motion_t} idx={idx} cmd_norm={np.linalg.norm(motion_command):.2f}")

            if headless:
                renderer.update_scene(data, camera=cam_vid)
                if cmd.camera_follow:
                    update_follow_camera(cam_vid, data, model)
                out.write(renderer.render())
            else:
                if cmd.camera_follow:
                    update_follow_camera(viewer.cam, data, model)
                viewer.render()

            motion_t += 1

        target_vel = np.zeros((cfg.robot_config.num_actions,), dtype=np.double)
        tau = pd_control(target_pos, q, target_vel, dq, cfg.robot_config.kps, cfg.robot_config.kds)
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
        data.ctrl = tau
        mj_macro_step(model, data, n_sub=phys_substeps)
        count_lowlevel += 1

    if headless:
        out.release()
    elif viewer is not None:
        viewer.close()
    keyboard_listener.stop()

    if show_depth_vis:
        cv2.destroyWindow(win_depth)

    # plots
    if time_data and commanded_joint_pos_data:
        time_arr = np.asarray(time_data)
        cmd_j = np.asarray(commanded_joint_pos_data)
        act_j = np.asarray(actual_joint_pos_data)
        num_joints = cfg.robot_config.num_actions
        n_cols = 4
        n_rows = (num_joints + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharex=True)
        axes = axes.flatten()
        for i in range(num_joints):
            ax = axes[i]
            ax.plot(time_arr, cmd_j[:, i], label="cmd", linestyle="--")
            ax.plot(time_arr, act_j[:, i], label="act")
            ax.set_title(f"Joint {i}")
            ax.grid(True)
            ax.legend()
        for i in range(num_joints, len(axes)):
            fig.delaxes(axes[i])
        fig.suptitle("RPO-Perceptive joint positions")
        plt.tight_layout()
        fig.savefig("joint_positions_perceptive.png")
        print("Saved joint_positions_perceptive.png")


# Policy proprio term order (matches PolicyCfg).
_PERCEPTIVE_OBS_HISTORY_KEYS: tuple[str, ...] = (
    "command",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RPO-Perceptive sim2sim (depth_encoder.onnx + actor.onnx).")
    default_export_path = "logs/rsl_rl/rpo_perceptive/2026-06-15_23-57-18/exported"
    parser.add_argument(
        "--depth_encoder",
        type=str,
        default=f"{default_export_path}/0-depth_encoder.onnx",
        help="Path to depth encoder ONNX (exported stem: depth_encoder).",
    )
    parser.add_argument(
        "--actor",
        type=str,
        default=f"{default_export_path}/actor.onnx",
        help="Path to actor ONNX (includes obs normalizer if exported with normalization).",
    )
    parser.add_argument(
        "--motion_file",
        type=str,
        default="robolab/data/motions/rpo_bm/beyond_reverse_vault_003_aug001_dm_aug8.npz",
        help="Reference motion NPZ (joint_pos/joint_vel for command obs).",
    )
    parser.add_argument(
        "--terrain_mesh",
        type=str,
        default="robolab/data/motions/rpo_bm/beyond_reverse_vault_003_aug001_dm_aug8_terrain.obj",
        help="Training OBJ terrain mesh (converted to MuJoCo hfield at runtime).",
    )
    parser.add_argument(
        "--no_motion_terrain",
        action="store_true",
        default=False,
        help="Do not auto-load *_terrain.obj; use --scene / --mujoco_xml ground only.",
    )
    parser.add_argument(
        "--terrain_hfield_res",
        type=float,
        default=0.03,
        help="XY sampling step (m) when converting OBJ mesh to MuJoCo hfield.",
    )
    parser.add_argument(
        "--mujoco_xml",
        type=str,
        default=None,
        help="MJCF path; overrides --scene when set.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        choices=("terrain", "plane"),
        default="plane",
        help="Base robot MJCF when no --terrain_mesh: terrain=rpo_terrain.xml, plane=rpo.xml.",
    )
    parser.add_argument("--loop", action="store_true", default=False, help="Loop motion reference.")
    parser.add_argument("--headless", action="store_true", default=False, help="Record simulation_perceptive.mp4.")
    parser.add_argument("--no_depth_vis", action="store_true", default=False, help="Disable OpenCV depth preview.")
    parser.add_argument("--debug_obs", action="store_true", default=False, help="Print observation shapes.")
    parser.add_argument(
        "--zero_latent",
        action="store_true",
        default=False,
        help="Debug: feed actor with [proprio, zeros(128)] instead of depth encoder output.",
    )
    args = parser.parse_args()

    if ort is None:
        raise RuntimeError("onnxruntime is required for this script") from _ORT_IMPORT_ERROR

    mjcf_dir = f"{ISAAC_DATA_DIR}/robots/roboparty/rpo/mjcf"
    scene_xml = {
        "terrain": f"{mjcf_dir}/rpo_terrain.xml",
        "plane": f"{mjcf_dir}/rpo.xml",
    }

    terrain_mesh = None if args.no_motion_terrain else args.terrain_mesh

    terrain_work_dir: tempfile.TemporaryDirectory[str] | None = None
    mujoco_model: mujoco.MjModel | None = None
    if terrain_mesh is not None:
        base_xml = args.mujoco_xml if args.mujoco_xml else f"{ISAAC_DATA_DIR}/robots/roboparty/rpo/mjcf/rpo.xml"
        mujoco_model, terrain_work_dir = build_mujoco_model(
            base_xml,
            terrain_mesh,
            hfield_res=args.terrain_hfield_res,
        )
        xml_path = base_xml
    else:
        xml_path = args.mujoco_xml if args.mujoco_xml else scene_xml[args.scene]

    class Sim2simCfg:
        class sim_config:
            mujoco_model_path = xml_path
            mujoco_model = mujoco_model
            terrain_work_dir = terrain_work_dir
            terrain_mesh_path = terrain_mesh
            sim_duration = 1_000_000.0
            dt = 0.005
            decimation = 4
            depth_camera_body = "torso_link"

        class depth_config:
            """Depth camera / encoder grid (match ``PerceptiveSceneCfg.camera``)."""
            raw_depth_h, raw_depth_w = 36, 64
            fov_y_deg = 58.29
            encoder_h, encoder_w = 18, 32
            crop_region = (18, 0, 16, 16)
            depth_clip = (0.0, 2.5)
            depth_backend = "ray"
            depth_ray_max = 2.5
            camera_offset_pos_body = np.array([0.0875, 0.01, 0.20568], dtype=np.float64)
            camera_offset_quat_wxyz = np.array([0.866, 0.0, 0.5, 0.0], dtype=np.float64)
            depth_znear_scale = 0.001
            depth_zfar_scale = 50.0

        class robot_config:
            kps = np.array(
                [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40, 150, 40, 40, 40, 30, 20, 40, 40, 40, 30, 20],
                dtype=np.double,
            )
            kds = np.array(
                [3.3, 3.3, 3.3, 5.0, 2.0, 2.0, 3.3, 3.3, 3.3, 5.0, 2.0, 2.0, 5.0, 2.0, 2.0, 2.0, 1.5, 1.0, 2.0, 2.0, 2.0, 1.5, 1.0],
                dtype=np.double,
            )
            default_pos = np.array(
                [0, 0, -0.1, 0.3, -0.2, 0, 0, 0, -0.1, 0.3, -0.2, 0, 0, 0.18, 0.06, 0, 0.78, 0, 0.18, -0.06, 0, 0.78, 0],
                dtype=np.double,
            )
            tau_limit = 200.0 * np.ones(23, dtype=np.double)
            frame_stack = 8  # obs history length (match perceptive_env_cfg ObservationsCfg)
            depth_history_len = 37  # camera data_histories distance_to_image_plane_noised
            depth_num_output_frames = 8  # PolicyCfg depth_image num_output_frames
            depth_history_skip_frames = 5  # PolicyCfg depth_image history_skip_frames
            ground_clearance = 0.03  # min link z above ground / terrain (see csv_on_terrain.py)
            ground_geom_margin = 0.05  # extra cushion: foot collision geoms sit below link origin
            num_actions = 23
            action_scale = 0.5
            # Isaac Lab joint order -> MuJoCo URDF qpos index (same as sim2sim_rpo_bm).
            usd2urdf = [0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22]

    depth_encoder, actor = build_onnx_sessions(
        args.depth_encoder, args.actor, providers=["CPUExecutionProvider"]
    )

    run_mujoco_onnx(
        depth_encoder,
        actor,
        Sim2simCfg(),
        args.motion_file,
        headless=args.headless,
        loop_motion=args.loop,
        show_depth_vis=not args.no_depth_vis and not args.headless,
        debug_obs=args.debug_obs,
        quiet=not args.debug_obs,
        zero_latent=args.zero_latent,
    )
