# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MuJoCo sim2sim for RP1 AMP Get-Up policy (RP1-AMP-GetUp).

Initial pose: random prone/supine root orientation with default joint positions
plus uniform noise (matches training ``fallen_deviated`` reset).
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque

import cv2
import glfw
import matplotlib.pyplot as plt
import mujoco
import mujoco_viewer
import numpy as np
import torch
from pynput import keyboard
from robolab.assets import ISAAC_DATA_DIR
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

_OBS_HISTORY_KEYS = (
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
)

# Aligned with rp1_amp_get_up_env_cfg.py
FALLEN_JOINT_POS_NOISE = 0.3
FALLEN_ROOT_HEIGHT_RANGE = (0.2, 0.4)
SOFT_JOINT_POS_LIMIT_FACTOR = 0.90


class TermHistory:
    """Isaac CircularBuffer semantics: maxlen ring, flattened oldest-to-newest per observation term."""

    def __init__(self, max_len: int, term_dim: int):
        self.max_len = max_len
        self.term_dim = term_dim
        self._dq: deque[np.ndarray] = deque(maxlen=max_len)

    def reset(self):
        self._dq.clear()

    def append(self, x: np.ndarray):
        self._dq.append(np.asarray(x, dtype=np.float32).reshape(-1))

    def fill_tile(self, x: np.ndarray):
        self.reset()
        v = np.asarray(x, dtype=np.float32).reshape(-1)
        for _ in range(self.max_len):
            self._dq.append(v.copy())

    def flat(self) -> np.ndarray:
        if len(self._dq) == 0:
            return np.zeros(self.max_len * self.term_dim, dtype=np.float32)
        return np.concatenate(list(self._dq), axis=0)


class CompactOverlayMujocoViewer(mujoco_viewer.MujocoViewer):
    """MujocoViewer with the default left-side overlay hidden."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_200.value)

    def _create_overlay(self):
        super()._create_overlay()
        self._overlay.pop(mujoco.mjtGridPos.mjGRID_TOPLEFT, None)
        self._overlay.pop(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, None)


def open_interactive_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    fallback_width: int = 1920,
    fallback_height: int = 1080,
) -> mujoco_viewer.MujocoViewer:
    viewer = CompactOverlayMujocoViewer(
        model, data, mode="window", width=int(fallback_width), height=int(fallback_height)
    )
    viewer.cam.distance = 4.0
    viewer.cam.azimuth = 45.0
    viewer.cam.elevation = -20.0
    viewer.cam.lookat = [0, 0, 0.4]
    return viewer


def sleep_until(target_time: float, busy_wait_margin: float) -> None:
    remaining = target_time - time.perf_counter()
    if remaining > busy_wait_margin:
        time.sleep(remaining - busy_wait_margin)
    while time.perf_counter() < target_time:
        pass


class Cmd:
    camera_follow = True
    reset_requested = False

    @classmethod
    def toggle_camera_follow(cls):
        cls.camera_follow = not cls.camera_follow
        print(f"Camera follow: {cls.camera_follow}")


def on_press(key):
    try:
        if hasattr(key, "char") and key.char is not None:
            c = key.char.lower()
            if c == "f":
                Cmd.toggle_camera_follow()
            elif c == "0":
                Cmd.reset_requested = True
                print("Reset requested (0 key pressed)")
    except AttributeError:
        pass


def start_keyboard_listener():
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def get_obs(data: mujoco.MjData):
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)
    omega = data.sensor("angular-velocity").data.astype(np.double)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    return q, dq, quat, v, omega, gvec


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def _soft_joint_limits(model: mujoco.MjModel, joint_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match Isaac Lab soft_joint_pos_limit_factor on URDF/MJCF joint ranges."""
    lo = model.jnt_range[joint_ids, 0].copy()
    hi = model.jnt_range[joint_ids, 1].copy()
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) * SOFT_JOINT_POS_LIMIT_FACTOR
    return mid - half, mid + half


def _sample_fallen_root_quat_wxyz(rng: np.random.Generator) -> np.ndarray:
    """Random prone/supine orientation with uniform heading (matches mdp.events)."""
    pitch = 0.5 * math.pi if rng.random() < 0.5 else -0.5 * math.pi
    yaw = rng.uniform(-math.pi, math.pi)
    quat_xyzw = R.from_euler("xyz", [0.0, pitch, yaw]).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)


def reset_fallen_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cfg,
    rng: np.random.Generator,
) -> None:
    """Reset root to origin with random prone/supine pose and noisy default joints."""
    num_actions = cfg.robot_config.num_actions
    joint_ids = np.arange(1, 1 + num_actions, dtype=np.int32)
    lo, hi = _soft_joint_limits(model, joint_ids)

    height = rng.uniform(FALLEN_ROOT_HEIGHT_RANGE[0], FALLEN_ROOT_HEIGHT_RANGE[1])
    data.qpos[0:3] = [0.0, 0.0, height]
    data.qpos[3:7] = _sample_fallen_root_quat_wxyz(rng)

    joint_pos = cfg.robot_config.default_pos.copy()
    if FALLEN_JOINT_POS_NOISE > 0.0:
        joint_pos += rng.uniform(-FALLEN_JOINT_POS_NOISE, FALLEN_JOINT_POS_NOISE, size=joint_pos.shape)
    joint_pos = np.clip(joint_pos, lo, hi)
    data.qpos[-num_actions:] = joint_pos

    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def run_mujoco(policy, cfg, headless: bool = False):
    print("=" * 60)
    print("RP1 AMP Get-Up sim2sim")
    print("  0 key: Resample fallen pose (prone/supine + joint noise)")
    print("  F key: Toggle camera follow")
    print("  Velocity command is fixed to zero (standing, as in training)")
    print("=" * 60)
    keyboard_listener = start_keyboard_listener()
    rng = np.random.default_rng()

    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    reset_fallen_pose(model, data, cfg, rng)

    if headless:
        renderer = mujoco.Renderer(model, width=1920, height=1080)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cam = mujoco.MjvCamera()
        cam.distance = 4.0
        cam.azimuth = 45.0
        cam.elevation = -20.0
        cam.lookat = [0, 0, 0.4]
        out = cv2.VideoWriter(
            "simulation_get_up.mp4",
            fourcc,
            1.0 / cfg.sim_config.dt / cfg.sim_config.decimation,
            (1920, 1080),
        )
    else:
        viewer = open_interactive_viewer(model, data)

    target_pos = np.zeros((cfg.robot_config.num_actions,), dtype=np.double)
    action = np.zeros((cfg.robot_config.num_actions,), dtype=np.double)
    hist = {
        "base_ang_vel": TermHistory(cfg.robot_config.frame_stack, 3),
        "projected_gravity": TermHistory(cfg.robot_config.frame_stack, 3),
        "velocity_commands": TermHistory(cfg.robot_config.frame_stack, 3),
        "joint_pos": TermHistory(cfg.robot_config.frame_stack, cfg.robot_config.num_actions),
        "joint_vel": TermHistory(cfg.robot_config.frame_stack, cfg.robot_config.num_actions),
        "actions": TermHistory(cfg.robot_config.frame_stack, cfg.robot_config.num_actions),
    }

    count_lowlevel = 0
    next_render_time = 0.0
    render_interval = 1.0 / cfg.sim_config.render_fps
    is_first_frame = True
    start_time = time.perf_counter()

    time_data = []
    commanded_joint_pos_data = []
    actual_joint_pos_data = []
    base_height_data = []
    tau = np.zeros((cfg.robot_config.num_actions,), dtype=np.double)
    tau_data = []

    for step in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating...", mininterval=1.0):
        if Cmd.reset_requested:
            reset_fallen_pose(model, data, cfg, rng)
            action[:] = 0.0
            target_pos[:] = cfg.robot_config.default_pos.copy()
            tau[:] = 0.0
            for h in hist.values():
                h.reset()
            is_first_frame = True
            Cmd.reset_requested = False

        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.robot_config.num_actions :]
        dq = dq[-cfg.robot_config.num_actions :]

        if count_lowlevel % cfg.sim_config.decimation == 0:
            q_obs = np.zeros((cfg.robot_config.num_actions,), dtype=np.double)
            dq_obs = np.zeros((cfg.robot_config.num_actions,), dtype=np.double)
            q_ = q - cfg.robot_config.default_pos
            for i in range(len(cfg.robot_config.usd2urdf)):
                q_obs[i] = q_[cfg.robot_config.usd2urdf[i]]
                dq_obs[i] = dq[cfg.robot_config.usd2urdf[i]]

            # rel_standing_envs=1.0 during get-up training -> zero velocity command
            vel_cmd = np.zeros(3, dtype=np.float32)
            vecs_policy = (
                omega.astype(np.float32),
                gvec.astype(np.float32),
                vel_cmd,
                q_obs.astype(np.float32),
                dq_obs.astype(np.float32),
                action.astype(np.float32),
            )

            if is_first_frame:
                for key, vec in zip(_OBS_HISTORY_KEYS, vecs_policy):
                    hist[key].fill_tile(vec)
                is_first_frame = False
            else:
                for key, vec in zip(_OBS_HISTORY_KEYS, vecs_policy):
                    hist[key].append(vec)

            policy_input = np.concatenate([hist[k].flat() for k in _OBS_HISTORY_KEYS], axis=0)[None, :].astype(np.float32)
            assert policy_input.shape[1] == cfg.robot_config.num_observations, (
                f"Expected policy input dim {cfg.robot_config.num_observations}, got {policy_input.shape[1]}."
            )
            with torch.inference_mode():
                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()

            target_q = action * cfg.robot_config.action_scale
            target_pos[:] = cfg.robot_config.default_pos.copy()
            for i in range(len(cfg.robot_config.usd2urdf)):
                target_pos[cfg.robot_config.usd2urdf[i]] += target_q[i]

            time_data.append(step * cfg.sim_config.dt)
            commanded_joint_pos_data.append(target_pos.copy())
            actual_joint_pos_data.append(q.copy())
            base_height_data.append(float(data.qpos[2]))
            tau_data.append(tau.copy())

            if headless:
                renderer.update_scene(data, camera=cam)
                if Cmd.camera_follow:
                    base_pos = data.qpos[0:3].tolist()
                    cam.lookat = [float(base_pos[0]), float(base_pos[1]), float(base_pos[2])]
                out.write(renderer.render())

        target_vel = np.zeros((cfg.robot_config.num_actions,), dtype=np.double)
        tau = pd_control(target_pos, q, cfg.robot_config.kps, target_vel, dq, cfg.robot_config.kds)
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
        data.ctrl = tau
        mujoco.mj_step(model, data)

        count_lowlevel += 1

        if not headless:
            sim_time = (step + 1) * cfg.sim_config.dt
            if sim_time >= next_render_time:
                if Cmd.camera_follow:
                    base_pos = data.qpos[0:3].tolist()
                    viewer.cam.lookat = [float(base_pos[0]), float(base_pos[1]), float(base_pos[2])]
                viewer.render()
                while next_render_time <= sim_time:
                    next_render_time += render_interval

        target_wall_time = start_time + (step + 1) * cfg.sim_config.dt
        sleep_until(target_wall_time, cfg.sim_config.busy_wait_margin)

    if headless:
        out.release()
    else:
        viewer.close()
    keyboard_listener.stop()

    print("Simulation finished. Generating plots...")
    time_data = np.array(time_data)
    commanded_joint_pos_data = np.array(commanded_joint_pos_data)
    actual_joint_pos_data = np.array(actual_joint_pos_data)
    base_height_data = np.array(base_height_data)
    tau_data = np.array(tau_data)

    num_joints = cfg.robot_config.num_actions
    n_cols = 4
    n_rows = (num_joints + n_cols - 1) // n_cols
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharex=True)
    axes1 = axes1.flatten()
    for i in range(num_joints):
        ax = axes1[i]
        ax.plot(time_data, commanded_joint_pos_data[:, i], label="Commanded", linestyle="--")
        ax.plot(time_data, actual_joint_pos_data[:, i], label="Actual")
        ax.set_title(f"Joint {i + 1}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position [rad]")
        ax.legend()
        ax.grid(True)
    for i in range(num_joints, len(axes1)):
        fig1.delaxes(axes1[i])
    fig1.suptitle("Commanded vs Actual Joint Positions", fontsize=16)
    plt.tight_layout()

    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 4))
    ax2.plot(time_data, base_height_data, label="Base height (z)")
    ax2.axhline(0.55, color="gray", linestyle=":", label="phase3 threshold")
    ax2.axhline(0.75, color="gray", linestyle="--", label="target height")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Height [m]")
    ax2.legend()
    ax2.grid(True)
    fig2.suptitle("Base Height During Get-Up", fontsize=16)
    plt.tight_layout()

    fig1.savefig("get_up_joint_positions.png")
    fig2.savefig("get_up_base_height.png")
    print("Plots saved: get_up_joint_positions.png, get_up_base_height.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RP1 AMP Get-Up MuJoCo sim2sim.")
    parser.add_argument("--load_model", type=str, default="policy.pt", help="TorchScript policy path.")
    parser.add_argument(
        "--mujoco_xml",
        type=str,
        default=f"{ISAAC_DATA_DIR}/robots/roboparty/rp1.3/mjcf/rp1_flat.xml",
        help="MuJoCo scene XML.",
    )
    parser.add_argument("--headless", action="store_true", help="Run without GUI and save video.")
    args = parser.parse_args()

    class Sim2simCfg:
        class sim_config:
            mujoco_model_path = args.mujoco_xml
            sim_duration = 1_000_000.0
            dt = 0.005
            decimation = 4
            render_fps = 120.0
            busy_wait_margin = 0.0005

        class robot_config:
            kps = np.array(
                [
                    120.0, 120.0, 100.0, 100.0, 40.0, 40.0,
                    120.0, 120.0, 100.0, 100.0, 40.0, 40.0,
                    120.0, 100.0,
                    40.0, 40.0, 20.0, 30.0, 20.0,
                    40.0, 40.0, 20.0, 30.0, 20.0,
                ],
                dtype=np.double,
            )
            kds = np.array(
                [
                    12.0, 12.0, 5.0, 5.0, 2.0, 2.0,
                    12.0, 12.0, 5.0, 5.0, 2.0, 2.0,
                    12.0, 5.0,
                    4.0, 4.0, 1.0, 3.0, 1.0,
                    4.0, 4.0, 1.0, 3.0, 1.0,
                ],
                dtype=np.double,
            )
            default_pos = np.array(
                [
                    -0.2, 0.0, 0.0, -0.4, -0.2, 0.0,
                    0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
                    0.0, 0.0,
                    0.2, 0.2, 0.0, -1.2, 0.0,
                    -0.2, -0.2, 0.0, 1.2, 0.0,
                ],
                dtype=np.double,
            )
            tau_limit = np.array(
                [141.7] * 6 + [141.7] * 6 + [141.7] * 2 + [35.3] * 5 + [35.3] * 5,
                dtype=np.double,
            )
            frame_stack = 8
            num_actions = 24
            num_single_obs = 3 + 3 + 3 + num_actions * 3
            num_observations = num_single_obs * frame_stack
            action_scale = 0.25
            usd2urdf = [0, 6, 12, 1, 7, 13, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22, 18, 23]

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg(), args.headless)
