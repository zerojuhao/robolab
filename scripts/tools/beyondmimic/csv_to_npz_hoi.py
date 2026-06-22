
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay robot + object HOI motion from CSV and export a ground-corrected npz.

.. code-block:: bash

    python robolab/scripts/tools/beyondmimic/csv_to_npz_hoi.py
    python robolab/scripts/tools/beyondmimic/csv_to_npz_hoi.py --input_fps 30 --output_fps 50
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay HOI motion from csv files and output npz.")
parser.add_argument(
    "--input_file",
    "-f",
    type=str,
    default="robolab/data/motions/rpo_bm/sub10_largebox_000.csv",
    help="Robot motion csv (time + root pose + dof).",
)
parser.add_argument("--input_fps", type=int, default=30, help="Input motion fps.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help="Inclusive frame range (1-based). Default: all frames.",
)
parser.add_argument("--output_name", type=str, help="Output npz path.")
parser.add_argument("--output_fps", type=int, default=50, help="Output motion fps.")
parser.add_argument(
    "--object_file",
    type=str,
    default="robolab/data/motions/rpo_bm/object_0_largebox.csv",
    help="Object motion csv (time + pos xyz + quat xyzw).",
)
parser.add_argument(
    "--object_mesh",
    type=str,
    default="robolab/data/motions/rpo_bm/largebox_cleaned_simplified.obj",
    help="Object mesh (.obj / .stl).",
)
parser.add_argument(
    "--ground_clearance",
    type=float,
    default=0.04,
    help="Minimum link z above ground after correction (meters).",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.output_name:
    args_cli.output_name = (
        "/".join(args_cli.input_file.split("/")[:-1])
        + "/"
        + args_cli.input_file.split("/")[-1].replace(".csv", ".npz")
    )

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
from dataclasses import MISSING, dataclass, field

import torch

from robolab.assets.robots import RPO_CFG
from robolab.utils.hoi_object import make_hoi_object_cfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

ROBOT_JOINT_SDK_NAMES = [
    "left_thigh_yaw_joint",
    "left_thigh_roll_joint",
    "left_thigh_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_thigh_yaw_joint",
    "right_thigh_roll_joint",
    "right_thigh_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "torso_joint",
    "left_arm_pitch_joint",
    "left_arm_roll_joint",
    "left_arm_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_yaw_joint",
    "right_arm_pitch_joint",
    "right_arm_roll_joint",
    "right_arm_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_yaw_joint",
]


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = RPO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class ReplayHoiSceneCfg(ReplayMotionsSceneCfg):
    box_object: RigidObjectCfg = MISSING


@dataclass
class GroundScan:
    """Tracks link heights during the original-motion preview pass."""

    global_min_z: float = float("inf")
    link_min_z: dict[str, float] = field(default_factory=dict)

    def update(self, body_names: list[str], body_pos_w: np.ndarray) -> None:
        z_vals = body_pos_w[:, 2]
        self.global_min_z = min(self.global_min_z, float(z_vals.min()))
        for idx, name in enumerate(body_names):
            z = float(z_vals[idx])
            if z < 0.0:
                self.link_min_z[name] = min(self.link_min_z.get(name, z), z)

    def z_shift(self, ground_clearance: float) -> float:
        if self.global_min_z == float("inf"):
            return 0.0
        return -self.global_min_z + ground_clearance

    def log_report(self, ground_clearance: float) -> None:
        if self.link_min_z:
            print("[INFO] Links with z < 0 in pass 1 (original motion):")
            for name, z in sorted(self.link_min_z.items(), key=lambda item: item[1]):
                print(f"[INFO]   {name}: min_z={z:.6f}")
        else:
            print("[INFO] All links have z >= 0 in pass 1 (original motion).")

        shift = self.z_shift(ground_clearance)
        if shift != 0.0:
            print(
                f"[INFO] Ground correction: min_z={self.global_min_z:.6f},"
                f" shift z by +{shift:.6f} m"
                f" (z_shift = -min_z + ground_clearance, ground_clearance={ground_clearance:.3f} m)"
            )
        else:
            print(
                f"[INFO] No ground correction needed: global min_z={self.global_min_z:.6f}"
                f" (already >= ground_clearance={ground_clearance:.3f} m)"
            )


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        object_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.device = device
        self.frame_range = frame_range
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / input_fps
        self.output_dt = 1.0 / output_fps
        self.current_idx = 0
        self._load_robot_csv(motion_file)
        self._load_object_csv(object_file)
        self._interpolate_motion()
        self._compute_velocities()
        self._orig_base_pos = self.motion_base_poss.clone()
        self._orig_object_pos = self.motion_object_poss.clone()
        self.z_shift = 0.0
        self.use_corrected = False

    def _load_csv(self, csv_file: str) -> torch.Tensor:
        if self.frame_range is None:
            data = torch.from_numpy(np.loadtxt(csv_file, delimiter=","))
        else:
            data = torch.from_numpy(
                np.loadtxt(
                    csv_file,
                    delimiter=",",
                    skiprows=self.frame_range[0] - 1,
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                )
            )
        return data.to(torch.float32).to(self.device)

    def _load_robot_csv(self, motion_file: str) -> None:
        motion = self._load_csv(motion_file)
        if motion.shape[1] >= 31:
            motion = motion[:, 1:]
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7][:, [3, 0, 1, 2]]
        self.motion_dof_poss_input = motion[:, 7:]
        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt

    def _load_object_csv(self, object_file: str) -> None:
        motion = self._load_csv(object_file)
        if motion.shape[1] != 8:
            raise ValueError(f"Object csv must have 8 columns, got {motion.shape[1]} in {object_file}.")
        motion = motion[:, 1:]
        if motion.shape[0] != self.input_frames:
            raise ValueError("Object motion frame count must match robot motion.")
        self.motion_object_poss_input = motion[:, :3]
        self.motion_object_rots_input = motion[:, 3:7][:, [3, 0, 1, 2]]

    def _interpolate_motion(self) -> None:
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        i0, i1, blend = self._frame_blend(times)
        self.motion_base_poss = self._lerp(self.motion_base_poss_input[i0], self.motion_base_poss_input[i1], blend)
        self.motion_base_rots = self._slerp(self.motion_base_rots_input[i0], self.motion_base_rots_input[i1], blend)
        self.motion_dof_poss = self._lerp(self.motion_dof_poss_input[i0], self.motion_dof_poss_input[i1], blend)
        self.motion_object_poss = self._lerp(self.motion_object_poss_input[i0], self.motion_object_poss_input[i1], blend)
        self.motion_object_rots = self._slerp(self.motion_object_rots_input[i0], self.motion_object_rots_input[i1], blend)
        print(
            f"[INFO] Motion interpolated: {self.input_frames} @ {self.input_fps} fps ->"
            f" {self.output_frames} @ {self.output_fps} fps"
        )

    def _frame_blend(self, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phase = times / self.duration
        i0 = (phase * (self.input_frames - 1)).floor().long()
        i1 = torch.minimum(i0 + 1, torch.tensor(self.input_frames - 1, device=self.device))
        blend = (phase * (self.input_frames - 1) - i0).unsqueeze(1)
        return i0, i1, blend.squeeze(1)

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        if blend.ndim == 1:
            blend = blend.unsqueeze(1)
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(a)
        for i in range(a.shape[0]):
            out[i] = quat_slerp(a[i], b[i], blend[i])
        return out

    def _compute_velocities(self) -> None:
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)
        self.motion_object_lin_vels = torch.gradient(self.motion_object_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_object_ang_vels = self._so3_derivative(self.motion_object_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        return torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    def get_next_state(self) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], bool]:
        idx = self.current_idx
        robot_state = (
            self.motion_base_poss[idx : idx + 1],
            self.motion_base_rots[idx : idx + 1],
            self.motion_base_lin_vels[idx : idx + 1],
            self.motion_base_ang_vels[idx : idx + 1],
            self.motion_dof_poss[idx : idx + 1],
            self.motion_dof_vels[idx : idx + 1],
        )
        object_state = (
            self.motion_object_poss[idx : idx + 1],
            self.motion_object_rots[idx : idx + 1],
            self.motion_object_lin_vels[idx : idx + 1],
            self.motion_object_ang_vels[idx : idx + 1],
        )
        self.current_idx += 1
        reset_flag = self.current_idx >= self.output_frames
        if reset_flag:
            self.current_idx = 0
        return robot_state, object_state, reset_flag

    def set_z_shift(self, z_shift: float) -> None:
        self.z_shift = z_shift

    def set_playback_corrected(self, corrected: bool) -> None:
        """Switch between original and z-corrected trajectories."""
        self.use_corrected = corrected
        self.motion_base_poss = self._orig_base_pos.clone()
        self.motion_object_poss = self._orig_object_pos.clone()
        if corrected:
            self.motion_base_poss[:, 2] += self.z_shift
            self.motion_object_poss[:, 2] += self.z_shift
        self.current_idx = 0


def write_sim_states(
    robot,
    box_object,
    scene: InteractiveScene,
    robot_joint_indexes,
    robot_state: tuple[torch.Tensor, ...],
    object_state: tuple[torch.Tensor, ...],
) -> None:
    """Write robot and object root/joint states to sim."""
    base_pos, base_rot, base_lin_vel, base_ang_vel, dof_pos, dof_vel = robot_state
    obj_pos, obj_rot, obj_lin_vel, obj_ang_vel = object_state

    root_states = robot.data.default_root_state.clone()
    root_states[:, :3] = base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = base_rot
    root_states[:, 7:10] = base_lin_vel
    root_states[:, 10:] = base_ang_vel
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, robot_joint_indexes] = dof_pos
    joint_vel[:, robot_joint_indexes] = dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    obj_states = box_object.data.default_root_state.clone()
    obj_states[:, :3] = obj_pos
    obj_states[:, :2] += scene.env_origins[:, :2]
    obj_states[:, 3:7] = obj_rot
    obj_states[:, 7:10] = obj_lin_vel
    obj_states[:, 10:] = obj_ang_vel
    box_object.write_root_state_to_sim(obj_states)


def new_log_buffer() -> dict:
    return {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
        "object_pos_w": [],
        "object_quat_w": [],
    }


def append_log_frame(log: dict, robot, box_object) -> None:
    log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
    log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
    log["body_pos_w"].append(robot.data.body_pos_w[0].cpu().numpy().copy())
    log["body_quat_w"].append(robot.data.body_quat_w[0].cpu().numpy().copy())
    log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0].cpu().numpy().copy())
    log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0].cpu().numpy().copy())
    log["object_pos_w"].append(box_object.data.root_pos_w[0].cpu().numpy().copy())
    log["object_quat_w"].append(box_object.data.root_quat_w[0].cpu().numpy().copy())


def save_log(log: dict, output_name: str, ground_clearance: float) -> None:
    keys = (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "object_pos_w",
        "object_quat_w",
    )
    for key in keys:
        log[key] = np.stack(log[key], axis=0)

    body_pos = log["body_pos_w"]
    min_z = float(body_pos[..., 2].min())
    z_shift = -min_z + ground_clearance
    if abs(z_shift) > 1e-6:
        body_pos[..., 2] += z_shift
        log["body_pos_w"] = body_pos
        log["object_pos_w"][..., 2] += z_shift
        print(
            f"[INFO] body_pos_w z corrected: global min_z={min_z:.6f}, shifted by +{z_shift:.6f} m"
            f" (z_shift = -min_z + ground_clearance, ground_clearance={ground_clearance:.3f} m)"
        )

    np.savez(output_name, **log)
    print(f"[INFO] Ground-corrected motion npz saved to {output_name}")


def run_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        object_file=args_cli.object_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
    )

    robot = scene["robot"]
    box_object = scene["box_object"]
    robot_joint_indexes = robot.find_joints(ROBOT_JOINT_SDK_NAMES, preserve_order=True)[0]

    ground_scan = GroundScan()
    log = new_log_buffer()
    info_logged = False
    npz_saved = False

    while simulation_app.is_running():
        robot_state, object_state, cycle_done = motion.get_next_state()
        write_sim_states(robot, box_object, scene, robot_joint_indexes, robot_state, object_state)
        scene.write_data_to_sim()
        sim.render()
        scene.update(sim.get_physics_dt())

        if not info_logged:
            ground_scan.update(robot.body_names, robot.data.body_pos_w[0].cpu().numpy())
        elif motion.use_corrected and not npz_saved:
            append_log_frame(log, robot, box_object)

        if not cycle_done:
            continue

        if not info_logged:
            ground_scan.log_report(args_cli.ground_clearance)
            motion.set_z_shift(ground_scan.z_shift(args_cli.ground_clearance))
            motion.set_playback_corrected(True)
            print("[INFO] Alternating replay: original motion <-> z-corrected motion.")
            info_logged = True
            continue

        if motion.use_corrected and not npz_saved:
            save_log(log, args_cli.output_name, args_cli.ground_clearance)
            npz_saved = True
            log = new_log_buffer()

        motion.set_playback_corrected(not motion.use_corrected)


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(
        ReplayHoiSceneCfg(
            num_envs=1,
            env_spacing=2.0,
            box_object=make_hoi_object_cfg(
                args_cli.object_mesh,
                prim_path="{ENV_REGEX_NS}/Object",
                default_mass=1.0,
                kinematic=True,
            ),
        )
    )
    sim.reset()
    print("[INFO] Setup complete.")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
