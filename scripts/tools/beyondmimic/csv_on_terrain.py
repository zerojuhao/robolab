# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a motion CSV on a custom OBJ terrain and optionally export npz.

.. code-block:: bash

    ./isaaclab.sh -p robolab/scripts/tools/beyondmimic/csv_on_terrain.py
    ./isaaclab.sh -p robolab/scripts/tools/beyondmimic/csv_on_terrain.py \\
        -f motion.csv --terrain_obj motion_terrain.obj --input_fps 30 --output_fps 50
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay motion CSV on an OBJ terrain mesh.")
parser.add_argument(
    "--input_file",
    "-f",
    type=str,
    default="robolab/data/motions/rpo_bm/beyond_reverse_vault_003_aug001_dm_aug8.csv",
    help="The path to the input motion csv file.",
)
parser.add_argument("--input_fps", type=int, default=30, help="The fps of the input motion.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
        " loaded."
    ),
)
parser.add_argument("--output_name", type=str, help="The name of the motion npz file.")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")
parser.add_argument(
    "--with_object",
    action="store_true",
    default=False,
    help=(
        "Parse object pose from the last 7 CSV columns (pos xyz + quat xyzw) and write"
        " object_pos_w / object_quat_w to the output npz. Default: off."
    ),
)
parser.add_argument("--terrain_obj", type=str, help="OBJ/STL terrain mesh paired with the motion CSV.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.output_name:
    args_cli.output_name = (
        "/".join(args_cli.input_file.split("/")[:-1]) + "/" + args_cli.input_file.split("/")[-1].replace(".csv", ".npz")
    )
if not args_cli.terrain_obj:
    args_cli.terrain_obj = (
        "/".join(args_cli.input_file.split("/")[:-1])
        + "/"
        + args_cli.input_file.split("/")[-1].replace(".csv", "_terrain.obj")
    )

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from robolab.assets.robots import RPO_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

from robolab.utils.obj_terrain import import_obj_as_terrain


@configclass
class CsvOnTerrainSceneCfg(InteractiveSceneCfg):
    """Scene with robot and lights; terrain OBJ is spawned separately."""

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = RPO_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=RPO_CFG.spawn.replace(
            rigid_props=RPO_CFG.spawn.rigid_props.replace(disable_gravity=True),
        ),
    )


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
        with_object: bool = False,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self.with_object = with_object
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from the csv file."""
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=self.frame_range[0] - 1,
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                )
            )
        motion = motion.to(torch.float32).to(self.device)
        # Export CSV: time, root_xyz, root_qxyzw, dof_* => drop the time column.
        if motion.shape[1] >= 31:
            motion = motion[:, 1:]
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]
        self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]  # convert to wxyz
        if self.with_object:
            self.motion_dof_poss_input = motion[:, 7:-7]
            self.motion_object_poss_input = motion[:, -7:-4]
            self.motion_object_rots_input = motion[:, -4:]
            self.motion_object_rots_input = self.motion_object_rots_input[:, [3, 0, 1, 2]]  # convert to wxyz
        else:
            self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        if self.with_object:
            self.motion_object_poss = self._lerp(
                self.motion_object_poss_input[index_0],
                self.motion_object_poss_input[index_1],
                blend.unsqueeze(1),
            )
            self.motion_object_rots = self._slerp(
                self.motion_object_rots_input[index_0],
                self.motion_object_rots_input[index_1],
                blend,
            )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations."""
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Gets the next state of the motion."""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
        with_object=args_cli.with_object,
    )

    robot = scene["robot"]
    joint_sdk_names = [
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
    robot_joint_indexes = robot.find_joints(joint_sdk_names, preserve_order=True)[0]

    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    if motion.with_object:
        log["object_pos_w"] = []
        log["object_quat_w"] = []
    file_saved = False

    while simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        scene.write_data_to_sim()
        sim.render()
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())
            if motion.with_object:
                object_pos = motion.motion_object_poss[motion.current_idx - 1 : motion.current_idx]
                object_rot = motion.motion_object_rots[motion.current_idx - 1 : motion.current_idx]
                log["object_pos_w"].append(object_pos[0].cpu().numpy().copy())
                log["object_quat_w"].append(object_rot[0].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            save_keys = (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            )
            if motion.with_object:
                save_keys = save_keys + ("object_pos_w", "object_quat_w")
            for k in save_keys:
                log[k] = np.stack(log[k], axis=0)

            # Lift all bodies so the lowest link clears the ground (avoid penetration on replay).
            body_pos = log["body_pos_w"]
            min_z = body_pos[..., 2].min()
            z_shift = -min_z + 0.03
            body_pos[..., 2] += z_shift
            log["body_pos_w"] = body_pos
            if motion.with_object:
                log["object_pos_w"][..., 2] += z_shift
            print(
                f"[INFO]: body_pos_w z corrected: global min_z={min_z:.6f}, shifted by +{z_shift:.6f}"
                f" (all links z >= 0.03 m)"
            )

            np.savez(args_cli.output_name, **log)
            print("[INFO]: Motion npz file saved to", args_cli.output_name)


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)

    import_obj_as_terrain("/World/ground", os.path.abspath(args_cli.terrain_obj))

    scene_cfg = CsvOnTerrainSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
