# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Load and visualize a RoboLab robot, then print its Isaac Lab joint/link order.

Example usage:

.. code-block:: bash

    ./isaaclab.sh -p robolab/scripts/tools/visualize_robot.py
    ./isaaclab.sh -p robolab/scripts/tools/visualize_robot.py --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize RP1_3_CFG and print Isaac Lab joint order.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from robolab.assets.robots.roboparty import RP1_3_CFG


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Scene with ground, lighting, and the RP1 robot."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = RP1_3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def print_joint_order(robot: Articulation) -> None:
    """Print Isaac Lab joint names in rp1_24dof.yaml lab_dof_names format."""
    joint_names = list(robot.joint_names)
    num_joints = len(joint_names)

    print("\n" + "=" * 72)
    print(f"Isaac Lab joint order ({num_joints} DoF)")
    print("=" * 72)
    for name in joint_names:
        print(f"  - {name}")
    print("=" * 72)
    for idx, name in enumerate(joint_names):
        print(f"  [{idx:2d}] {name}")
    print("=" * 72 + "\n")


def print_link_order(robot: Articulation) -> None:
    """Print Isaac Lab body names in roboparty.py PR1_LINKS format."""
    body_names = list(robot.body_names)
    num_bodies = len(body_names)

    print("\n" + "=" * 72)
    print(f"Isaac Lab link order ({num_bodies} links)")
    print("=" * 72)
    for idx, name in enumerate(body_names):
        comma = "," if idx < num_bodies - 1 else ""
        print(f"    '{name}'{comma}")
    print("]")
    print("=" * 72)
    for idx, name in enumerate(body_names):
        print(f"  [{idx:2d}] {name}")
    print("=" * 72 + "\n")


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    """Render loop: hold the robot at its default pose."""
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()

    root_states = robot.data.default_root_state.clone()
    root_states[:, :3] += scene.env_origins
    robot.write_root_state_to_sim(root_states)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    scene.write_data_to_sim()

    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.render()
        scene.update(sim_dt)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = RobotSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot: Articulation = scene["robot"]
    print_joint_order(robot)
    print_link_order(robot)

    if not args_cli.headless:
        root_pos = robot.data.default_root_state[0, :3].cpu().tolist()
        sim.set_camera_view(
            [root_pos[0] + 2.5, root_pos[1] + 2.5, root_pos[2] + 0.8],
            root_pos,
        )
        run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
