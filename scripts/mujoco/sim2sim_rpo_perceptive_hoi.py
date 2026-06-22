# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# MuJoCo sim2sim play for RPO-PerceptiveHoi policies (depth_encoder.onnx + actor.onnx).
#
# Extends sim2sim_rpo_perceptive with:
#   - HOI motion NPZ (object_pos_w / object_quat_w)
#   - largebox mesh as kinematic mocap body (depth rays + visualization)
#   - flat ground (no OBJ terrain)

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from typing import Any

import cv2
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from tqdm import tqdm

from robolab.assets import ISAAC_DATA_DIR

# Reuse sim2sim utilities from the perceptive script in this directory.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import sim2sim_rpo_perceptive as sp  # noqa: E402


def build_hoi_mujoco_model(base_xml_path: str, object_mesh_path: str | None) -> tuple[mujoco.MjModel, int | None]:
    """Load robot MJCF and optionally attach HOI object mesh as a mocap body."""
    base_xml_path = os.path.abspath(base_xml_path)
    if not object_mesh_path:
        return mujoco.MjModel.from_xml_path(base_xml_path), None

    object_mesh_path = os.path.abspath(object_mesh_path)
    if not os.path.isfile(object_mesh_path):
        raise FileNotFoundError(f"Object mesh not found: {object_mesh_path}")

    mesh_file = object_mesh_path
    if object_mesh_path.lower().endswith(".obj"):
        import trimesh

        work_mesh = trimesh.load(object_mesh_path, force="mesh")
        mesh_file = os.path.join(_SCRIPT_DIR, ".hoi_object_play.stl")
        work_mesh.export(mesh_file)

    spec = mujoco.MjSpec.from_file(base_xml_path)
    mesh_name = "hoi_box_mesh"
    spec.add_mesh(name=mesh_name, file=mesh_file)
    mocap_body = spec.worldbody.add_body(name="box_object", mocap=True)
    mocap_body.add_geom(
        name="box_object_geom",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh_name,
        rgba=[0.8, 0.6, 0.4, 1],
        contype=1,
        conaffinity=15,
        group=0,
    )
    model = spec.compile()
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box_object")
    mocap_id = int(model.body_mocapid[body_id])
    print(f"[INFO] HOI object mesh: {object_mesh_path}")
    return model, mocap_id


def apply_object_mocap_pose(
    data: mujoco.MjData,
    mocap_id: int | None,
    object_pos: np.ndarray,
    object_quat: np.ndarray,
    idx: int,
    *,
    z_lift: float = 0.0,
) -> None:
    """Drive kinematic HOI object from motion reference (matches Isaac reference tracking)."""
    if mocap_id is None or mocap_id < 0:
        return
    data.mocap_pos[mocap_id] = object_pos[idx].astype(np.float64)
    data.mocap_pos[mocap_id, 2] += z_lift
    data.mocap_quat[mocap_id] = object_quat[idx].astype(np.float64)


def run_mujoco_onnx_hoi(
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
    has_object = "object_pos_w" in motion and "object_quat_w" in motion
    object_pos = motion["object_pos_w"] if has_object else None
    object_quat = motion["object_quat_w"] if has_object else None

    frame_counts = [
        m_joint_pos.shape[0],
        m_joint_vel.shape[0],
        motion_pos.shape[0],
        motion_quat.shape[0],
        m_body_lin_vel.shape[0],
        m_body_ang_vel.shape[0],
    ]
    if has_object:
        frame_counts.extend([object_pos.shape[0], object_quat.shape[0]])
    num_frames = min(frame_counts)

    if has_object:
        print(f"[INFO] HOI motion loaded: {num_frames} frames with object reference.")
    else:
        print("[WARN] Motion NPZ has no object_pos_w/object_quat_w; playing robot-only.")

    sp.cmd.loop_motion = loop_motion
    keyboard_listener = sp.start_keyboard_listener()
    sp.print_controls_guide()
    if zero_latent:
        print("[INFO] --zero_latent: actor_in = [proprio, zeros(128)] (depth encoder skipped for actor)")

    dc = cfg.depth_config
    mocap_id = getattr(cfg.sim_config, "object_mocap_id", None)

    if cfg.sim_config.mujoco_model is not None:
        model = cfg.sim_config.mujoco_model
    else:
        model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    sp.configure_depth_rendering(model, dc)
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
    root_z_lift = sp.compute_root_z_lift(
        motion_pos,
        0,
        min_clearance=ground_clearance + ground_geom_margin,
        terrain_mesh_path=None,
    )
    if root_z_lift > 0.0:
        print(
            f"[INFO] Root z lift: +{root_z_lift:.4f} m "
            f"(motion link min_z={float(motion_pos[..., 2].min()):.4f}, "
            f"target clearance={ground_clearance + ground_geom_margin:.3f} m)"
        )

    data = mujoco.MjData(model)
    sp.apply_motion_initial_state(
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
    apply_object_mocap_pose(data, mocap_id, object_pos, object_quat, 0, z_lift=root_z_lift)
    mujoco.mj_forward(model, data)

    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()
    initial_mocap_pos = data.mocap_pos.copy() if mocap_id is not None and mocap_id >= 0 else None
    initial_mocap_quat = data.mocap_quat.copy() if mocap_id is not None and mocap_id >= 0 else None

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
    num_proprio = sp.proprio_obs_dim(frame_stack, cfg.robot_config.num_actions)
    hist = {
        "command": sp.TermHistory(frame_stack, num_command),
        "base_ang_vel": sp.TermHistory(frame_stack, 3),
        "projected_gravity": sp.TermHistory(frame_stack, 3),
        "joint_pos": sp.TermHistory(frame_stack, cfg.robot_config.num_actions),
        "joint_vel": sp.TermHistory(frame_stack, cfg.robot_config.num_actions),
        "actions": sp.TermHistory(frame_stack, cfg.robot_config.num_actions),
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
            "simulation_perceptive_hoi.mp4",
            fourcc,
            1.0 / cfg.sim_config.dt / cfg.sim_config.decimation,
            (1920, 1080),
        )
        viewer = None
    else:
        viewer = sp.open_interactive_viewer(model, data)
        renderer = None
        out = None
        cam_vid = None

    win_depth = "Raw Depth | Encoder Input (HOI)"
    if show_depth_vis:
        cv2.namedWindow(win_depth, cv2.WINDOW_NORMAL)

    time_data: list[float] = []
    commanded_joint_pos_data: list[np.ndarray] = []
    actual_joint_pos_data: list[np.ndarray] = []

    n_step = int(cfg.sim_config.sim_duration / cfg.sim_config.dt)
    for step in tqdm(range(n_step), desc="Simulating HOI..."):
        if sp.cmd.reset_requested:
            data.qpos[:] = initial_qpos
            data.qvel[:] = initial_qvel
            data.ctrl[:] = 0.0
            if initial_mocap_pos is not None:
                data.mocap_pos[:] = initial_mocap_pos
                data.mocap_quat[:] = initial_mocap_quat
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
            sp.cmd.reset_requested = False

        q, dq, _, _, omega, gvec = sp.get_obs(data, model)
        q = q[-cfg.robot_config.num_actions :]
        dq = dq[-cfg.robot_config.num_actions :]

        if count_lowlevel % cfg.sim_config.decimation == 0:
            idx = sp.frame_idx(motion_t, num_frames, sp.cmd.loop_motion)

            q_ = q - cfg.robot_config.default_pos
            q_obs = np.zeros((cfg.robot_config.num_actions,), dtype=np.double)
            dq_obs = np.zeros((cfg.robot_config.num_actions,), dtype=np.double)
            for i in range(len(cfg.robot_config.usd2urdf)):
                q_obs[i] = q_[cfg.robot_config.usd2urdf[i]]
                dq_obs[i] = dq[cfg.robot_config.usd2urdf[i]]

            motion_command = sp.build_motion_command(m_joint_pos, m_joint_vel, idx)
            vecs_policy = (
                motion_command,
                omega.astype(np.float32),
                gvec.astype(np.float32),
                q_obs.astype(np.float32),
                dq_obs.astype(np.float32),
                action.astype(np.float32),
            )
            if is_first_policy:
                for key, vec in zip(sp._PERCEPTIVE_OBS_HISTORY_KEYS, vecs_policy):
                    hist[key].fill_tile(vec)
                is_first_policy = False
            else:
                for key, vec in zip(sp._PERCEPTIVE_OBS_HISTORY_KEYS, vecs_policy):
                    hist[key].append(vec)

            if has_object:
                apply_object_mocap_pose(data, mocap_id, object_pos, object_quat, idx, z_lift=root_z_lift)
                mujoco.mj_forward(model, data)

            raw_depth = sp.capture_raycaster_depth(
                depth_renderer,
                depth_cam,
                data,
                model,
                cfg.sim_config.depth_camera_body,
                vertical_fov_deg,
                horizontal_fov_deg,
                dc,
            )
            z_depth_low = sp.raw_depth_to_metric_grid(raw_depth, model, dc)
            dproc = sp.preprocess_depth_frame_from_lowres(z_depth_low, dc)
            depth_ring.append(dproc)
            depth_frames = sp.sample_depth_history(
                list(depth_ring),
                history_len=depth_history_len,
                num_output_frames=depth_num_output_frames,
                history_skip_frames=depth_history_skip_frames,
                encoder_h=dc.encoder_h,
                encoder_w=dc.encoder_w,
            )
            depth_bchw = depth_frames[None, ...].astype(np.float32)

            if show_depth_vis:
                sp.show_depth_camera_side_by_side(z_depth_low, dproc, win_depth, depth_vis_scale, dc)

            flat_proprio = np.concatenate([hist[k].flat() for k in sp._PERCEPTIVE_OBS_HISTORY_KEYS], axis=0)
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
                print(f"[debug] motion_t={motion_t} idx={idx}" + (" [zero_latent]" if zero_latent else ""))
                print("q_obs:", q_obs)
                print("cmd pos:", motion_command[:23])
                print("action:", action)

            time_data.append(step * cfg.sim_config.dt)
            commanded_joint_pos_data.append(target_pos.copy())
            actual_joint_pos_data.append(q.copy())

            if not quiet:
                print(f"motion_t={motion_t} idx={idx} cmd_norm={np.linalg.norm(motion_command):.2f}")

            if headless:
                renderer.update_scene(data, camera=cam_vid)
                if sp.cmd.camera_follow:
                    sp.update_follow_camera(cam_vid, data, model)
                out.write(renderer.render())
            else:
                if sp.cmd.camera_follow:
                    sp.update_follow_camera(viewer.cam, data, model)
                viewer.render()

            motion_t += 1

        target_vel = np.zeros((cfg.robot_config.num_actions,), dtype=np.double)
        tau = sp.pd_control(target_pos, q, target_vel, dq, cfg.robot_config.kps, cfg.robot_config.kds)
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
        data.ctrl = tau
        sp.mj_macro_step(model, data, n_sub=phys_substeps)
        count_lowlevel += 1

    if headless:
        out.release()
    elif viewer is not None:
        viewer.close()
    keyboard_listener.stop()

    if show_depth_vis:
        cv2.destroyWindow(win_depth)

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
        fig.suptitle("RPO-PerceptiveHoi joint positions")
        plt.tight_layout()
        fig.savefig("joint_positions_perceptive_hoi.png")
        print("Saved joint_positions_perceptive_hoi.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RPO-PerceptiveHoi sim2sim (depth_encoder.onnx + actor.onnx).")
    default_export_path = "logs/rsl_rl/rpo_perceptive_hoi/exported"
    parser.add_argument(
        "--depth_encoder",
        type=str,
        default=f"{default_export_path}/0-depth_encoder.onnx",
        help="Path to depth encoder ONNX.",
    )
    parser.add_argument(
        "--actor",
        type=str,
        default=f"{default_export_path}/actor.onnx",
        help="Path to actor ONNX.",
    )
    parser.add_argument(
        "--motion_file",
        type=str,
        default="robolab/data/motions/rpo_bm/sub10_largebox_000.npz",
        help="HOI reference motion NPZ (robot + object_pos_w/object_quat_w).",
    )
    parser.add_argument(
        "--object_mesh",
        type=str,
        default="robolab/data/motions/rpo_bm/largebox_cleaned_simplified.obj",
        help="HOI object mesh for MuJoCo depth rays.",
    )
    parser.add_argument(
        "--no_object_mesh",
        action="store_true",
        default=False,
        help="Skip loading object mesh (robot-only play).",
    )
    parser.add_argument(
        "--mujoco_xml",
        type=str,
        default=None,
        help="Robot MJCF path (default: rpo.xml on flat ground).",
    )
    parser.add_argument("--loop", action="store_true", default=False, help="Loop motion reference.")
    parser.add_argument("--headless", action="store_true", default=False, help="Record simulation_perceptive_hoi.mp4.")
    parser.add_argument("--no_depth_vis", action="store_true", default=False, help="Disable OpenCV depth preview.")
    parser.add_argument("--debug_obs", action="store_true", default=False, help="Print observation debug info.")
    parser.add_argument(
        "--zero_latent",
        action="store_true",
        default=False,
        help="Debug: feed actor with [proprio, zeros(128)] instead of depth encoder output.",
    )
    args = parser.parse_args()

    if sp.ort is None:
        raise RuntimeError("onnxruntime is required for this script") from sp._ORT_IMPORT_ERROR

    mjcf_dir = f"{ISAAC_DATA_DIR}/robots/roboparty/rpo/mjcf"
    base_xml = args.mujoco_xml if args.mujoco_xml else f"{mjcf_dir}/rpo.xml"
    object_mesh = None if args.no_object_mesh else args.object_mesh
    model, mocap_id = build_hoi_mujoco_model(base_xml, object_mesh)

    class Sim2simHoiCfg:
        class sim_config:
            mujoco_model_path = base_xml
            mujoco_model = model
            object_mocap_id = mocap_id
            sim_duration = 1_000_000.0
            dt = 0.005
            decimation = 4
            depth_camera_body = "torso_link"

        class depth_config:
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
            frame_stack = 8
            depth_history_len = 37
            depth_num_output_frames = 8
            depth_history_skip_frames = 5
            ground_clearance = 0.03
            ground_geom_margin = 0.05
            num_actions = 23
            action_scale = 0.5
            usd2urdf = [0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22]

    depth_encoder, actor = sp.build_onnx_sessions(
        args.depth_encoder, args.actor, providers=["CPUExecutionProvider"]
    )

    run_mujoco_onnx_hoi(
        depth_encoder,
        actor,
        Sim2simHoiCfg(),
        args.motion_file,
        headless=args.headless,
        loop_motion=args.loop,
        show_depth_vis=not args.no_depth_vis and not args.headless,
        debug_obs=args.debug_obs,
        quiet=not args.debug_obs,
        zero_latent=args.zero_latent,
    )
