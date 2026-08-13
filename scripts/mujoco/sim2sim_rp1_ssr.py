# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# MuJoCo sim2sim for RP1 SSR policies exported as:
#   depth_encoder.onnx + policy_ssr.onnx
# where policy_ssr embeds estimation (v̂ / z / f̂) + MoE actor.
# Observation / control loop is shared with ``sim2sim_rp1_parkour.py``.

"""Play an exported RP1-SSR ONNX policy in MuJoCo.

Example:
::

    python robolab/scripts/mujoco/sim2sim_rp1_ssr.py --scene stairs
    python robolab/scripts/mujoco/sim2sim_rp1_ssr.py --headless --video --video_length 200
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from robolab.assets import ISAAC_DATA_DIR

if TYPE_CHECKING:
    from argparse import Namespace

# ---------------------------------------------------------------------------
# Shared parkour sim2sim helpers (loaded by path to avoid package packaging).
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_PARKOUR_PATH = _THIS_DIR / "sim2sim_rp1_parkour.py"

# Headless / no-DISPLAY hosts cannot import pynput's X11 backend.
if "pynput" not in sys.modules:
    try:
        import pynput  # noqa: F401
    except Exception:
        _keyboard = types.ModuleType("pynput.keyboard")

        class _DummyKey:
            pass

        class _DummyListener:
            def __init__(self, on_press=None, on_release=None):
                self.on_press = on_press
                self.on_release = on_release

            def start(self):
                return self

            def stop(self):
                return None

        _keyboard.Key = _DummyKey
        _keyboard.Listener = _DummyListener
        _pynput = types.ModuleType("pynput")
        _pynput.keyboard = _keyboard
        sys.modules["pynput"] = _pynput
        sys.modules["pynput.keyboard"] = _keyboard


def _load_parkour_module():
    """Import ``sim2sim_rp1_parkour`` from the sibling file."""
    spec = importlib.util.spec_from_file_location("sim2sim_rp1_parkour", _PARKOUR_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load parkour sim2sim module from {_PARKOUR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["sim2sim_rp1_parkour"] = module
    spec.loader.exec_module(module)
    return module


pk = _load_parkour_module()

# SSR depth encoder expects single-frame cropped 36×36 (Isaac crop (0,0,14,14) on 64×36).
_SSR_CROP_REGION = (0, 0, 14, 14)
_SSR_ENCODER_HW = (36, 36)

_DEFAULT_EXPORT = "rp1_ssr"
_DEFAULT_VIDEO_PATH = "simulation_ssr.mp4"
_MJCF_DIR = f"{ISAAC_DATA_DIR}/robots/roboparty/rp1.3/mjcf"
_SCENE_XML = {
    "stairs": f"{_MJCF_DIR}/rp1_stairs.xml",
    "terrain": f"{_MJCF_DIR}/rp1_rough.xml",
    "plane": f"{_MJCF_DIR}/rp1_flat_24dof.xml",
}

# PD / default pose in URDF order (matches parkour sim2sim / training).
_KPS = np.array(
    [
        150.0, 150.0, 100.0, 150.0, 60.0, 60.0,
        150.0, 150.0, 100.0, 150.0, 60.0, 60.0,
        300.0, 250.0,
        30.0, 30.0, 20.0, 30.0, 20.0,
        30.0, 30.0, 20.0, 30.0, 20.0,
    ],
    dtype=np.double,
)
_KDS = np.array(
    [
        6.0, 6.0, 4.0, 6.0, 3.0, 3.0,
        6.0, 6.0, 4.0, 6.0, 3.0, 3.0,
        15.0, 12.5,
        1.5, 1.5, 1.0, 1.5, 1.0,
        1.5, 1.5, 1.0, 1.5, 1.0,
    ],
    dtype=np.double,
)
_DEFAULT_POS = np.array(
    [
        -0.1, 0.0, 0.0, -0.3, -0.2, 0.0,
        0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        0.0, 0.0,
        0.2, 0.2, 0.0, -1.2, 0.0,
        -0.2, -0.2, 0.0, 1.2, 0.0,
    ],
    dtype=np.double,
)
_TAU_LIMIT = np.array(
    [145.53, 145.53, 145.53, 145.53, 56.0, 56.0]
    + [145.53, 145.53, 145.53, 145.53, 56.0, 56.0]
    + [145.53, 145.53]
    + [28.0] * 5
    + [28.0] * 5,
    dtype=np.double,
)
# lab_dof_names[i] -> gmr/URDF index.
_USD2URDF = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22, 18, 23
]

_SIM_DT = 0.005
_DECIMATION = 4


def add_ssr_sim2sim_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments (Isaac Lab play-style naming where applicable)."""
    parser.add_argument(
        "--depth_encoder",
        type=str,
        default=f"{_DEFAULT_EXPORT}/depth_encoder.onnx",
        help="Path to depth encoder ONNX.",
    )
    parser.add_argument(
        "--actor",
        type=str,
        default=f"{_DEFAULT_EXPORT}/policy_ssr.onnx",
        help="Path to SSR actor ONNX (estimation + MoE).",
    )
    parser.add_argument(
        "--mujoco_xml",
        type=str,
        default=None,
        help="MJCF path; if set, overrides --scene.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        choices=tuple(_SCENE_XML.keys()),
        default="stairs",
        help="Built-in scene: stairs / terrain / plane.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without GUI (required for offscreen video capture).",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        default=False,
        help="Record an mp4 (enables headless offscreen rendering).",
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=200,
        help="Recording length in policy steps when --video is set (default: 200).",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=_DEFAULT_VIDEO_PATH,
        help=f"Output video path (default: {_DEFAULT_VIDEO_PATH}).",
    )
    parser.add_argument(
        "--no_depth_vis",
        action="store_true",
        default=False,
        help="Do not open OpenCV depth preview.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Simulation duration in seconds (ignored when --video sets length).",
    )
    parser.add_argument(
        "--hold_forward",
        action="store_true",
        default=False,
        help="Hold forward velocity command (keyboard key 8) for the whole run.",
    )


def _resolve_xml_path(args: Namespace) -> str:
    if args.mujoco_xml:
        return args.mujoco_xml
    return _SCENE_XML[args.scene]


def _resolve_sim_duration(args: Namespace) -> float:
    """Return simulation horizon in seconds."""
    if args.video:
        # Policy step = dt * decimation (matches Isaac Lab "env step" pacing).
        return float(args.video_length) * _SIM_DT * float(_DECIMATION)
    if args.duration is not None:
        return float(args.duration)
    return 1_000_000.0


def make_sim2sim_cfg(xml_path: str, duration_s: float):
    """Build the nested cfg object expected by ``run_mujoco_onnx``."""

    class Sim2simCfg:
        class sim_config:
            mujoco_model_path = xml_path
            # Bind from outer ``duration_s`` (class body cannot self-ref ``sim_duration``).
            sim_duration = duration_s
            dt = _SIM_DT
            decimation = _DECIMATION
            depth_camera_body = "waist_yaw_link"

        class robot_config:
            kps = _KPS
            kds = _KDS
            default_pos = _DEFAULT_POS
            tau_limit = _TAU_LIMIT
            frame_stack = 8
            # SSR depth encoder is single-frame (exported as one crop per call).
            depth_encoder_frames = 1
            num_actions = 24
            action_scale = 0.25
            usd2urdf = _USD2URDF

    return Sim2simCfg()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RP1 SSR MuJoCo sim2sim (depth_encoder.onnx + policy_ssr.onnx)."
    )
    add_ssr_sim2sim_args(parser)
    args = parser.parse_args()

    # Offscreen VideoWriter lives on the headless path in ``run_mujoco_onnx``.
    headless = bool(args.headless or args.video)
    if args.video and not args.headless:
        print("[INFO] --video enables headless offscreen capture.")

    xml_path = _resolve_xml_path(args)
    sim_duration = _resolve_sim_duration(args)
    cfg = make_sim2sim_cfg(xml_path, sim_duration)

    if args.hold_forward:
        pk.cmd._pressed.add("8")
        print("[INFO] Holding forward velocity command (key 8).")

    # Match exported SSR depth encoder crop / resolution.
    pk._CROP_REGION = _SSR_CROP_REGION
    pk._ENCODER_H, pk._ENCODER_W = _SSR_ENCODER_HW

    enc_sess, act_sess = pk.build_onnx_sessions(
        args.depth_encoder,
        args.actor,
        providers=pk._SIM2SIM_PERF_ONNX_PROVIDERS,
    )
    pk.run_mujoco_onnx(
        enc_sess,
        act_sess,
        cfg,
        headless=headless,
        debug_obs=pk._SIM2SIM_PERF_DEBUG_OBS,
        show_depth_vis=not args.no_depth_vis,
        depth_vis_scale=max(1, pk._SIM2SIM_PERF_DEPTH_VIS_SCALE),
        realtime_sync=pk._SIM2SIM_PERF_REALTIME_SYNC,
        quiet=pk._SIM2SIM_PERF_QUIET,
        depth_vis_every_step=pk._SIM2SIM_PERF_DEPTH_VIS_EVERY_STEP,
        depth_vis_policy_stride=max(1, pk._SIM2SIM_PERF_DEPTH_VIS_POLICY_STRIDE),
        viewer_sync_every=pk._SIM2SIM_PERF_VIEWER_SYNC_EVERY,
        viewer_fallback_width=max(320, pk._SIM2SIM_PERF_VIEWER_FALLBACK_W),
        viewer_fallback_height=max(240, pk._SIM2SIM_PERF_VIEWER_FALLBACK_H),
        record_video=bool(args.video),
        video_path=args.video_path,
        save_plots=not args.video,
    )


if __name__ == "__main__":
    main()
