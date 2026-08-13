#!/usr/bin/env python3
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Record a headless MuJoCo video of RP1-SSR walking forward (no PNG plots).
#
# Example:
#   MUJOCO_GL=egl python robolab/scripts/mujoco/record_ssr_forward_video.py \
#     --depth_encoder logs/rsl_rl/rp1_ssr/<run>/exported/depth_encoder.onnx \
#     --actor logs/rsl_rl/rp1_ssr/<run>/exported/policy_ssr.onnx \
#     --output ssr_forward_10s.mp4 --duration 10

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np

# Offscreen rendering on machines without DISPLAY.
os.environ.setdefault("MUJOCO_GL", "egl")

from robolab.assets import ISAAC_DATA_DIR

# Stub pynput when X11 is unavailable (headless recorders only need Listener no-ops).
if "pynput" not in sys.modules:
    try:
        import pynput  # noqa: F401
    except Exception:
        _pynput = types.ModuleType("pynput")
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
        _pynput.keyboard = _keyboard
        sys.modules["pynput"] = _pynput
        sys.modules["pynput.keyboard"] = _keyboard

_PARKOUR_PATH = Path(__file__).resolve().parent / "sim2sim_rp1_parkour.py"
_SPEC = importlib.util.spec_from_file_location("sim2sim_rp1_parkour", _PARKOUR_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load parkour sim2sim module from {_PARKOUR_PATH}")
_parkour = importlib.util.module_from_spec(_SPEC)
sys.modules["sim2sim_rp1_parkour"] = _parkour
_SPEC.loader.exec_module(_parkour)


def _default_export_dir() -> Path:
    root = Path("logs/rsl_rl/rp1_ssr")
    if not root.is_dir():
        return Path("rp1_ssr")
    runs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)
    for run in reversed(runs):
        exported = run / "exported"
        if (exported / "depth_encoder.onnx").is_file() and (exported / "policy_ssr.onnx").is_file():
            return exported
    return Path("rp1_ssr")


def main() -> None:
    default_export = _default_export_dir()
    mjcf_dir = Path(ISAAC_DATA_DIR) / "robots/roboparty/rp1.3/mjcf"
    parser = argparse.ArgumentParser(
        description="Record RP1-SSR forward-walk MuJoCo video (no plot PNGs)."
    )
    parser.add_argument(
        "--depth_encoder",
        type=str,
        default=str(default_export / "depth_encoder.onnx"),
        help="Path to depth_encoder.onnx",
    )
    parser.add_argument(
        "--actor",
        type=str,
        default=str(default_export / "policy_ssr.onnx"),
        help="Path to policy_ssr.onnx",
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
        choices=("stairs", "terrain", "plane"),
        default="stairs",
        help="Scene: stairs / terrain / plane (default: stairs).",
    )
    parser.add_argument("--duration", type=float, default=10.0, help="Sim duration in seconds.")
    parser.add_argument(
        "--output",
        type=str,
        default="ssr_forward_10s.mp4",
        help="Output mp4 path.",
    )
    parser.add_argument(
        "--vx",
        type=float,
        default=None,
        help="Override forward target speed (m/s). Default: keyboard hold max (0.8).",
    )
    args = parser.parse_args()

    scene_xml = {
        "stairs": mjcf_dir / "rp1_stairs.xml",
        "terrain": mjcf_dir / "rp1_rough.xml",
        "plane": mjcf_dir / "rp1_flat_24dof.xml",
    }
    xml_path = args.mujoco_xml if args.mujoco_xml else str(scene_xml[args.scene])

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Match exported SSR depth encoder: single-frame cropped 36x36.
    _parkour._CROP_REGION = (0, 0, 14, 14)
    _parkour._ENCODER_H, _parkour._ENCODER_W = 36, 36
    _parkour.cmd._pressed.add("8")
    if args.vx is not None:
        _parkour.cmd.hold_vx_forward = float(args.vx)
    print(
        f"[INFO] scene={args.scene if args.mujoco_xml is None else xml_path}; "
        f"holding forward cmd; duration={args.duration:.1f}s -> {out_path}"
    )

    class Sim2simCfg:
        class sim_config:
            mujoco_model_path = xml_path
            sim_duration = float(args.duration)
            dt = 0.005
            decimation = 4
            depth_camera_body = "waist_yaw_link"

        class robot_config:
            kps = np.array(
                [
                    150.0, 150.0, 100.0, 150.0, 60.0, 60.0,
                    150.0, 150.0, 100.0, 150.0, 60.0, 60.0,
                    300.0, 250.0,
                    30.0, 30.0, 20.0, 30.0, 20.0,
                    30.0, 30.0, 20.0, 30.0, 20.0,
                ],
                dtype=np.double,
            )
            kds = np.array(
                [
                    6.0, 6.0, 4.0, 6.0, 3.0, 3.0,
                    6.0, 6.0, 4.0, 6.0, 3.0, 3.0,
                    15.0, 12.5,
                    1.5, 1.5, 1.0, 1.5, 1.0,
                    1.5, 1.5, 1.0, 1.5, 1.0,
                ],
                dtype=np.double,
            )
            default_pos = np.array(
                [
                    -0.1, 0.0, 0.0, -0.3, -0.2, 0.0,
                    0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                    0.0, 0.0,
                    0.2, 0.2, 0.0, -1.2, 0.0,
                    -0.2, -0.2, 0.0, 1.2, 0.0,
                ],
                dtype=np.double,
            )
            tau_limit = np.array(
                [145.53, 145.53, 145.53, 145.53, 56.0, 56.0]
                + [145.53, 145.53, 145.53, 145.53, 56.0, 56.0]
                + [145.53, 145.53]
                + [28.0] * 5
                + [28.0] * 5,
                dtype=np.double,
            )
            frame_stack = 8
            depth_encoder_frames = 1
            num_actions = 24
            action_scale = 0.25
            usd2urdf = [
                0, 6, 12, 1, 7, 13, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22, 18, 23
            ]

    enc_sess, act_sess = _parkour.build_onnx_sessions(
        args.depth_encoder,
        args.actor,
        providers=_parkour._SIM2SIM_PERF_ONNX_PROVIDERS,
    )
    _parkour.run_mujoco_onnx(
        enc_sess,
        act_sess,
        Sim2simCfg(),
        headless=True,
        show_depth_vis=False,
        realtime_sync=False,
        quiet=True,
        record_video=True,
        video_path=str(out_path),
        save_plots=False,
    )


if __name__ == "__main__":
    main()
