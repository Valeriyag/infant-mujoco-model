#!/usr/bin/env python3
"""Visualize segment coordinate systems in MuJoCo.

Displays local anatomical coordinate frames on each body segment:
  - Local +X (Red):   Mediolateral (Left -> Right)
  - Local +Y (Green): Anteroposterior (Back -> Front / Tummy)
  - Local +Z (Blue):  Longitudinal (Toes -> Head)

Usage:
  # With canonical model
  mjpython simulations/visualize_segment_frames_mujoco.py

  # With SMIL skin model
  mjpython simulations/visualize_segment_frames_mujoco.py --model models/infant_4mo_M_smil.xml
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

# ADR-0003 Canonical Prone ("Tummy Time") Resting Joint Angles (rad)
# Harmonized with physiological resting posture and continuous skin geometry
TUMMY_TIME_RESTING_ANGLES = {
    "i_2_1": 0.0,
    "i_1_1": 0.0,
    "i_1_2": 0.0,
    "i_1_3": 0.0,
    "ii_1_1": 0.2800,
    "ii_1_2": -0.3800,
    "ii_1_3": -0.2200,
    "ii_2_1": 2.2689,
    "ii_3_2": 0.1000,
    "iii_1_1": 0.2800,
    "iii_1_2": 0.3800,
    "iii_1_3": 0.2200,
    "iii_2_1": 2.2689,
    "iii_3_2": -0.1000,
    "iv_1_1": 0.0800,
    "iv_1_2": -0.3800,
    "iv_1_3": -0.4500,
    "iv_2_1": -0.5236,
    "iv_3_1": 0.0,
    "v_1_1": 0.0800,
    "v_1_2": 0.3800,
    "v_1_3": 0.4500,
    "v_2_1": -0.5236,
    "v_3_1": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize segment anatomical coordinate systems in MuJoCo."
    )
    parser.add_argument(
        "positional_model",
        nargs="?",
        default=None,
        help="Optional positional path to MuJoCo XML model.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to MuJoCo XML model (default: models/infant_4mo_M.xml).",
    )
    parser.add_argument(
        "--frame",
        type=str,
        default="body",
        choices=["body", "geom", "site", "none"],
        help="Coordinate frame visualization type (default: body).",
    )
    parser.add_argument(
        "--opaque",
        action="store_true",
        help="Disable semi-transparency of segments.",
    )
    parser.add_argument(
        "--framelength",
        type=float,
        default=0.35,
        help="Axis frame length scale (default: 0.35).",
    )
    parser.add_argument(
        "--framewidth",
        type=float,
        default=0.02,
        help="Axis frame width scale (default: 0.02).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve model path from either --model or positional argument
    if args.model is not None:
        model_path = args.model
    elif args.positional_model is not None:
        model_path = Path(args.positional_model)
    else:
        model_path = Path("models/infant_4mo_M.xml")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"[INFO] Loading MuJoCo model: {model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Frame scale adjustment for clear axis visibility
    model.vis.scale.framelength = args.framelength
    model.vis.scale.framewidth = args.framewidth

    # Apply canonical tummy-time resting pose
    data.qpos[:7] = model.qpos0[:7]
    for jname, angle in TUMMY_TIME_RESTING_ANGLES.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = angle

    # Set initial actuators to hold posture
    for aid in range(model.nu):
        jid = model.actuator_trnid[aid, 0]
        if 0 <= jid < model.njnt and model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
            data.ctrl[aid] = data.qpos[model.jnt_qposadr[jid]]

    mujoco.mj_forward(model, data)

    frame_map = {
        "body": mujoco.mjtFrame.mjFRAME_BODY,
        "geom": mujoco.mjtFrame.mjFRAME_GEOM,
        "site": mujoco.mjtFrame.mjFRAME_SITE,
        "none": mujoco.mjtFrame.mjFRAME_NONE,
    }

    print("\n" + "=" * 67)
    print("   MUJOCO SEGMENT COORDINATE FRAMES VISUALIZER")
    print("=" * 67)
    print(f"  Model: {model_path}")
    print("  Coordinate Frame Color Legend:")
    print("    - RED Axis:   Local +X (Mediolateral, Left -> Right)")
    print("    - GREEN Axis: Local +Y (Anteroposterior, Back -> Front / Tummy)")
    print("    - BLUE Axis:  Local +Z (Longitudinal, Toes -> Head)")
    print("\n  Controls:")
    print("    - Key 'F':     Cycle frame visibility (Body / Geom / Site / None)")
    print("    - Key '0':     Reset camera to default view")
    print("    - Spacebar:    Pause / unpause physics stepping")
    print("    - Mouse drag:  Rotate / Pan / Zoom camera")
    print("=" * 67 + "\n")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            with viewer.lock():
                viewer.opt.frame = frame_map[args.frame]
                if not args.opaque:
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1

            # Camera perspective
            viewer.cam.lookat[0] = -0.08
            viewer.cam.lookat[1] = 0.20
            viewer.cam.lookat[2] = 0.05
            viewer.cam.distance = 0.90
            viewer.cam.elevation = -25.0
            viewer.cam.azimuth = 140.0

            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)

    except RuntimeError as err:
        if "mjpython" in str(err):
            print("\n" + "=" * 70)
            print("  [ERROR] On macOS, interactive visualization requires 'mjpython'.")
            print("  Please run the script using:")
            print(f"    mjpython simulations/visualize_segment_frames_mujoco.py --model {model_path}")
            print("=" * 70 + "\n")
            sys.exit(1)
        raise err


if __name__ == "__main__":
    main()
