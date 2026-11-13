#!/usr/bin/env python3
"""Interactive MuJoCo Viewer with Active Joint Position Controls.

Allows interactive inspection and control of infant model joints on macOS
using `mujoco.viewer.launch_passive` with full side UI panels enabled.
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

# ADR-0003 Canonical Prone ("Tummy Time") Resting Joint Angles (rad)
# Ground-propped forearms and physiological flexed resting leg flare
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


def set_tummy_time_pose(model: mujoco.MjModel, data: mujoco.MjData):
    """Set initial prone tummy-time resting posture."""
    data.qpos[:7] = model.qpos0[:7]
    for jname, angle in TUMMY_TIME_RESTING_ANGLES.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = angle

    # Initialize actuator controls to maintain posture
    for aid in range(model.nu):
        jid = model.actuator_trnid[aid, 0]
        if 0 <= jid < model.njnt and model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
            data.ctrl[aid] = data.qpos[model.jnt_qposadr[jid]]

    mujoco.mj_forward(model, data)


def main():
    parser = argparse.ArgumentParser(description="Interactive MuJoCo Joint Control Viewer.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/infant_4mo_M.xml"),
        help="Path to MuJoCo XML model (default: models/infant_4mo_M.xml)",
    )
    parser.add_argument(
        "--opaque",
        action="store_true",
        help="Disable semi-transparency of segments.",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    print(f"\n[INFO] Loading MuJoCo model: {args.model}")
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)

    # Scale frames for clear visualization
    model.vis.scale.framelength = 0.35
    model.vis.scale.framewidth = 0.02

    # Set canonical resting posture
    set_tummy_time_pose(model, data)

    print("\n" + "=" * 68)
    print("   INTERACTIVE JOINT CONTROL VIEWER (MUJOCO PASSIVE GUI)")
    print("=" * 68)
    print("  Controls:")
    print("    • Left & Right UI Panes: Open 'Control' tab to see actuators.")
    print("    • Move Joints Directly:  Ctrl + Right-Click & Drag on any limb!")
    print("    • Rotate Joint Torques:  Ctrl + Left-Click & Drag on any limb.")
    print("    • Camera Controls:       Left-Click (rotate), Right-Click (zoom),")
    print("                             Middle-Click / Shift+Left-Click (pan).")
    print("    • Spacebar:              Pause / Resume simulation.")
    print("    • 'F' key:               Cycle coordinate frames (Body/Geom/Site/None).")
    print("=" * 68 + "\n")

    try:
        with mujoco.viewer.launch_passive(
            model,
            data,
            show_left_ui=True,
            show_right_ui=True,
        ) as viewer:
            with viewer.lock():
                viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
                if not args.opaque:
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1

            # Set good viewing perspective
            viewer.cam.lookat[0] = -0.08
            viewer.cam.lookat[1] = 0.20
            viewer.cam.lookat[2] = 0.05
            viewer.cam.distance = 0.95
            viewer.cam.elevation = -25.0
            viewer.cam.azimuth = 140.0

            while viewer.is_running():
                step_start = time.time()
                mujoco.mj_step(model, data)
                viewer.sync()

                elapsed = time.time() - step_start
                if elapsed < model.opt.timestep:
                    time.sleep(model.opt.timestep - elapsed)

    except RuntimeError as err:
        if "mjpython" in str(err):
            print("\n" + "=" * 70)
            print("  [ERROR] On macOS, interactive visualization requires 'mjpython'.")
            print("  Run with:")
            print(f"    mjpython simulations/interactive_joint_control.py --model {args.model}")
            print("=" * 70 + "\n")
            sys.exit(1)
        raise err


if __name__ == "__main__":
    main()

