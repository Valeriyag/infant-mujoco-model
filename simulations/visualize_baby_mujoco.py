#!/usr/bin/env python3
"""Interactive 3D MuJoCo Viewer for Infant Biomechanical Model.

Part of the Baby MoCap EMG Biomechanical Modeling Suite.
Valeriya Gritsenko & Antigravity (2026).

Structured after the verified working viewer in:
  /Users/valeriyagritsenko/Documents/GitHub/msk_mujoco/mujoco_udp.py

Uses `mujoco.viewer.launch_passive` without mjpython trampolines.
Implements ADR-0003: Always represents the baby in the prone ("tummy time") pose
with elbows and hands resting on the floor, stomach and pelvis on the floor,
stationary legs with knees bent at 30 deg (-0.5236 rad) and feet facing outwards,
and active motion applied to the back extension DOF (i_2_1) and neck DOFs
(i_1_1, i_1_2, i_1_3).
"""

from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import sys
import time
import numpy as np

import mujoco
import mujoco.viewer


LEGACY_TO_CANONICAL = {
    'hd_pitch': 'i_1_1', 'hd_roll': 'i_1_2', 'hd_yaw': 'i_1_3',
    'pt_e_f': 'i_2_1',
    'ra_sh_e_f': 'ii_1_1', 'ra_sh_ab_ad': 'ii_1_2', 'ra_sh_e_i': 'ii_1_3',
    'ra_el_e_f': 'ii_2_1', 'ra_wr_e_f': 'ii_3_2',
    'la_sh_e_f': 'iii_1_1', 'la_sh_ab_ad': 'iii_1_2', 'la_sh_e_i': 'iii_1_3',
    'la_el_e_f': 'iii_2_1', 'la_wr_e_f': 'iii_3_2',
    'rl_h_e_f': 'iv_1_1', 'rl_h_ab_ad': 'iv_1_2', 'rl_h_e_i': 'iv_1_3',
    'rl_k_e_f': 'iv_2_1', 'rl_a_e_f': 'iv_3_1',
    'll_h_e_f': 'v_1_1', 'll_h_ab_ad': 'v_1_2', 'll_h_e_i': 'v_1_3',
    'll_k_e_f': 'v_2_1', 'll_a_e_f': 'v_3_1',
}
CANONICAL_TO_LEGACY = {v: k for k, v in LEGACY_TO_CANONICAL.items()}
# Support both canonical _1 (X-axis flex/ext) and legacy _2 DOF naming
CANONICAL_TO_LEGACY['ii_3_1'] = 'ra_wr_e_f'
CANONICAL_TO_LEGACY['iii_3_1'] = 'la_wr_e_f'
LEGACY_TO_CANONICAL['ra_wr_e_f_alt'] = 'ii_3_1'
LEGACY_TO_CANONICAL['la_wr_e_f_alt'] = 'iii_3_1'


def set_hinge(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str, angle_rad: float, angle_vel: float = 0.0):
    """Set hinge joint kinematics directly, supporting canonical or legacy name."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        alt = LEGACY_TO_CANONICAL.get(joint_name) or CANONICAL_TO_LEGACY.get(joint_name)
        if alt:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, alt)
    if jid < 0:
        raise ValueError(f"Joint '{joint_name}' not found in model.")
    qadr = model.jnt_qposadr[jid]
    assert model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE
    data.qpos[qadr] = angle_rad
    data.qvel[model.jnt_dofadr[jid]] = angle_vel


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


def set_tummy_time_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    back_ext: float = 0.0,
    neck_pitch: float = 0.0,
    neck_roll: float = 0.0,
    neck_yaw: float = 0.0,
    neck_pitch_vel: float = 0.0,
):
    """Apply the ADR-0003 prone ('tummy time') pose with stationary bent legs and floor-supported elbows/hands."""
    data.qpos[:7] = model.qpos0[:7]
    data.qvel[:] = 0.0

    for jname, angle in TUMMY_TIME_RESTING_ANGLES.items():
        set_hinge(model, data, jname, angle, 0.0)

    # Apply axial motion DOFs
    set_hinge(model, data, "i_2_1", back_ext, 0.0)
    set_hinge(model, data, "i_1_1", neck_pitch, neck_pitch_vel)
    set_hinge(model, data, "i_1_2", neck_roll, 0.0)
    set_hinge(model, data, "i_1_3", neck_yaw, 0.0)

    # Forearm propping compensation when back extends so elbows/hands remain resting on the ground
    u = float(np.clip(back_ext / 0.25, 0.0, 1.2))
    set_hinge(model, data, "ii_1_1", 0.5800 + 0.5800 * u, 0.0)
    set_hinge(model, data, "ii_1_2", -0.9600 + 0.0500 * u, 0.0)
    set_hinge(model, data, "ii_1_3", -0.7800 + 0.7800 * u, 0.0)
    set_hinge(model, data, "ii_2_1", 2.2689 - 0.4689 * u, 0.0)

    set_hinge(model, data, "iii_1_1", 0.5800 + 0.5800 * u, 0.0)
    set_hinge(model, data, "iii_1_2", 0.9600 - 0.0500 * u, 0.0)
    set_hinge(model, data, "iii_1_3", 0.7800 - 0.7800 * u, 0.0)
    set_hinge(model, data, "iii_2_1", 2.2689 - 0.4689 * u, 0.0)

    # Set actuator controls to hold the target joint angles during physics stepping
    for aid in range(model.nu):
        jid = model.actuator_trnid[aid, 0]
        if 0 <= jid < model.njnt and model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
            qadr = model.jnt_qposadr[jid]
            data.ctrl[aid] = data.qpos[qadr]


def run_viewer(
    model_path: Path,
    posture: str = "prone",
    frame: str = "body",
    mode: str = "step",
    transparent: bool = True,
    skin_only: bool = False,
    sim_time: float = 60.0
):
    """Launch interactive MuJoCo viewer with segment coordinate systems visualization."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"\n[INFO] Loading MuJoCo model: {model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Frame scale adjustment so axes are clearly visible
    model.vis.scale.framelength = 0.35
    model.vis.scale.framewidth = 0.02

    dt = model.opt.timestep
    steps = int(sim_time / dt)

    print(f"[INFO] Model Details:")
    print(f"  Total Generalized Coordinates (nq): {model.nq}")
    print(f"  Degrees of Freedom (nv):             {model.nv}")
    print(f"  Number of Joints (njnt):             {model.njnt}")
    print(f"  Number of Bodies (nbody):            {model.nbody}")
    print(f"  Number of Actuators (nu):            {model.nu}")
    print(f"  Simulation Timestep (dt):            {dt:.4f} s")
    print(f"  Posture:                             {posture.upper()}")
    print(f"  Frame Visualization:                 {'NONE (SKIN ONLY)' if skin_only else frame.upper()}")
    print(f"  Simulation Mode:                     {mode.upper()}")
    print(f"  Transparency:                        {False if skin_only else transparent}")
    print(f"  Skin Only Mode:                      {skin_only}")

    # Check and print joint i_2_1 range of motion
    jid_i21 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "i_2_1")
    if jid_i21 >= 0:
        r_i21 = model.jnt_range[jid_i21]
        print(f"  Joint i_2_1 (Pelvis-Trunk) Range:    [{r_i21[0]:.4f}, {r_i21[1]:.4f}] rad")

    # Configure initial posture: always ADR-0003 prone ("tummy time") pose
    set_tummy_time_pose(model, data)

    # Initialize forward kinematics
    mujoco.mj_forward(model, data)

    frame_map = {
        "body": mujoco.mjtFrame.mjFRAME_BODY,
        "geom": mujoco.mjtFrame.mjFRAME_GEOM,
        "site": mujoco.mjtFrame.mjFRAME_SITE,
        "none": mujoco.mjtFrame.mjFRAME_NONE,
    }
    frame_type = frame_map.get(frame.lower(), mujoco.mjtFrame.mjFRAME_BODY)

    print("\n===================================================================")
    print("   LAUNCHING INTERACTIVE 3D MUJOCO BABY MODEL VISUALIZER          ")
    print("===================================================================")
    print("  Segment Coordinate Frame Axes:")
    print("    - RED Axis:   Local +X (Mediolateral, Left -> Right)")
    print("    - GREEN Axis: Local +Y (Anteroposterior, Back -> Front / Tummy)")
    print("    - BLUE Axis:  Local +Z (Longitudinal, Toes -> Head)")
    print("\n  Controls:")
    print("    - Left-click + Drag:   Rotate camera")
    print("    - Right-click + Drag:  Zoom in / out")
    print("    - Middle-click + Drag: Pan camera")
    print("    - Key 'F':             Cycle coordinate frames (Body / Geom / Site / None)")
    print("    - Key '1':             Toggle internal bounding geoms (Group 1)")
    print("    - Key '2':             Toggle SMIL skin mesh (Group 2)")
    print("    - Key 'T':             Toggle transparency")
    print("    - Spacebar:            Pause / unpause simulation")
    print("  Close the visualization window when finished.")
    print("===================================================================\n")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Enable visual options under viewer lock
            with viewer.lock():
                if skin_only:
                    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
                    viewer.opt.geomgroup[1] = 0  # Hide internal segment bounding geoms (group 1)
                    viewer.opt.geomgroup[2] = 1  # Show SMIL skin mesh (group 2)
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0  # Solid opaque skin
                else:
                    viewer.opt.frame = frame_type
                    if transparent:
                        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1

            # Optimal camera viewpoint
            viewer.cam.lookat[0] = -0.08
            viewer.cam.lookat[1] = 0.20
            viewer.cam.lookat[2] = 0.05
            viewer.cam.distance = 0.90
            viewer.cam.elevation = -25.0
            viewer.cam.azimuth = 140.0

            for i in range(steps):
                if not viewer.is_running():
                    print("[INFO] Viewer closed by user.")
                    break

                t = i * dt

                if mode == "step":
                    mujoco.mj_step(model, data)
                elif mode == "head_pitch":
                    head_angle = 0.3927 * (1.0 - math.cos(2.0 * math.pi * 0.3 * t))
                    set_tummy_time_pose(model, data, back_ext=0.0, neck_pitch=head_angle)
                    mujoco.mj_forward(model, data)
                elif mode in ("tummy_time", "trunk_and_neck"):
                    # ADR-0003 tummy-time motion: up-and-down trunk extension + head pitch, yaw, and tilt
                    back_ext = 0.25 * (0.5 - 0.5 * math.cos(2.0 * math.pi * 0.25 * t))
                    neck_pitch = 0.35 * (0.5 - 0.5 * math.cos(2.0 * math.pi * 0.25 * t))
                    neck_yaw = 0.45 * math.sin(2.0 * math.pi * 0.20 * t)
                    neck_roll = 0.20 * math.sin(2.0 * math.pi * 0.40 * t)
                    set_tummy_time_pose(
                        model,
                        data,
                        back_ext=back_ext,
                        neck_pitch=neck_pitch,
                        neck_roll=neck_roll,
                        neck_yaw=neck_yaw,
                    )
                    mujoco.mj_forward(model, data)
                else:  # static mode: inspect resting tummy-time posture
                    set_tummy_time_pose(model, data)
                    mujoco.mj_forward(model, data)

                # Sync viewer
                viewer.sync()

                # Sleep to match real-time simulation speed
                time.sleep(dt)

    except RuntimeError as err:
        if "mjpython" in str(err):
            print("\n" + "=" * 70)
            print("  [ERROR] On macOS, interactive visualization requires 'mjpython'.")
            print("  Please run the script using:")
            print("    conda activate mujoco_env")
            print("    mjpython simulations/visualize_baby_mujoco.py")
            print("  or directly:")
            print("    /opt/homebrew/Caskroom/miniforge/base/envs/mujoco_env/bin/mjpython simulations/visualize_baby_mujoco.py")
            print("=" * 70 + "\n")
            sys.exit(1)
        raise err

    print("Visualization session completed.")


def main():
    parser = argparse.ArgumentParser(description="Interactive MuJoCo 3D Viewer for Baby Model.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/infant_4mo_M.xml"),
        help="Path to MuJoCo XML model (default: models/infant_4mo_M.xml)",
    )
    parser.add_argument(
        "--frame",
        type=str,
        choices=["body", "geom", "site", "none"],
        default="body",
        help="Coordinate frame visualization type (default: body)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["step", "static", "head_pitch", "tummy_time", "trunk_and_neck"],
        default="tummy_time",
        help="Simulation mode: tummy_time/trunk_and_neck (active back extension + neck motion), head_pitch (active head extension), step (physics step in tummy-time pose), static (fixed resting tummy-time pose) (default: tummy_time)",
    )
    parser.add_argument(
        "--posture",
        type=str,
        choices=["prone"],
        default="prone",
        help="Initial posture mode (always prone per ADR-0003)",
    )
    parser.add_argument(
        "--opaque",
        action="store_true",
        help="Disable transparency (by default bodies are transparent so frames are visible inside)",
    )
    parser.add_argument(
        "--skin-only",
        action="store_true",
        help="Show only the continuous SMIL skin mesh (hides internal primitive geoms and coordinate frames, renders skin opaque)",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=60.0,
        help="Simulation duration in seconds (default: 60.0)",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"MuJoCo model file not found: {args.model}")

    run_viewer(
        args.model,
        posture=args.posture,
        frame=args.frame,
        mode=args.mode,
        transparent=not args.opaque,
        skin_only=args.skin_only,
        sim_time=args.time,
    )


if __name__ == "__main__":
    main()

