#!/usr/bin/env python3
"""Prone Play Upward Head Pitch Simulation Test Suite (MuJoCo).

Part of the Baby MoCap EMG Biomechanical Modeling Suite.
Valeriya Gritsenko & Antigravity (2026).

This script simulates an infant in the prone play position (spine horizontal,
ventral surface on floor) executing active upward head extension (head pitch,
raising head center of mass away from the mat against gravity).

Evaluates:
  1. Kinematic elevation of head center of mass as head pitch sweeps 0 to 45 deg (0.785 rad).
  2. Gravitational extensor neck torque required to sustain head elevation against gravity.
  3. Direct validation of MuJoCo generalized gravity torque against analytical pediatric torque model.
  4. Generates publication-ready multi-panel figure, validation report, and optional MP4 video animation.
  5. Interactive real-time 3D visualization via `--visualize` (mujoco.viewer).
"""

from __future__ import annotations
import argparse
import csv
import math
import os
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Ensure mujoco is available
try:
    import mujoco
except ImportError:
    mujoco_py = "/opt/homebrew/Caskroom/miniforge/base/envs/mujoco_env/bin/python"
    if os.path.exists(mujoco_py) and sys.executable != mujoco_py:
        os.execv(mujoco_py, [mujoco_py] + sys.argv)
    else:
        raise ImportError("MuJoCo python package not found. Please run with mujoco_env python.")


# Canonical Prone ("Tummy Time") Resting Pose per ADR-0003:
# - Stomach and pelvis lying on the ground (qpos0[:7])
# - Elbows and hands resting on the ground (propped alongside trunk)
# - Stationary legs lying on the ground with knees bent at 30 deg (-0.5236 rad) and feet facing outwards
TUMMY_TIME_RESTING_ANGLES = {
    'i_2_1': 0.0,
    'i_1_1': 0.0,
    'i_1_2': 0.0,
    'i_1_3': 0.0,
    # Right arm propped with elbow and hand resting on ground
    'ii_1_1': 0.5800,
    'ii_1_2': -0.9600,
    'ii_1_3': -0.7800,
    'ii_2_1': 2.2689,
    'ii_3_2': 0.2500,
    # Left arm propped with elbow and hand resting on ground
    'iii_1_1': 0.5800,
    'iii_1_2': 0.9600,
    'iii_1_3': 0.7800,
    'iii_2_1': 2.2689,
    'iii_3_2': -0.2500,
    # Right leg stationary on ground, knee bent 30 deg, foot facing outwards (+X)
    'iv_1_1': 0.2000,
    'iv_1_2': -0.4500,
    'iv_1_3': -1.5708,
    'iv_2_1': -0.5236,
    'iv_3_1': 0.0,
    # Left leg stationary on ground, knee bent 30 deg, foot facing outwards (-X)
    'v_1_1': 0.2000,
    'v_1_2': 0.4500,
    'v_1_3': 1.5708,
    'v_2_1': -0.5236,
    'v_3_1': 0.0,
}


def set_tummy_time_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    back_ext: float = 0.0,
    neck_pitch: float = 0.0,
    neck_roll: float = 0.0,
    neck_yaw: float = 0.0,
) -> None:
    """Set canonical ADR-0003 prone ('tummy time') pose with optional axial motion."""
    data.qpos[:7] = model.qpos0[:7]
    u = float(np.clip(back_ext / 0.25, 0.0, 1.2))
    angles = dict(TUMMY_TIME_RESTING_ANGLES)
    angles.update({
        'i_2_1': back_ext,
        'i_1_1': neck_pitch,
        'i_1_2': neck_roll,
        'i_1_3': neck_yaw,
        'ii_1_1': 0.5800 + 0.5800 * u,
        'ii_1_2': -0.9600 + 0.0500 * u,
        'ii_1_3': -0.7800 + 0.7800 * u,
        'ii_2_1': 2.2689 - 0.4689 * u,
        'iii_1_1': 0.5800 + 0.5800 * u,
        'iii_1_2': 0.9600 - 0.0500 * u,
        'iii_1_3': 0.7800 - 0.7800 * u,
        'iii_2_1': 2.2689 - 0.4689 * u,
    })
    for name, angle in angles.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            qadr = model.jnt_qposadr[jid]
            data.qpos[qadr] = float(np.clip(angle, model.jnt_range[jid, 0], model.jnt_range[jid, 1]))
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0


def run_interactive_viewer(
    model_path: Path,
    skin_model_path: Path | None = None,
    display: str = 'overlay',
    frame: str = 'body',
    transparent: bool = True,
    sim_time: float = 60.0,
):
    """Launch interactive 3D GUI visualizer — prone head pitch, inverse dynamics.

    Matches the structure of ``visualize_baby_mujoco.py``:
    - Model placed on the ground using ``model.qpos0`` (prone, tummy-down).
    - ``mj_forward`` + ``mj_inverse`` compute required joint torques each step
      (true inverse dynamics: prescribed kinematics → joint torque demand).
    - Body coordinate frames and transparency are on by default.
    - ``dt``-based time loop at the model's integration timestep.

    Parameters
    ----------
    model_path:
        Physics MJCF (``models/infant_4mo_M.xml``).
    skin_model_path:
        Optional ``_smil.xml`` for visual rendering. Kinematics are still
        driven from the physics model; the skin model just supplies rendering.
    display:
        Visual mode when skin is loaded: ``skin``, ``segments``, or ``overlay``.
    frame:
        Coordinate frame overlay: ``body``, ``geom``, ``site``, or ``none``.
    transparent:
        Enable transparency so inner coordinate frames are visible.
    sim_time:
        Viewer runs for at most this many simulated seconds (default 60 s).
    """
    try:
        import mujoco.viewer
    except ImportError:
        print("[WARNING] mujoco.viewer module not available in current environment.")
        return

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"\n[INFO] Loading MuJoCo model: {model_path}")
    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)

    # Match framelength/width from visualize_baby_mujoco.py
    m.vis.scale.framelength = 0.35
    m.vis.scale.framewidth = 0.02

    dt = m.opt.timestep
    steps = int(sim_time / dt)

    print(f"[INFO] Model Details:")
    print(f"  nq={m.nq}  nv={m.nv}  njnt={m.njnt}  nbody={m.nbody}  nu={m.nu}")
    print(f"  Timestep: {dt:.4f} s   Sim duration: {sim_time:.1f} s")

    # Prone ("tummy time") ground placement per ADR-0003:
    set_tummy_time_pose(m, d)
    mujoco.mj_forward(m, d)

    # ---- skin model (optional display layer) --------------------------------
    display_m = m
    display_d = d
    set_display_fn = None
    if skin_model_path is None:
        default_smil = Path(__file__).resolve().parents[1] / "models" / "infant_4mo_M_smil.xml"
        if default_smil.is_file():
            skin_model_path = default_smil

    if skin_model_path is not None:
        if not Path(skin_model_path).is_file():
            print(f"[WARNING] Skin model not found: {skin_model_path}. Falling back to physics model.")
        else:
            try:
                import importlib.util, sys as _sys
                skin_dir = Path(__file__).resolve().parents[1] / 'skin'
                spec = importlib.util.spec_from_file_location(
                    'view_smil_skin', skin_dir / 'view_smil_skin.py')
                vsmod = importlib.util.module_from_spec(spec)
                _sys.modules.setdefault('view_smil_skin', vsmod)
                spec.loader.exec_module(vsmod)
                set_display_fn = vsmod.set_display
                display_m = mujoco.MjModel.from_xml_path(str(skin_model_path))
                display_m.vis.scale.framelength = 0.35
                display_m.vis.scale.framewidth = 0.02
                display_d = mujoco.MjData(display_m)
                mujoco.mj_resetData(display_m, display_d)
                display_d.qpos[:] = d.qpos
                mujoco.mj_forward(display_m, display_d)
                print(f"[INFO] Rendering with skin model: {skin_model_path}")
            except Exception as e:
                print(f"[WARNING] Could not load skin model ({e}). Falling back to physics model.")

    # Configure visual transparency: segment blocks transparent, skin opaque
    if transparent:
        # Make segment blocks transparent in the display model (and physics model)
        for model_to_style in {m, display_m}:
            # Materials 1..N (exclude floor mat 0)
            for mat_idx in range(1, model_to_style.nmat):
                model_to_style.mat_rgba[mat_idx, 3] = 0.35
            # Geoms 1..N (exclude floor geom 0)
            for geom_idx in range(1, model_to_style.ngeom):
                model_to_style.geom_rgba[geom_idx, 3] = 0.35
        # Ensure skin mesh remains fully opaque
        if display_m.nskin > 0:
            display_m.skin_rgba[:, 3] = 1.0

    # ---- resolve head pitch joint ----------------------------------------
    jnt_name = "i_1_1"
    jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
    if jnt_id == -1:
        jnt_name = "hd_pitch"
        jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
    if jnt_id == -1:
        raise ValueError("Joint 'i_1_1' (or 'hd_pitch') not found in model.")
    qpos_idx = m.jnt_qposadr[jnt_id]
    dof_idx  = m.jnt_dofadr[jnt_id]

    frame_map = {
        'body': mujoco.mjtFrame.mjFRAME_BODY,
        'geom': mujoco.mjtFrame.mjFRAME_GEOM,
        'site': mujoco.mjtFrame.mjFRAME_SITE,
        'none': mujoco.mjtFrame.mjFRAME_NONE,
    }
    frame_type = frame_map.get(frame.lower(), mujoco.mjtFrame.mjFRAME_BODY)

    print("\n===================================================================")
    print("   LAUNCHING INTERACTIVE 3D MUJOCO VISUALIZER (PRONE HEAD PITCH)  ")
    print("===================================================================")
    print("  Coordinate Frame Axes (body frames):")
    print("    RED  = Local +X (Mediolateral, Left → Right)")
    print("    GREEN= Local +Y (Anteroposterior, Back → Front)")
    print("    BLUE = Local +Z (Longitudinal, Toes → Head)")
    print("  Simulation: Inverse dynamics — prescribed kinematics → joint torques")
    print("  Motion: Upward head pitch 0° → 45° (prone play)")
    if skin_model_path is not None and set_display_fn is not None:
        print(f"  Skin display: {display}   Keys: 1=skin  2=segments  3=overlay")
    print("  Spacebar: Pause  |  F: Cycle frames  |  Close window to exit")
    print("===================================================================\n")

    def apply_display_mode(m_disp, opt, mode):
        if set_display_fn is not None:
            set_display_fn(m_disp, opt, mode)
        # Custom transparency rule: skin is non-transparent (opaque), blocks are transparent
        if m_disp.nskin > 0:
            m_disp.skin_rgba[:, 3] = 1.0  # Skin always fully opaque
        if transparent:
            for m_obj in {m, m_disp}:
                for mat_idx in range(1, m_obj.nmat):
                    m_obj.mat_rgba[mat_idx, 3] = 0.35
                for geom_idx in range(1, m_obj.ngeom):
                    m_obj.geom_rgba[geom_idx, 3] = 0.35

    try:
        with mujoco.viewer.launch_passive(display_m, display_d) as viewer:
            with viewer.lock():
                viewer.opt.frame = frame_type
                # Keep global mjVIS_TRANSPARENT off so skin is not forced translucent,
                # while geom/mat alpha=0.35 provides transparent segment blocks
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0
                apply_display_mode(display_m, viewer.opt, display)

            # Camera — match visualize_baby_mujoco.py optimal prone viewpoint
            viewer.cam.lookat[0] = -0.08
            viewer.cam.lookat[1] = 0.20
            viewer.cam.lookat[2] = 0.05
            viewer.cam.distance = 0.90
            viewer.cam.elevation = -25.0
            viewer.cam.azimuth = 140.0

            # Key callbacks (1/2/3 for skin modes)
            pending = []
            def _key(k): pending.append(k)

            for i in range(steps):
                if not viewer.is_running():
                    print("[INFO] Viewer closed by user.")
                    break

                t = i * dt

                # Prescribed kinematics: prone tummy-time pose (ADR-0003) + back & neck motion
                # Head pitch sweeps 0 → ~22.5° → 0 at 0.3 Hz, accompanied by gentle back extension & head yaw/tilt
                phase = 0.5 * (1.0 - math.cos(2.0 * math.pi * 0.3 * t))
                head_angle = float(np.clip(0.7854 * phase,
                                           m.jnt_range[jnt_id, 0],
                                           m.jnt_range[jnt_id, 1]))
                back_angle = 0.15 * phase
                yaw_angle = 0.35 * math.sin(2.0 * math.pi * 0.15 * t) * phase
                roll_angle = 0.15 * math.sin(2.0 * math.pi * 0.30 * t) * phase

                set_tummy_time_pose(
                    m, d,
                    back_ext=back_angle,
                    neck_pitch=head_angle,
                    neck_roll=roll_angle,
                    neck_yaw=yaw_angle,
                )

                # Forward kinematics → positions, Jacobians, bias forces
                mujoco.mj_forward(m, d)

                # Inverse dynamics: given prescribed accelerations (all zero here —
                # quasi-static), compute the joint torques required to maintain the
                # current configuration against gravity and passive forces.
                mujoco.mj_inverse(m, d)  # writes d.qfrc_inverse

                # Copy kinematic state to display model (skin layer)
                if display_m is not m:
                    display_d.qpos[:] = d.qpos
                    display_d.qvel[:] = d.qvel
                    mujoco.mj_forward(display_m, display_d)

                with viewer.lock():
                    while pending:
                        k = pending.pop(0)
                        if k in [49, 50, 51]:
                            apply_display_mode(display_m, viewer.opt,
                                               ['skin', 'segments', 'overlay'][k - 49])

                viewer.sync()
                time.sleep(dt)

    except RuntimeError as err:
        if "mjpython" in str(err):
            print("\n" + "=" * 70)
            print("  [ERROR] On macOS, interactive visualization requires 'mjpython'.")
            print("  Please run using:")
            print("    conda activate mujoco_env")
            print("    mjpython simulations/test_prone_head_pitch_mujoco.py --visualize")
            print("  or directly:")
            print("    /opt/homebrew/Caskroom/miniforge/base/envs/mujoco_env/bin/mjpython simulations/test_prone_head_pitch_mujoco.py --visualize")
            print("=" * 70 + "\n")
            sys.exit(1)
        raise err

    print("Visualization session completed.")


def render_simulation_video(m: mujoco.MjModel, d: mujoco.MjData, qpos_idx: int, output_video: Path):
    """Render offscreen frames and compile an MP4/GIF animation of the prone head extension."""
    try:
        import imageio
    except ImportError:
        print("  [Note] imageio not installed; skipping MP4/GIF animation export.")
        return

    print(f"  Rendering video animation to: {output_video} ...")
    renderer = mujoco.Renderer(m, height=480, width=640)
    
    # Camera setup: view side of infant
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat = [-0.08, 0.25, 0.05]
    camera.distance = 0.80
    camera.elevation = -12.0
    camera.azimuth = 135.0

    frames = []
    num_frames = 60
    for i in range(num_frames):
        phase = math.sin(math.pi * i / num_frames)
        rad = 0.785398 * phase

        set_tummy_time_pose(m, d, back_ext=0.0, neck_pitch=rad)
        mujoco.mj_forward(m, d)

        renderer.update_scene(d, camera=camera)
        pixels = renderer.render()
        frames.append(pixels)

    output_video.parent.mkdir(parents=True, exist_ok=True)
    if output_video.suffix == '.mp4':
        try:
            imageio.mimsave(str(output_video), frames, fps=25, codec='libx264')
        except Exception:
            gif_path = output_video.with_suffix('.gif')
            imageio.mimsave(str(gif_path), frames, fps=20)
            print(f"  [Saved GIF animation fallback] {gif_path}")
    elif output_video.suffix == '.gif':
        imageio.mimsave(str(output_video), frames, fps=20)
    print(f"  [SUCCESS] Animation saved to: {output_video}")


def run_prone_head_pitch_simulation(model_path: Path, output_fig: Path, output_csv: Path, output_video: Path | None = None) -> dict:
    """Run upward head pitch sweep in prone orientation."""
    if not model_path.is_file():
        raise FileNotFoundError(f"MuJoCo model not found at: {model_path}")

    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)

    jnt_name = "i_1_1"
    jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
    if jnt_id == -1:
        jnt_name = "hd_pitch"
        jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
    if jnt_id == -1:
        raise ValueError("Joint 'i_1_1' (or 'hd_pitch') not found in model.")

    qpos_idx = m.jnt_qposadr[jnt_id]
    dof_idx  = m.jnt_dofadr[jnt_id]

    body_hd_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "HD")
    if body_hd_id == -1:
        body_hd_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "o_hd")
    body_p_id  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "P")
    if body_p_id == -1:
        body_p_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "i_p")
    body_t_id  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "T")
    if body_t_id == -1:
        body_t_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "i_t")

    m_head = float(m.body_mass[body_hd_id])
    g = abs(float(m.opt.gravity[2])) if abs(float(m.opt.gravity[2])) > 0 else 9.80665

    # Test sweep across upward extension angle values (0 to 45 deg)
    pitch_angles_deg = np.linspace(0.0, 45.0, 46)
    pitch_angles_rad = np.radians(pitch_angles_deg)

    results = {
        'pitch_deg': [],
        'pitch_rad': [],
        'head_z_cm': [],
        'head_elevation_gain_cm': [],
        'torque_sim_Nm': [],
        'torque_analytical_Nm': [],
        'head_pos_x': [],
        'head_pos_y': [],
        'head_pos_z': [],
        'neck_pos_x': [],
        'neck_pos_y': [],
        'neck_pos_z': [],
        'trunk_pos_x': [],
        'trunk_pos_y': [],
        'trunk_pos_z': [],
        'pelvis_pos_x': [],
        'pelvis_pos_y': [],
        'pelvis_pos_z': []
    }

    # Evaluate neutral baseline in canonical tummy-time pose (ADR-0003) to find neck-to-COM lever arm
    set_tummy_time_pose(m, d, back_ext=0.0, neck_pitch=0.0)
    mujoco.mj_forward(m, d)
    neck_pos_0 = d.xpos[body_hd_id].copy()
    head_com_0 = d.xipos[body_hd_id].copy()
    lever_arm_0 = float(np.linalg.norm(head_com_0 - neck_pos_0))
    z_baseline_cm = float(head_com_0[2] * 100.0)

    for deg, rad in zip(pitch_angles_deg, pitch_angles_rad):
        set_tummy_time_pose(m, d, back_ext=0.0, neck_pitch=float(rad))

        mujoco.mj_forward(m, d)

        head_com = d.xipos[body_hd_id]
        head_z_cm = float(head_com[2] * 100.0)
        elevation_gain_cm = head_z_cm - z_baseline_cm

        tau_sim = float(abs(d.qfrc_bias[dof_idx]))
        tau_analytical = float(m_head * g * lever_arm_0 * math.cos(abs(rad)))

        results['pitch_deg'].append(deg)
        results['pitch_rad'].append(rad)
        results['head_z_cm'].append(head_z_cm)
        results['head_elevation_gain_cm'].append(elevation_gain_cm)
        results['torque_sim_Nm'].append(tau_sim)
        results['torque_analytical_Nm'].append(tau_analytical)
        results['head_pos_x'].append(float(head_com[0]))
        results['head_pos_y'].append(float(head_com[1]))
        results['head_pos_z'].append(float(head_com[2]))
        results['neck_pos_x'].append(float(d.xpos[body_hd_id][0]))
        results['neck_pos_y'].append(float(d.xpos[body_hd_id][1]))
        results['neck_pos_z'].append(float(d.xpos[body_hd_id][2]))
        results['trunk_pos_x'].append(float(d.xpos[body_t_id][0]))
        results['trunk_pos_y'].append(float(d.xpos[body_t_id][1]))
        results['trunk_pos_z'].append(float(d.xpos[body_t_id][2]))
        results['pelvis_pos_x'].append(float(d.xpos[body_p_id][0]))
        results['pelvis_pos_y'].append(float(d.xpos[body_p_id][1]))
        results['pelvis_pos_z'].append(float(d.xpos[body_p_id][2]))

    for k in results:
        results[k] = np.array(results[k])

    # -------------------------------------------------------------------------
    # Save Results CSV
    # -------------------------------------------------------------------------
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Pitch_Angle_deg', 'Pitch_Angle_rad', 'Head_COM_Z_cm',
            'Elevation_Gain_cm', 'Torque_MuJoCo_Nm', 'Torque_Analytical_Nm', 'Torque_Residual_Nm'
        ])
        for i in range(len(pitch_angles_deg)):
            writer.writerow([
                f"{results['pitch_deg'][i]:.2f}",
                f"{results['pitch_rad'][i]:.4f}",
                f"{results['head_z_cm'][i]:.3f}",
                f"{results['head_elevation_gain_cm'][i]:.3f}",
                f"{results['torque_sim_Nm'][i]:.5f}",
                f"{results['torque_analytical_Nm'][i]:.5f}",
                f"{abs(results['torque_sim_Nm'][i] - results['torque_analytical_Nm'][i]):.6f}"
            ])

    # -------------------------------------------------------------------------
    # Plot Publication-Quality Verification Figure
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    # Panel A: Head COM Elevation vs Pitch Angle
    axA = axes[0]
    axA.plot(results['pitch_deg'], results['head_z_cm'], color='#1f77b4', lw=2.5, label='Head COM Height (cm)')
    axA.plot(results['pitch_deg'], results['head_elevation_gain_cm'], color='#2ca02c', lw=2, ls='--', label='Elevation Gain $\Delta z$ (cm)')
    axA.axvline(20.0, color='#ff7f0e', ls=':', lw=2, alpha=0.8, label='AIMS Propping Ref (20°)')
    axA.set_xlabel('Head Pitch Angle (degrees)\n[Positive = Upward Head Extension]', fontsize=11, fontweight='bold')
    axA.set_ylabel('Head Elevation (cm)', fontsize=11, fontweight='bold')
    axA.set_title('A. Head Elevation in Prone Play\n(Perpendicular to Gravity)', fontsize=12, fontweight='bold')
    axA.grid(True, alpha=0.3)
    axA.legend(loc='upper left', frameon=True)

    # Panel B: Gravitational Torque Demand vs Analytical Model
    axB = axes[1]
    axB.plot(results['pitch_deg'], results['torque_sim_Nm'], 'o-', color='#d62728', lw=2, markersize=4, label='MuJoCo Gravity Bias ($g(q)$)')
    axB.plot(results['pitch_deg'], results['torque_analytical_Nm'], '--', color='#2ca02c', lw=2.5, label='Analytical Model ($m g d \cos\\theta$)')
    r2 = 1.0 - np.sum((results['torque_sim_Nm'] - results['torque_analytical_Nm'])**2) / np.sum((results['torque_sim_Nm'] - np.mean(results['torque_sim_Nm']))**2)
    axB.set_xlabel('Head Pitch Angle (degrees)', fontsize=11, fontweight='bold')
    axB.set_ylabel('Neck Extensor Torque (N·m)', fontsize=11, fontweight='bold')
    axB.set_title(f'B. Extensor Neck Torque\n(Analytical Concordance $R^2={r2:.5f}$)', fontsize=12, fontweight='bold')
    axB.grid(True, alpha=0.3)
    axB.legend(loc='lower left', frameon=True)

    # Panel C: Multi-Pose Stick Posture in Sagittal Plane (Y-Z plane in world)
    axC = axes[2]
    # In world, longitudinal spine is along Y, floor is Z=0, head lifts upward into +Z
    sample_indices = [0, 15, 30, 45]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Ground surface
    axC.axhline(0.0, color='gray', lw=2, ls='-', label='Mat/Floor Surface ($z=0$)')

    for idx, col in zip(sample_indices, colors):
        deg = results['pitch_deg'][idx]
        y_pts = [results['pelvis_pos_y'][idx], results['trunk_pos_y'][idx], results['neck_pos_y'][idx], results['head_pos_y'][idx]]
        z_pts = [results['pelvis_pos_z'][idx], results['trunk_pos_z'][idx], results['neck_pos_z'][idx], results['head_pos_z'][idx]]
        axC.plot(y_pts, z_pts, 'o-', color=col, lw=2.5, markersize=6, label=f'Pitch {deg:.0f}°')

    axC.set_xlabel('Anteroposterior Position (m)', fontsize=11, fontweight='bold')
    axC.set_ylabel('Vertical Elevation (m)', fontsize=11, fontweight='bold')
    axC.set_title('C. Prone Head Extension Trajectory\n(Sagittal View in MuJoCo)', fontsize=12, fontweight='bold')
    axC.set_aspect('equal')
    axC.grid(True, alpha=0.3)
    axC.legend(loc='upper left', fontsize=9, frameon=True)

    plt.tight_layout()
    output_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_fig, dpi=300)
    pdf_out = output_fig.with_suffix('.pdf')
    fig.savefig(pdf_out)
    plt.close(fig)
    print(f"  [SUCCESS] Vector PDF saved to: {pdf_out}")
    print(f"  [SUCCESS] Figure saved to: {output_fig}")
    print(f"  [SUCCESS] Simulation CSV saved to: {output_csv}")

    # Optional video rendering
    if output_video:
        render_simulation_video(m, d, qpos_idx, output_video)

    return {
        'max_elevation_gain_cm': float(np.max(results['head_elevation_gain_cm'])),
        'max_torque_Nm': float(np.max(results['torque_sim_Nm'])),
        'r2_concordance': float(r2),
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(description="Prone play upward head pitch simulation test.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/infant_4mo_M.xml"),
        help="Path to MuJoCo XML model (default: models/infant_4mo_M.xml)",
    )
    parser.add_argument(
        "--output-fig",
        type=Path,
        default=Path("simulations/validation_prone_head_pitch_mujoco.png"),
        help="Output path for validation figure",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("simulations/validation_prone_head_pitch_mujoco.csv"),
        help="Output path for simulation CSV",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=None,
        help="Optional path for animation video (.mp4 or .gif)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Launch interactive 3D visualizer in window",
    )
    parser.add_argument(
        "--skin-model",
        type=Path,
        default=None,
        help="Optional path to a SMIL skin model XML (e.g. models/infant_4mo_M_smil.xml). "
             "When provided the interactive viewer renders the skin instead of the segment blocks."
    )
    parser.add_argument(
        "--display",
        choices=["skin", "segments", "overlay"],
        default="overlay",
        help="Visual mode when --skin-model is given: 'skin' (surface only), "
             "'segments' (block + frames), or 'overlay' (both). Default: overlay.",
    )
    parser.add_argument(
        "--frame",
        type=str,
        choices=["body", "geom", "site", "none"],
        default="body",
        help="Coordinate frame overlay for interactive viewer (default: body)",
    )
    parser.add_argument(
        "--opaque",
        action="store_true",
        help="Disable transparency in interactive viewer (by default bodies are transparent so inner frames are visible)",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=60.0,
        help="Interactive viewer run duration in seconds (default: 60.0)",
    )
    args = parser.parse_args()

    if args.visualize:
        run_interactive_viewer(
            args.model,
            skin_model_path=args.skin_model,
            display=args.display,
            frame=args.frame,
            transparent=not args.opaque,
            sim_time=args.time,
        )
        return

    res = run_prone_head_pitch_simulation(args.model, args.output_fig, args.output_csv, args.output_video)
    print("\n--- PRONE PLAY SIMULATION TEST SUMMARY ---")
    print(f"  Model File:              {args.model}")
    print(f"  Max Head Elevation Gain: {res['max_elevation_gain_cm']:.2f} cm")
    print(f"  Max Neck Extensor Torque:{res['max_torque_Nm']:.4f} N·m")
    print(f"  Analytical Concordance:  R² = {res['r2_concordance']:.6f}")
    print("-------------------------------------------\n")


if __name__ == "__main__":
    main()
