#!/usr/bin/env python3
"""Generate a MuJoCo XML (MJCF) infant model matching metaSegment & metaDOF_baby.

Part of the Baby MoCap EMG Biomechanical Modeling Suite.
Valeriya Gritsenko & Antigravity (2026).

Standardized Anatomical Convention (ADR-0001):
  - Coordinate System: Right-handed Cartesian (X x Y = Z)
  - X: Mediolateral axis, pointing from Left to Right shoulder ([1, 0, 0], Axis _1)
  - Y: Anteroposterior axis, pointing from Back to Front / Tummy ([0, 1, 0], Axis _2)
  - Z: Longitudinal axis, pointing from Toes to Head ([0, 0, 1], Axis _3)
  - Canonical 24 articulated DOFs across 6 anatomical chains:
      * Chain O & I (Axial): i_1_1, i_1_2, i_1_3 (neck), i_2_1 (spine back hinge)
      * Chain II (Right Arm): ii_1_1, ii_1_2, ii_1_3 (shoulder), ii_2_1 (elbow), ii_3_2 (wrist flex/ext about Y)
      * Chain III (Left Arm): iii_1_1, iii_1_2, iii_1_3 (shoulder), iii_2_1 (elbow), iii_3_2 (wrist flex/ext about Y)
      * Chain IV (Right Leg): iv_1_1, iv_1_2, iv_1_3 (hip), iv_2_1 (knee), iv_3_1 (ankle)
      * Chain V (Left Leg): v_1_1, v_1_2, v_1_3 (hip), v_2_1 (knee), v_3_1 (ankle)
  - Global Axis Alignment: All _1 axes rotate about [1, 0, 0], all _2 rotate about [0, 1, 0], all _3 rotate about [0, 0, 1].
"""

from __future__ import annotations
from dataclasses import dataclass
import argparse
import csv
import math
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET


LEGACY_TO_CANONICAL_SEGMENTS = {
    'P': 'i_p', 'T': 'i_t', 'HD': 'o_hd',
    'H_R': 'ii_h', 'RU_R': 'ii_ru', 'MC_R': 'ii_mcp',
    'H_L': 'iii_h', 'RU_L': 'iii_ru', 'MC_L': 'iii_mcp',
    'F_R': 'iv_f', 'TB_R': 'iv_tb', 'FT_R': 'iv_ft',
    'F_L': 'v_f', 'TB_L': 'v_tb', 'FT_L': 'v_ft',
}

CANONICAL_TO_LEGACY_SEGMENTS = {v: k for k, v in LEGACY_TO_CANONICAL_SEGMENTS.items()}

LEGACY_TO_CANONICAL_DOFS = {
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

CANONICAL_TO_LEGACY_DOFS = {v: k for k, v in LEGACY_TO_CANONICAL_DOFS.items()}
# Support both canonical _1 (X-axis flex/ext) and legacy _2 DOF naming
CANONICAL_TO_LEGACY_DOFS['ii_3_2'] = 'ra_wr_e_f'
CANONICAL_TO_LEGACY_DOFS['iii_3_2'] = 'la_wr_e_f'
LEGACY_TO_CANONICAL_DOFS['ra_wr_e_f'] = 'ii_3_2'
LEGACY_TO_CANONICAL_DOFS['la_wr_e_f'] = 'iii_3_2'


@dataclass
class Segment:
    name: str
    block_type: str
    width: float   # Dim1 (X: Mediolateral)
    depth: float   # Dim2 (Y: Anteroposterior)
    length: float  # Dim3 (Z: Longitudinal)
    mass: float
    com: tuple[float, float, float]
    inertia: tuple[float, float, float, float, float, float]
    note: str


def parse_vector_string(s: str) -> list[float]:
    s = s.strip().strip('[]()')
    parts = re.split(r'[\s,]+', s)
    return [float(p) for p in parts if p]


def load_metasegment(csv_path: Path) -> dict[str, Segment]:
    """Load segments supporting both canonical and legacy naming schemas."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"metaSegment CSV not found: {csv_path}")

    segments: dict[str, Segment] = {}
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_name = (row.get('sBody') or row.get('segment_name') or row.get('name') or '').strip()
            block = (row.get('sBlock') or row.get('shape_type') or 'box').strip()
            
            if 'nDim1' in row:
                w = float(row['nDim1'])
                d = float(row['nDim2'])
                l = float(row['nDim3'])
            else:
                w = float(row.get('width_m', 0.1))
                d = float(row.get('depth_m', 0.1) or 0.1)
                l = float(row.get('length_m', 0.1))
                
            mass = float(row.get('nMass') or row.get('mass_kg', 1.0))
            
            if 'sCOM' in row:
                com_vals = parse_vector_string(row['sCOM'])
                com = (com_vals[0], com_vals[1], com_vals[2]) if len(com_vals) >= 3 else (0.0, 0.0, 0.0)
            else:
                com = (float(row.get('com_x', 0)), float(row.get('com_y', 0)), float(row.get('com_z', 0)))
                
            if 'sInertia' in row:
                i_vals = parse_vector_string(row['sInertia'])
                if len(i_vals) == 6:
                    inertia = tuple(i_vals)
                elif len(i_vals) == 3:
                    inertia = (i_vals[0], i_vals[1], i_vals[2], 0.0, 0.0, 0.0)
                else:
                    inertia = (0.001, 0.001, 0.001, 0.0, 0.0, 0.0)
            else:
                inertia = (
                    float(row.get('Ixx', 0.001)),
                    float(row.get('Iyy', 0.001)),
                    float(row.get('Izz', 0.001)),
                    float(row.get('Ixy', 0.0)),
                    float(row.get('Ixz', 0.0)),
                    float(row.get('Iyz', 0.0)),
                )
                
            note = row.get('sNote', '')
            seg = Segment(
                name=raw_name,
                block_type=block.lower(),
                width=w,
                depth=d,
                length=l,
                mass=mass,
                com=com,
                inertia=inertia,
                note=note
            )
            # Store under raw name, canonical name, and legacy name for maximum compatibility
            segments[raw_name] = seg
            canonical = LEGACY_TO_CANONICAL_SEGMENTS.get(raw_name)
            if canonical:
                segments[canonical] = seg
            legacy = CANONICAL_TO_LEGACY_SEGMENTS.get(raw_name)
            if legacy:
                segments[legacy] = seg

    return segments


def resolve_metadof_path(specified_path: Path | None = None) -> Path:
    """Find the most relevant metaDOF_baby.csv file, checking candidate directories and modification times."""
    if specified_path and specified_path.is_file():
        return specified_path

    candidates = [
        Path("baby_anthropometry/metaDOF_baby.csv"),
        Path("generators/metaDOF_baby.csv"),
        Path("shared_data/metaDOF_baby.csv"),
    ]
    existing = [p for p in candidates if p.is_file()]
    if not existing:
        if specified_path:
            raise FileNotFoundError(f"metaDOF CSV not found: {specified_path}")
        raise FileNotFoundError("Could not find metaDOF_baby.csv in baby_anthropometry/, generators/, or shared_data/")

    # Pick the most recently modified candidate
    newest = max(existing, key=lambda p: p.stat().st_mtime)
    return newest


def load_metadof(csv_path: Path) -> dict[str, dict]:
    """Load DOF parameters supporting both canonical and legacy naming schemas."""
    csv_path = resolve_metadof_path(csv_path)
    print(f"[INFO] Ingesting metaDOF from: {csv_path}")

    dofs = {}
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_dof = row['sDOF'].strip()
            parent = row['sBodyP'].strip()
            child = row['sBodyF'].strip()
            if raw_dof == 'rl_a_e_f' and child in ('F_R', 'iv_f'):
                child = 'FT_R'
            if raw_dof == 'll_a_e_f' and child in ('F_L', 'v_f'):
                child = 'FT_L'

            canonical_name = LEGACY_TO_CANONICAL_DOFS.get(raw_dof, raw_dof)
            legacy_name = CANONICAL_TO_LEGACY_DOFS.get(raw_dof, raw_dof)

            info = {
                'name': canonical_name,
                'legacy_name': legacy_name,
                'parent': parent,
                'child': child,
                'range_min': float(row['nRangeMin']),
                'range_max': float(row['nRangeMax']),
                'ic': float(row['nIC']) if row.get('nIC') and row['nIC'].strip() else 0.0,
                'viscosity': float(row.get('nViscosity', 0.01) or 0.01),
                'stiffness': float(row.get('nStiffness', 0.01) or 0.01),
                'slack_ang': float(row.get('nSlackAng', 0.0) or 0.0),
                'range_stiffness': float(row.get('nRangeStiffness', 0.05) or 0.05),
                'range_damping': float(row.get('nRangeDamping', 0.005) or 0.005),
                'range_transition': float(row.get('nRangeTransition', 0.0017) or 0.0017),
                'note': row.get('sNote', '')
            }
            # Store under both canonical and legacy names
            dofs[canonical_name] = info
            dofs[legacy_name] = info
            dofs[raw_dof] = info

    if 'i_2_1' in dofs:
        r_min = dofs['i_2_1']['range_min']
        r_max = dofs['i_2_1']['range_max']
        print(f"[INFO] Ingested joint i_2_1 (pelvis-to-trunk) range of motion: [{r_min}, {r_max}] rad")

    return dofs


def fmt(val: float) -> str:
    if abs(val) < 1e-15:
        val = 0.0
    return f"{val:.17g}"


def vec3(vals: tuple[float, float, float] | list[float]) -> str:
    return f"{fmt(vals[0])} {fmt(vals[1])} {fmt(vals[2])}"


def limb_radius(seg: Segment) -> float:
    """Return effective radial extent for cylindrical visual representation."""
    if seg.block_type == 'frustum of a cone':
        return (seg.width + seg.depth) / 4.0
    return seg.width / 2.0


def build_mujoco_xml(segments: dict[str, Segment], dofs: dict[str, dict], model_name: str = "baby_mocap_model") -> ET.ElementTree:
    """Construct MuJoCo MJCF XML tree matching standardized anatomical coordinate conventions (ADR-0001)."""
    root = ET.Element("mujoco", {"model": model_name})
    
    # Compiler
    ET.SubElement(root, "compiler", {"angle": "radian", "coordinate": "local", "inertiafromgeom": "false"})
    
    # Visual Global
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": "1920", "offheight": "1080"})
    ET.SubElement(visual, "scale", {"framelength": "0.35", "framewidth": "0.02"})
    
    # Defaults
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", {"damping": "0.01", "armature": "0.001", "frictionloss": "0.001"})
    ET.SubElement(default, "geom", {"contype": "1", "conaffinity": "1", "condim": "3", "friction": "0.8 0.1 0.1"})
    
    # Visual Assets
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", {
        "name": "grid", "type": "2d", "builtin": "checker",
        "width": "512", "height": "512", "rgb1": "0.9 0.9 0.9", "rgb2": "0.75 0.75 0.75"
    })
    ET.SubElement(asset, "material", {"name": "matplane", "texture": "grid", "texrepeat": "5 5", "reflectance": "0.1"})
    ET.SubElement(asset, "material", {"name": "mat_axial", "rgba": "0.85 0.85 0.85 1"})
    ET.SubElement(asset, "material", {"name": "mat_right", "rgba": "0.85 0.25 0.25 1"})
    ET.SubElement(asset, "material", {"name": "mat_left",  "rgba": "0.25 0.40 0.85 1"})
    
    # Worldbody
    wb = ET.SubElement(root, "worldbody")
    ET.SubElement(wb, "light", {"pos": "0 0 2.5", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(wb, "geom", {"name": "floor", "type": "plane", "size": "2 2 0.05", "material": "matplane"})
    
    def get_mat(b_name: str) -> str:
        if b_name.endswith('_R') or b_name.startswith('ii_') or b_name.startswith('iv_'): return "mat_right"
        if b_name.endswith('_L') or b_name.startswith('iii_') or b_name.startswith('v_'): return "mat_left"
        return "mat_axial"

    def add_joint(parent_body: ET.Element, dof_name: str, axis: tuple[float, float, float]):
        """Add joint with canonical name and ingested parameters."""
        info = dofs.get(dof_name, dofs.get(CANONICAL_TO_LEGACY_DOFS.get(dof_name), dofs.get(LEGACY_TO_CANONICAL_DOFS.get(dof_name), {
            "range_min": -1.57, "range_max": 1.57, "ic": 0.0,
            "viscosity": 0.01, "stiffness": 0.01, "slack_ang": 0.0
        })))
        # Use initial condition from metaDOF as spring reference
        spring_ref = info.get('ic', info.get('slack_ang', 0.0))
        ET.SubElement(parent_body, "joint", {
            "name": dof_name,
            "type": "hinge",
            "pos": "0 0 0",
            "axis": vec3(axis),
            "range": f"{fmt(info['range_min'])} {fmt(info['range_max'])}",
            "damping": fmt(info['viscosity']),
            "stiffness": fmt(info['stiffness']),
            "springref": fmt(spring_ref)
        })

    # Segment references (supporting both canonical and legacy keys)
    p_seg   = segments.get('i_p') or segments['P']
    t_seg   = segments.get('i_t') or segments['T']
    hd_seg  = segments.get('o_hd') or segments['HD']
    h_r_seg = segments.get('ii_h') or segments['H_R']
    ru_r_seg= segments.get('ii_ru') or segments['RU_R']
    mc_r_seg= segments.get('ii_mcp') or segments['MC_R']
    f_r_seg = segments.get('iv_f') or segments['F_R']
    tb_r_seg= segments.get('iv_tb') or segments['TB_R']
    ft_r_seg= segments.get('iv_ft') or segments['FT_R']
    
    h_l_seg = segments.get('iii_h') or segments['H_L']
    ru_l_seg= segments.get('iii_ru') or segments['RU_L']
    mc_l_seg= segments.get('iii_mcp') or segments['MC_L']
    f_l_seg = segments.get('v_f') or segments['F_L']
    tb_l_seg= segments.get('v_tb') or segments['TB_L']
    ft_l_seg= segments.get('v_ft') or segments['FT_L']

    # -------------------------------------------------------------------------
    # 1. Pelvis 'P' (Root of kinematic chain)
    # Origin: Back bottom right corner of the pelvis (0, 0, 0), close to right hip
    # Pelvis extends: X in [-W_P, 0], Y in [0, D_P], Z in [0, L_P]
    # Prone orientation: rotate about X by -90 deg (-1.570796 rad) so tummy faces floor (-Z)
    #
    # Ground clearance derivation:
    #   R_x(-90°)·[x,y,z]ᵀ = [x, z, -y]ᵀ
    #   Pelvis geom local center: (-W/2, D/2, L/2)
    #   After rotation, world Z center = p_ground_z - D/2
    #   World Z half-extent (from local Y): D/2
    #   Bottom world Z = p_ground_z - D/2 - D/2 = p_ground_z - D
    #   For bottom to be at z=0: p_ground_z = D_p  (+ small clearance)
    # -------------------------------------------------------------------------
    w_p, d_p, l_p = p_seg.width, p_seg.depth, p_seg.length
    p_ground_z = d_p + 0.005  # body origin height: bottom face sits 5 mm above floor
    
    b_P = ET.SubElement(wb, "body", {
        "name": "P",
        "pos": f"0 0 {fmt(p_ground_z)}",
        "euler": "-1.5707963267948966 0 0" # rotate pelvis about X -90 deg for tummy visualization
    })
    ET.SubElement(b_P, "freejoint", {"name": "root_freejoint"})
    ET.SubElement(b_P, "inertial", {
        "pos": f"{fmt(-w_p/2.0)} {fmt(d_p/2.0)} {fmt(l_p/2.0)}",
        "mass": fmt(p_seg.mass),
        "diaginertia": f"{fmt(p_seg.inertia[0])} {fmt(p_seg.inertia[1])} {fmt(p_seg.inertia[2])}"
    })
    ET.SubElement(b_P, "geom", {
        "name": "geom_P",
        "type": "box",
        "pos": f"{fmt(-w_p/2.0)} {fmt(d_p/2.0)} {fmt(l_p/2.0)}",
        "size": f"{fmt(w_p/2.0)} {fmt(d_p/2.0)} {fmt(l_p/2.0)}",
        "material": get_mat("P")
    })

    # -------------------------------------------------------------------------
    # 2. Thorax 'T': attached at top center of pelvis [-W_P/2, D_P/2, L_P]
    # Thorax local origin at bottom center (L5/S1 joint)
    # Back hinge joint: i_2_1 (legacy: pt_e_f) around X ([1, 0, 0])
    # -------------------------------------------------------------------------
    w_t, d_t, l_t = t_seg.width, t_seg.depth, t_seg.length
    b_T = ET.SubElement(b_P, "body", {
        "name": "T",
        "pos": f"{fmt(-w_p/2.0)} {fmt(d_p/2.0)} {fmt(l_p)}"
    })
    add_joint(b_T, "i_2_1", (1, 0, 0))
    ET.SubElement(b_T, "inertial", {
        "pos": f"0 0 {fmt(l_t/2.0)}",
        "mass": fmt(t_seg.mass),
        "diaginertia": f"{fmt(t_seg.inertia[0])} {fmt(t_seg.inertia[1])} {fmt(t_seg.inertia[2])}"
    })
    ET.SubElement(b_T, "geom", {
        "name": "geom_T",
        "type": "box",
        "pos": f"0 0 {fmt(l_t/2.0)}",
        "size": f"{fmt(w_t/2.0)} {fmt(d_t/2.0)} {fmt(l_t/2.0)}",
        "material": get_mat("T")
    })

    # -------------------------------------------------------------------------
    # 3. Head 'HD': attached at top center of thorax [0, 0, L_T]
    # Neck pivot at base of skull (C1 joint)
    # Canonical Cervical DOFs: i_1_1 (pitch: X), i_1_2 (roll: Y), i_1_3 (yaw: Z)
    # -------------------------------------------------------------------------
    w_hd, d_hd, l_hd = hd_seg.width, hd_seg.depth, hd_seg.length
    b_HD = ET.SubElement(b_T, "body", {
        "name": "HD",
        "pos": f"0 0 {fmt(l_t)}"
    })
    add_joint(b_HD, "i_1_1", (1, 0, 0))
    add_joint(b_HD, "i_1_2", (0, 1, 0))
    add_joint(b_HD, "i_1_3", (0, 0, 1))
    
    ET.SubElement(b_HD, "inertial", {
        "pos": f"0 0 {fmt(l_hd/2.0)}",
        "mass": fmt(hd_seg.mass),
        "diaginertia": f"{fmt(hd_seg.inertia[0])} {fmt(hd_seg.inertia[1])} {fmt(hd_seg.inertia[2])}"
    })
    # Head main ellipsoid
    ET.SubElement(b_HD, "geom", {
        "name": "geom_HD",
        "type": "ellipsoid",
        "pos": f"0 0 {fmt(l_hd/2.0)}",
        "size": f"{fmt(w_hd/2.0)} {fmt(d_hd/2.0)} {fmt(l_hd/2.0)}",
        "material": get_mat("HD")
    })
    # Visual Landmarks: Nose (+Y: Front), Right Ear (+X: RED), Left Ear (-X: BLUE)
    ET.SubElement(b_HD, "geom", {
        "name": "geom_HD_nose",
        "type": "box",
        "pos": f"0 {fmt(d_hd/2.0 + 0.006)} {fmt(l_hd/2.0)}",
        "size": "0.008 0.010 0.012",
        "rgba": "0.20 0.85 0.30 1"
    })
    ET.SubElement(b_HD, "geom", {
        "name": "geom_HD_ear_r",
        "type": "box",
        "pos": f"{fmt(w_hd/2.0 + 0.006)} 0 {fmt(l_hd/2.0)}",
        "size": "0.008 0.012 0.015",
        "rgba": "0.90 0.10 0.10 1" # RED right ear
    })
    ET.SubElement(b_HD, "geom", {
        "name": "geom_HD_ear_l",
        "type": "box",
        "pos": f"{fmt(-w_hd/2.0 - 0.006)} 0 {fmt(l_hd/2.0)}",
        "size": "0.008 0.012 0.015",
        "rgba": "0.10 0.30 0.90 1" # BLUE left ear
    })

    # -------------------------------------------------------------------------
    # 4. Right Arm: T -> H_R -> RU_R -> MC_R
    # Shoulder on +X (right lateral border)
    # Canonical DOFs: ii_1_1 (flex: X), ii_1_2 (ab: Y), ii_1_3 (rot: Z)
    # -------------------------------------------------------------------------
    sh_x_r = w_t / 2.0 + 0.15 * limb_radius(h_r_seg)
    sh_z_r = l_t - limb_radius(h_r_seg)
    b_H_R = ET.SubElement(b_T, "body", {
        "name": "H_R",
        "pos": f"{fmt(sh_x_r)} 0 {fmt(sh_z_r)}"
    })
    add_joint(b_H_R, "ii_1_1", (1, 0, 0)) # flexion
    add_joint(b_H_R, "ii_1_2", (0, 1, 0)) # abduction
    add_joint(b_H_R, "ii_1_3", (0, 0, 1)) # internal rotation
    ET.SubElement(b_H_R, "inertial", {
        "pos": f"0 0 {fmt(-h_r_seg.length/2.0)}",
        "mass": fmt(h_r_seg.mass),
        "diaginertia": f"{fmt(h_r_seg.inertia[0])} {fmt(h_r_seg.inertia[1])} {fmt(h_r_seg.inertia[2])}"
    })
    ET.SubElement(b_H_R, "geom", {
        "name": "geom_H_R",
        "type": "cylinder",
        "pos": f"0 0 {fmt(-h_r_seg.length/2.0)}",
        "size": f"{fmt(limb_radius(h_r_seg))} {fmt(h_r_seg.length/2.0)}",
        "material": get_mat("H_R")
    })

    # Right Elbow: ii_2_1 (elbow flexion around X)
    b_RU_R = ET.SubElement(b_H_R, "body", {
        "name": "RU_R",
        "pos": f"0 0 {fmt(-h_r_seg.length)}"
    })
    add_joint(b_RU_R, "ii_2_1", (1, 0, 0))
    ET.SubElement(b_RU_R, "inertial", {
        "pos": f"0 0 {fmt(-ru_r_seg.length/2.0)}",
        "mass": fmt(ru_r_seg.mass),
        "diaginertia": f"{fmt(ru_r_seg.inertia[0])} {fmt(ru_r_seg.inertia[1])} {fmt(ru_r_seg.inertia[2])}"
    })
    ET.SubElement(b_RU_R, "geom", {
        "name": "geom_RU_R",
        "type": "cylinder",
        "pos": f"0 0 {fmt(-ru_r_seg.length/2.0)}",
        "size": f"{fmt(limb_radius(ru_r_seg))} {fmt(ru_r_seg.length/2.0)}",
        "material": get_mat("RU_R")
    })

    # Right Wrist: ii_3_2 (wrist flexion/extension about Y [0, 1, 0], hand width axis, ADR-0001 Section 6.3)
    b_MC_R = ET.SubElement(b_RU_R, "body", {
        "name": "MC_R",
        "pos": f"0 0 {fmt(-ru_r_seg.length)}"
    })
    wrist_dof_r = "ii_3_2" if "ii_3_2" in dofs or "ii_3_2" in CANONICAL_TO_LEGACY_DOFS else "ii_3_1"
    add_joint(b_MC_R, wrist_dof_r, (0, 1, 0))
    ET.SubElement(b_MC_R, "inertial", {
        "pos": f"0 0 {fmt(-mc_r_seg.length/2.0)}",
        "mass": fmt(mc_r_seg.mass),
        "diaginertia": f"{fmt(mc_r_seg.inertia[0])} {fmt(mc_r_seg.inertia[1])} {fmt(mc_r_seg.inertia[2])}"
    })
    ET.SubElement(b_MC_R, "geom", {
        "name": "geom_MC_R",
        "type": "box",
        "pos": f"0 0 {fmt(-mc_r_seg.length/2.0)}",
        "size": f"{fmt(mc_r_seg.depth/2.0)} {fmt(mc_r_seg.width/2.0)} {fmt(mc_r_seg.length/2.0)}",
        "material": get_mat("MC_R")
    })

    # -------------------------------------------------------------------------
    # 5. Left Arm: T -> H_L -> RU_L -> MC_L
    # Shoulder on -X (left lateral border)
    # Canonical DOFs: iii_1_1 (flex: X), iii_1_2 (ab: Y), iii_1_3 (rot: Z)
    # -------------------------------------------------------------------------
    sh_x_l = -w_t / 2.0 - 0.15 * limb_radius(h_l_seg)
    sh_z_l = l_t - limb_radius(h_l_seg)
    b_H_L = ET.SubElement(b_T, "body", {
        "name": "H_L",
        "pos": f"{fmt(sh_x_l)} 0 {fmt(sh_z_l)}"
    })
    add_joint(b_H_L, "iii_1_1", (1, 0, 0))
    add_joint(b_H_L, "iii_1_2", (0, 1, 0))
    add_joint(b_H_L, "iii_1_3", (0, 0, 1))
    ET.SubElement(b_H_L, "inertial", {
        "pos": f"0 0 {fmt(-h_l_seg.length/2.0)}",
        "mass": fmt(h_l_seg.mass),
        "diaginertia": f"{fmt(h_l_seg.inertia[0])} {fmt(h_l_seg.inertia[1])} {fmt(h_l_seg.inertia[2])}"
    })
    ET.SubElement(b_H_L, "geom", {
        "name": "geom_H_L",
        "type": "cylinder",
        "pos": f"0 0 {fmt(-h_l_seg.length/2.0)}",
        "size": f"{fmt(limb_radius(h_l_seg))} {fmt(h_l_seg.length/2.0)}",
        "material": get_mat("H_L")
    })

    # Left Elbow: iii_2_1
    b_RU_L = ET.SubElement(b_H_L, "body", {
        "name": "RU_L",
        "pos": f"0 0 {fmt(-h_l_seg.length)}"
    })
    add_joint(b_RU_L, "iii_2_1", (1, 0, 0))
    ET.SubElement(b_RU_L, "inertial", {
        "pos": f"0 0 {fmt(-ru_l_seg.length/2.0)}",
        "mass": fmt(ru_l_seg.mass),
        "diaginertia": f"{fmt(ru_l_seg.inertia[0])} {fmt(ru_l_seg.inertia[1])} {fmt(ru_l_seg.inertia[2])}"
    })
    ET.SubElement(b_RU_L, "geom", {
        "name": "geom_RU_L",
        "type": "cylinder",
        "pos": f"0 0 {fmt(-ru_l_seg.length/2.0)}",
        "size": f"{fmt(limb_radius(ru_l_seg))} {fmt(ru_l_seg.length/2.0)}",
        "material": get_mat("RU_L")
    })

    # Left Wrist: iii_3_2 (wrist flexion/extension about Y [0, 1, 0], hand width axis, ADR-0001 Section 6.3)
    b_MC_L = ET.SubElement(b_RU_L, "body", {
        "name": "MC_L",
        "pos": f"0 0 {fmt(-ru_l_seg.length)}"
    })
    wrist_dof_l = "iii_3_2" if "iii_3_2" in dofs or "iii_3_2" in CANONICAL_TO_LEGACY_DOFS else "iii_3_1"
    add_joint(b_MC_L, wrist_dof_l, (0, 1, 0))
    ET.SubElement(b_MC_L, "inertial", {
        "pos": f"0 0 {fmt(-mc_l_seg.length/2.0)}",
        "mass": fmt(mc_l_seg.mass),
        "diaginertia": f"{fmt(mc_l_seg.inertia[0])} {fmt(mc_l_seg.inertia[1])} {fmt(mc_l_seg.inertia[2])}"
    })
    ET.SubElement(b_MC_L, "geom", {
        "name": "geom_MC_L",
        "type": "box",
        "pos": f"0 0 {fmt(-mc_l_seg.length/2.0)}",
        "size": f"{fmt(mc_l_seg.depth/2.0)} {fmt(mc_l_seg.width/2.0)} {fmt(mc_l_seg.length/2.0)}",
        "material": get_mat("MC_L")
    })

    # -------------------------------------------------------------------------
    # 6. Right Leg: P -> F_R -> TB_R -> FT_R
    # Right Hip at [-r_thigh, D_P/2, 0] so outer cylinder edge is flush with lateral right pelvis edge (0.0)
    # Canonical DOFs: iv_1_1 (flex: X), iv_1_2 (ab: Y), iv_1_3 (rot: Z)
    # -------------------------------------------------------------------------
    r_thigh_r = limb_radius(f_r_seg)
    hip_x_r = -r_thigh_r
    hip_y_r = d_p / 2.0
    b_F_R = ET.SubElement(b_P, "body", {
        "name": "F_R",
        "pos": f"{fmt(hip_x_r)} {fmt(hip_y_r)} 0"
    })
    add_joint(b_F_R, "iv_1_1", (1, 0, 0))
    add_joint(b_F_R, "iv_1_2", (0, 1, 0))
    add_joint(b_F_R, "iv_1_3", (0, 0, 1))
    ET.SubElement(b_F_R, "inertial", {
        "pos": f"0 0 {fmt(-f_r_seg.length/2.0)}",
        "mass": fmt(f_r_seg.mass),
        "diaginertia": f"{fmt(f_r_seg.inertia[0])} {fmt(f_r_seg.inertia[1])} {fmt(f_r_seg.inertia[2])}"
    })
    ET.SubElement(b_F_R, "geom", {
        "name": "geom_F_R",
        "type": "cylinder",
        "pos": f"0 0 {fmt(-f_r_seg.length/2.0)}",
        "size": f"{fmt(limb_radius(f_r_seg))} {fmt(f_r_seg.length/2.0)}",
        "material": get_mat("F_R")
    })

    # Right Knee: iv_2_1 (canonical axis [1, 0, 0] matching ADR-0001)
    b_TB_R = ET.SubElement(b_F_R, "body", {
        "name": "TB_R",
        "pos": f"0 0 {fmt(-f_r_seg.length)}"
    })
    add_joint(b_TB_R, "iv_2_1", (1, 0, 0))
    ET.SubElement(b_TB_R, "inertial", {
        "pos": f"0 0 {fmt(-tb_r_seg.length/2.0)}",
        "mass": fmt(tb_r_seg.mass),
        "diaginertia": f"{fmt(tb_r_seg.inertia[0])} {fmt(tb_r_seg.inertia[1])} {fmt(tb_r_seg.inertia[2])}"
    })
    ET.SubElement(b_TB_R, "geom", {
        "name": "geom_TB_R",
        "type": "cylinder",
        "pos": f"0 0 {fmt(-tb_r_seg.length/2.0)}",
        "size": f"{fmt(limb_radius(tb_r_seg))} {fmt(tb_r_seg.length/2.0)}",
        "material": get_mat("TB_R")
    })

    # Right Ankle: iv_3_1 (ankle pitch around X [1, 0, 0], foot projects along +Y, ADR-0001 Section 6.2)
    b_FT_R = ET.SubElement(b_TB_R, "body", {
        "name": "FT_R",
        "pos": f"0 0 {fmt(-tb_r_seg.length)}"
    })
    add_joint(b_FT_R, "iv_3_1", (1, 0, 0))
    # Foot segment shifted by the radius of the shank so the heel
    # is flush with the back surface of the shank cylinder:
    ft_com_y_r = ft_r_seg.length / 2.0 - limb_radius(tb_r_seg)
    ft_com_z_r = -ft_r_seg.depth / 2.0
    ET.SubElement(b_FT_R, "inertial", {
        "pos": f"0 {fmt(ft_com_y_r)} {fmt(ft_com_z_r)}",
        "mass": fmt(ft_r_seg.mass),
        "diaginertia": f"{fmt(ft_r_seg.inertia[0])} {fmt(ft_r_seg.inertia[1])} {fmt(ft_r_seg.inertia[2])}"
    })
    ET.SubElement(b_FT_R, "geom", {
        "name": "geom_FT_R",
        "type": "box",
        "pos": f"0 {fmt(ft_com_y_r)} {fmt(ft_com_z_r)}",
        "size": f"{fmt(ft_r_seg.width/2.0)} {fmt(ft_r_seg.length/2.0)} {fmt(ft_r_seg.depth/2.0)}",
        "material": get_mat("FT_R")
    })

    # -------------------------------------------------------------------------
    # 7. Left Leg: P -> F_L -> TB_L -> FT_L
    # Left Hip at [-(W_P - r_thigh), D_P/2, 0] so outer cylinder edge is flush with lateral left pelvis edge (-W_P)
    # Canonical DOFs: v_1_1 (flex: X), v_1_2 (ab: Y), v_1_3 (rot: Z)
    # -------------------------------------------------------------------------
    r_thigh_l = limb_radius(f_l_seg)
    hip_x_l = -w_p + r_thigh_l
    hip_y_l = d_p / 2.0
    b_F_L = ET.SubElement(b_P, "body", {
        "name": "F_L",
        "pos": f"{fmt(hip_x_l)} {fmt(hip_y_l)} 0"
    })
    add_joint(b_F_L, "v_1_1", (1, 0, 0))
    add_joint(b_F_L, "v_1_2", (0, 1, 0))
    add_joint(b_F_L, "v_1_3", (0, 0, 1))
    ET.SubElement(b_F_L, "inertial", {
        "pos": f"0 0 {fmt(-f_l_seg.length/2.0)}",
        "mass": fmt(f_l_seg.mass),
        "diaginertia": f"{fmt(f_l_seg.inertia[0])} {fmt(f_l_seg.inertia[1])} {fmt(f_l_seg.inertia[2])}"
    })
    ET.SubElement(b_F_L, "geom", {
        "name": "geom_F_L",
        "type": "cylinder",
        "pos": f"0 0 {fmt(-f_l_seg.length/2.0)}",
        "size": f"{fmt(limb_radius(f_l_seg))} {fmt(f_l_seg.length/2.0)}",
        "material": get_mat("F_L")
    })

    # Left Knee: v_2_1 (canonical axis [1, 0, 0])
    b_TB_L = ET.SubElement(b_F_L, "body", {
        "name": "TB_L",
        "pos": f"0 0 {fmt(-f_l_seg.length)}"
    })
    add_joint(b_TB_L, "v_2_1", (1, 0, 0))
    ET.SubElement(b_TB_L, "inertial", {
        "pos": f"0 0 {fmt(-tb_l_seg.length/2.0)}",
        "mass": fmt(tb_l_seg.mass),
        "diaginertia": f"{fmt(tb_l_seg.inertia[0])} {fmt(tb_l_seg.inertia[1])} {fmt(tb_l_seg.inertia[2])}"
    })
    ET.SubElement(b_TB_L, "geom", {
        "name": "geom_TB_L",
        "type": "cylinder",
        "pos": f"0 0 {fmt(-tb_l_seg.length/2.0)}",
        "size": f"{fmt(limb_radius(tb_l_seg))} {fmt(tb_l_seg.length/2.0)}",
        "material": get_mat("TB_L")
    })

    # Left Ankle: v_3_1 (canonical axis [1, 0, 0])
    b_FT_L = ET.SubElement(b_TB_L, "body", {
        "name": "FT_L",
        "pos": f"0 0 {fmt(-tb_l_seg.length)}"
    })
    add_joint(b_FT_L, "v_3_1", (1, 0, 0))
    # Foot segment shifted by the radius of the shank so the heel
    # is flush with the back surface of the shank cylinder:
    ft_com_y_l = ft_l_seg.length / 2.0 - limb_radius(tb_l_seg)
    ft_com_z_l = -ft_l_seg.depth / 2.0
    ET.SubElement(b_FT_L, "inertial", {
        "pos": f"0 {fmt(ft_com_y_l)} {fmt(ft_com_z_l)}",
        "mass": fmt(ft_l_seg.mass),
        "diaginertia": f"{fmt(ft_l_seg.inertia[0])} {fmt(ft_l_seg.inertia[1])} {fmt(ft_l_seg.inertia[2])}"
    })
    ET.SubElement(b_FT_L, "geom", {
        "name": "geom_FT_L",
        "type": "box",
        "pos": f"0 {fmt(ft_com_y_l)} {fmt(ft_com_z_l)}",
        "size": f"{fmt(ft_l_seg.width/2.0)} {fmt(ft_l_seg.length/2.0)} {fmt(ft_l_seg.depth/2.0)}",
        "material": get_mat("FT_L")
    })

    # -------------------------------------------------------------------------
    # Actuator Set (Position Actuators for all 24 articulated DOFs)
    # -------------------------------------------------------------------------
    act = ET.SubElement(root, "actuator")
    for j_elem in root.findall(".//worldbody//joint"):
        d_name = j_elem.get("name")
        if not d_name or d_name == "root_freejoint":
            continue
        d_info = dofs.get(d_name, dofs.get(CANONICAL_TO_LEGACY_DOFS.get(d_name), {
            "range_min": -1.57, "range_max": 1.57
        }))
        rmin = d_info.get('range_min', -1.57)
        rmax = d_info.get('range_max', 1.57)
        ET.SubElement(act, "position", {
            "name": f"act_{d_name}",
            "joint": d_name,
            "kp": "5.0",
            "kv": "0.5",
            "ctrlrange": f"{fmt(rmin)} {fmt(rmax)}"
        })

    return ET.ElementTree(root)


_DEFAULT_SMIL = Path.home() / 'Documents/SciDocs/Projects/Modelling/smil/smil_web.pkl'


def generate_mujoco_model(metasegment_path: Path, metadof_path: Path, output_path: Path, model_name: str = "baby_mocap_model") -> Path:
    segments = load_metasegment(metasegment_path)
    dofs = load_metadof(metadof_path)
    tree = build_mujoco_xml(segments, dofs, model_name)
    ET.indent(tree, space="    ")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def attach_skin(physics_xml: Path, smil_path: Path) -> Path:
    """Attach a SMIL skin to an existing physics MJCF XML and return the skin model path.

    Delegates to skin/build_smil_skin.py's build() function, which:
      - Fits the neutral SMIL template to the model's segment geometry.
      - Writes a ``<name>_smil.xml`` alongside the source model.
      - Saves skin assets (NPZ, OBJ, manifest JSON) under ``models/skin_assets/``.

    The source XML must not already contain skin elements or file includes
    (the self-contained primitive MJCF structure produced by this generator
    satisfies that requirement automatically).

    Parameters
    ----------
    physics_xml:
        Path to the generated physics MJCF (``models/infant_4mo_M.xml``).
    smil_path:
        Path to the locally licensed SMIL ``.pkl`` asset.

    Returns
    -------
    Path to the generated skin model XML (``models/infant_4mo_M_smil.xml``).
    """
    import importlib.util, sys as _sys

    # Locate build_smil_skin.py relative to this generator file.
    skin_script = Path(__file__).resolve().parents[1] / 'skin' / 'build_smil_skin.py'
    if not skin_script.is_file():
        raise FileNotFoundError(
            f"SMIL skin builder not found at: {skin_script}\n"
            "Ensure skin/build_smil_skin.py is present in the project."
        )
    if not smil_path.is_file():
        raise FileNotFoundError(
            f"SMIL asset not found: {smil_path}\n"
            "Provide the locally licensed smil_web.pkl via --smil."
        )

    spec = importlib.util.spec_from_file_location('build_smil_skin', skin_script)
    mod = importlib.util.module_from_spec(spec)
    _sys.modules['build_smil_skin'] = mod
    spec.loader.exec_module(mod)

    stem = physics_xml.stem
    output_smil = physics_xml.parent / (stem + '_smil.xml')
    mod.build(physics_xml, smil_path, output_smil)
    return output_smil


def main():
    parser = argparse.ArgumentParser(description="Generate MuJoCo XML infant model matching metaSegment & metaDOF.")
    parser.add_argument("--metasegment", type=Path, default=Path("shared_data/metaSegment_4mo_M.csv"))
    parser.add_argument("--metadof", type=Path, default=Path("shared_data/metaDOF_baby.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/infant_4mo_M.xml"))
    parser.add_argument("--name", type=str, default="infant_4mo_M")
    parser.add_argument(
        "--skin",
        action="store_true",
        help="After generating the physics XML, attach the SMIL infant skin surface. "
             "Produces <output_stem>_smil.xml alongside the physics model."
    )
    parser.add_argument(
        "--smil",
        type=Path,
        default=_DEFAULT_SMIL,
        help="Path to the locally licensed SMIL .pkl asset "
             f"(default: {_DEFAULT_SMIL})"
    )
    args = parser.parse_args()

    out_file = generate_mujoco_model(args.metasegment, args.metadof, args.output, args.name)
    print(f"[SUCCESS] Wrote MuJoCo infant model to: {out_file}")

    if args.skin:
        skin_file = attach_skin(out_file, args.smil)
        print(f"[SUCCESS] Wrote SMIL skin model to:   {skin_file}")


if __name__ == "__main__":
    main()
