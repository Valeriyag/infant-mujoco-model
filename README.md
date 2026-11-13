# Infant Biomechanical Model for MuJoCo

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MuJoCo 3.1+](https://img.shields.io/badge/MuJoCo-3.1+-orange.svg)](https://mujoco.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

An open-source biomechanical modeling suite providing individualized 15-segment, 24-DOF computational models of human infants in **MuJoCo** with pediatric multi-tissue density scaling and continuous 3D skin envelopes.

---

## Repository Layout

```text
infant-mujoco-model/
├── .gitignore
├── LICENSE
├── README.md
├── generators/
│   └── generate_baby_mujoco_model.py
├── models/
│   ├── infant_4mo_M.xml
│   ├── infant_4mo_M_smil.xml
│   └── skin_assets/
├── shared_data/
│   ├── metaDOF_baby.csv
│   └── metaSegment_4mo_M.csv
└── simulations/
    ├── interactive_joint_control.py
    ├── test_prone_head_pitch_mujoco.py
    ├── visualize_baby_mujoco.py
    └── visualize_segment_frames_mujoco.py
```

---

## Quickstart: Generating the MuJoCo Model

Generate the canonical infant MuJoCo model from the standardized segment and DOF tables:

```bash
python generators/generate_baby_mujoco_model.py \
    --metasegment shared_data/metaSegment_4mo_M.csv \
    --metadof shared_data/metaDOF_baby.csv \
    --output models/infant_4mo_M.xml \
    --name infant_4mo_M
```

To automatically attach the continuous SMIL surface mesh:
```bash
python generators/generate_baby_mujoco_model.py \
    --metasegment shared_data/metaSegment_4mo_M.csv \
    --metadof shared_data/metaDOF_baby.csv \
    --output models/infant_4mo_M.xml \
    --name infant_4mo_M \
    --skin
```

---

## Interactive 3D Visualization: `visualize_baby_mujoco.py`

The primary interactive visualization entry point is `simulations/visualize_baby_mujoco.py`. It launches a passive MuJoCo 3D viewer rendering the baby model in the standard prone ("tummy time") resting posture (elbows and hands resting on the floor, stomach and pelvis on the mat, knees flexed at 30° with feet outward, and anti-gravity spinal/neck motion).

### Running the Viewer

```bash
python simulations/visualize_baby_mujoco.py --model models/infant_4mo_M.xml
```

### Command-Line Arguments and Options

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model` | `Path` | `models/infant_4mo_M.xml` | Path to the MuJoCo MJCF XML model file. |
| `--frame` | `str` | `body` | Coordinate frame visualization type. Options: `body` (displays 3D coordinate frames at each segment origin), `geom` (frames at geometry centers), `site` (frames at anatomical sites), or `none` (disables frame axes). |
| `--mode` | `str` | `tummy_time` | Kinematic and dynamic simulation mode:<br>• `tummy_time` (or `trunk_and_neck`): Kinematic developmental prone anti-gravity movement with active lumbar extension ($q_{\text{ext}} = 0\text{--}14.3^\circ$), sinusoidal head pitch ($q_{\text{pitch}} = 0\text{--}20.1^\circ$), cervical yaw ($q_{\text{yaw}} = \pm 25.8^\circ$), and cervical lateral tilt ($q_{\text{roll}} = \pm 11.5^\circ$).<br>• `head_pitch`: Pure sagittal head extension sweep ($0^\circ \text{ to } 45^\circ$) with fixed trunk.<br>• `step`: Active physics forward dynamic stepping from tummy-time initial pose.<br>• `static`: Fixed resting tummy-time posture without active kinematic driving. |
| `--posture` | `str` | `prone` | Initial postural configuration (preset to canonical prone "tummy time"). |
| `--opaque` | `flag` | `False` | Disables segment geom transparency. By default, geoms are rendered semi-transparently ($\alpha = 0.5$) so internal segment coordinate frames and joint centers are visible. Passing `--opaque` renders solid geometric surfaces. |
| `--time` | `float` | `60.0` | Total duration of the simulation / viewer session in seconds. |

### Example Usage Scenarios

1. **Standard Tummy-Time Motion with Coordinate Axes (Default):**
   ```bash
   python simulations/visualize_baby_mujoco.py
   ```

2. **Opaque Visual Surfaces without Internal Coordinate Frames:**
   ```bash
   python simulations/visualize_baby_mujoco.py --frame none --opaque
   ```

3. **Sagittal Head Extension Sweep (0° to 45° Pitch):**
   ```bash
   python simulations/visualize_baby_mujoco.py --mode head_pitch
   ```

4. **Static Prone Inspection with Body Reference Frames:**
   ```bash
   python simulations/visualize_baby_mujoco.py --mode static --frame body
   ```

5. **Forward Physics Stepping with Floor Contact Dynamics:**
   ```bash
   python simulations/visualize_baby_mujoco.py --mode step --time 120.0
   ```

---

## Segment Coordinate Frame Inspection: `visualize_segment_frames_mujoco.py`

To inspect local segment anatomical coordinate systems and verify body frame orientations:

```bash
# Canonical model with RGB segment axes
mjpython simulations/visualize_segment_frames_mujoco.py

# Continuous skin model
mjpython simulations/visualize_segment_frames_mujoco.py --model models/infant_4mo_M_smil.xml
```

### Coordinate Frame Color Legend:
- **RED Axis (+X)**: Mediolateral (Left $\rightarrow$ Right)
- **GREEN Axis (+Y)**: Anteroposterior (Back $\rightarrow$ Front / Tummy)
- **BLUE Axis (+Z)**: Longitudinal (Toes $\rightarrow$ Head)

### Interactive Controls:
- **`F`**: Cycle coordinate frame modes (`body` $\rightarrow$ `geom` $\rightarrow$ `site` $\rightarrow$ `none`)
- **`0`**: Reset camera to default perspective
- **`Spacebar`**: Pause / unpause physics stepping
- **Mouse Drag**: Rotate (Left Click), Zoom (Right Click), Pan (Middle Click / Scroll)

---

## Interactive Joint Control Viewer: `interactive_joint_control.py`

To interactively manipulate joint positions using GUI sliders in the MuJoCo passive viewer:

```bash
mjpython simulations/interactive_joint_control.py --model models/infant_4mo_M.xml
```

Use the **Control** panel in the right-hand sidebar to adjust joint angles across all 24 DOFs in real time.

---

## Coordinate Conventions

- **+X**: Mediolateral axis pointing from Left to Right shoulder (`[1, 0, 0]`)
- **+Y**: Anteroposterior axis pointing from Back to Front / Tummy (`[0, 1, 0]`)
- **+Z**: Longitudinal axis pointing from Toes to Head (`[0, 0, 1]`)

---

## Citation

If you use this model or visualization tools in your research, please cite:
```bibtex
@article{gritsenko2026infant,
  title={Validated Individualized Infant Digital Twin Framework for Dynamical Simulations},
  author={Gritsenko, Valeriya},
  journal={bioRxiv},
  year={2026},
  DOI={10.64898/2026.05.27.728208}
}
```
