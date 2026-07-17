# EgoVLA robot export contract

This is the minimum offline boundary between EgoVLA/H1 simulation demonstrations and
`EGO-HAND-WM`. The current implementation is an inspector and coordinate adapter, not a completed
MANO exporter. It reads source data without modifying it and fails rather than inventing missing
geometry.

## Required raw episode fields

An episode is an HDF5 file with root attribute `sim`, root dataset `action`, and an `observations`
group. All arrays must have the same leading length `T`.

| Path | Required shape | Meaning |
|---|---:|---|
| `action` | `[T, 50]` | H1/Inspire command |
| `observations/images/main` | `[T, 720, 1280, 3]` | Fixed main-camera RGB |
| `observations/qpos`, `qvel` | `[T, 50]` each | Robot proprioception |
| `observations/left_finger_tip_pos`, `right_finger_tip_pos` | `[T, 5, 3]` each | Environment-local fingertip positions |
| `observations/left_ee_pose`, `right_ee_pose` | `[T, 7]` each | Realized `[x,y,z,qw,qx,qy,qz]` EE poses |

The benchmark environment has already subtracted each parallel environment's origin from EE
translations and fingertips. The export must not subtract it again.

The upstream EgoVLA loader silently replaces absent actual EE arrays with
`left_target_ee_pose/right_target_ee_pose`. These are commanded targets, not realized state. The
inspector refuses this by default. `--allow-target-ee` is only an explicit diagnostic override and
the resulting provenance is recorded as `commanded_target`; it must not be mixed into an
actual-state benchmark unnoticed.

## Camera and coordinate convention

The fixed camera is 1280 x 720 with

```text
K = [[488.6662,   0.0, 640.0],
     [  0.0, 488.6662, 360.0],
     [  0.0,   0.0,   1.0]]
```

The model-facing normalized intrinsics are
`[0.38177046875, 0.67870305556, 0.5, 0.5]`. Simulator quaternions are WXYZ.

The adapter reproduces EgoVLA's manual calibration exactly. Let `T_E_C` be the camera pose in the
environment-local frame, assembled from translation `[0.09, 0, 1.7]`, raw camera WXYZ quaternion
`[0.66446, 0.24184, -0.24184, -0.664464]`, and the calibrated WXYZ orientation
`[0.9063077870366499, 0, 0.42261826174069944, 0]`. With

```text
CAM_AXIS_TRANSFORM = [[0,-1, 0,0],
                      [0, 0,-1,0],
                      [1, 0, 0,0],
                      [0, 0, 0,1]],
```

the conversion is `T_CV_E = CAM_AXIS_TRANSFORM @ inverse(T_E_C)`. Points use `p_CV =
T_CV_E @ homogenize(p_E)` and EE poses use `T_CV_EE = T_CV_E @ T_E_EE`. The tests gate both point
and pose round trips. Because the main camera is static, its canonical trajectory is identity at
every step.

## Time

Prefer a strictly increasing `timestamps_s` dataset at the root (or under `observations`). Raw
benchmark control advances at 30 Hz (`sim.dt * decimation = 1/30 s`), so only when explicit times
are absent may the adapter derive `arange(T) / 30`. The 15-FPS evaluation MP4 setting is playback
metadata and must not replace the 30-Hz state clock.

## MANO and supervision boundary

Full model-facing output requires, for each side, a fitted MANO global root and all 15 local MANO
joint rotations. EgoVLA fits neutral-shape MANO to the realized EE plus five observed robot
fingertips, optimizes six PCA coefficients, pads to PCA15, and uses side-specific root-axis
corrections. Converting PCA15 to this project's 90-D rot6D hand stream requires the matching MANO
v1.2 `hand_components` and `hand_mean` from both licensed files:

```text
MANO_LEFT.pkl
MANO_RIGHT.pkl
```

The retarget-network weights in `EgoVLA_Release` do not replace these assets or the offline fit.
The adapter checks that both model files are non-empty but never unpickles them. Until the fitting
backend and licensed files are available, no code may emit zero MANO vectors and mark them valid.

A later full exporter should retain two masks: geometry availability and training supervision.
Robot camera identity is valid history geometry, but future camera supervision must be false so
robot post-training cannot teach the dynamic human camera head to collapse to identity. Wrist and
MANO masks become true only for finite fits passing a residual threshold. Short future windows are
masked, never padded by repeating the last state.

## Read-only inspection

Install the optional HDF5 dependency and inspect one episode:

```bash
PYTHONPATH=src python scripts/inspect_egovla_robot_hdf5.py /path/to/episode.hdf5
```

To prove MANO readiness as well:

```bash
PYTHONPATH=src python scripts/inspect_egovla_robot_hdf5.py /path/to/episode.hdf5 \
  --mano-root /path/to/mano_v1_2
```

Without `--mano-root`, the JSON explicitly reports `mano_assets.ready: false`. With the option,
either both files pass or inspection stops. This script does not fit MANO, write shards, execute
retargeting, import Isaac Lab, or launch a job.
