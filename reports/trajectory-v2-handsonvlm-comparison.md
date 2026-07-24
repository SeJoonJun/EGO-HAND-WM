# EGO-HAND-WM trajectory-v2 versus adapted HandsOnVLM

## Protocol

This is an adapted-baseline comparison, not a comparison to either method's native
paper table.

- Common manifests and targets: H6/K16 at 30 Hz.
- Observation: 6 frames (0.167 s from first to anchor).
- Forecast: 16 strictly future frames (0.533 s after the anchor).
- Coordinates: 3D hand/wrist position in the last-observed-camera frame.
- Metrics: trajectory ADE and final-step FDE in centimeters.
- Sampling: one prediction per context (`N=1`).
- Persistence residual: disabled.
- Test examples: H2O 3,832; EgoPAT3D seen 3,353; EgoPAT3D unseen 4,686.

EGO-HAND-WM checkpoint and inference selection used the complete validation split
only: H2O 1,830 examples and EgoPAT3D 1,735 examples. For each dataset, two
training recipes, ODE steps in `{8,16}`, and initial noise scales in
`{0.0,0.5,1.0}` were ranked by the predeclared mean of validation ADE and FDE.
After this selection was fixed, the selected configuration was evaluated once on
each held-out test split.

The selected zero initial-noise scale is a validation-selected deterministic flow
initialization. It is not a last-position/persistence residual: the model still
generates the complete anchored future trajectory through its learned flow field,
and `persistence_residual` is `false` in both result artifacts.

## Results

| Dataset/split | Model | ADE (cm) ↓ | FDE (cm) ↓ |
|---|---:|---:|---:|
| H2O | HandsOnVLM | 2.1620 | 3.5099 |
| H2O | **EGO-HAND-WM trajectory-v2** | **1.9173** | **3.4777** |
| EgoPAT3D seen | HandsOnVLM | 26.5224 | 44.8328 |
| EgoPAT3D seen | **EGO-HAND-WM trajectory-v2** | **22.6588** | **41.4541** |
| EgoPAT3D unseen | HandsOnVLM | 15.6744 | 23.2238 |
| EgoPAT3D unseen | **EGO-HAND-WM trajectory-v2** | **12.6004** | **21.4663** |

Absolute EGO-HAND-WM improvements over HandsOnVLM:

- H2O: 0.2447 cm ADE and 0.0323 cm FDE.
- EgoPAT3D seen: 3.8636 cm ADE and 3.3787 cm FDE.
- EgoPAT3D unseen: 3.0740 cm ADE and 1.7574 cm FDE.

## Selected EGO-HAND-WM configurations

- H2O: trajectory-v2 refine checkpoint, 8 Heun ODE steps, initial noise scale
  0.0. Full-validation ADE/FDE: 1.9328/3.5937 cm.
- EgoPAT3D: trajectory-v2 fresh checkpoint, 16 Heun ODE steps, initial noise
  scale 0.0. Full-validation ADE/FDE: 13.0741/21.4807 cm.

## Authoritative artifacts

EGO-HAND-WM:

- `reports/h2o-vitra-h6k16-trajectory-v2-selected-test.json`
- `reports/egopat3d-vitra-h6k16-trajectory-v2-selected-test.json`
- H2O checkpoint:
  `/scratch/jun.se/EGO-HAND-WM/runs/h2o-vitra-h6k16-trajectory-v2-refine/best.pt`
- EgoPAT3D checkpoint:
  `/scratch/jun.se/EGO-HAND-WM/runs/egopat3d-vitra-h6k16-trajectory-v2-iofix/best.pt`

HandsOnVLM:

- `/projects/torresani-lab/sejoon/runs/baselines/HandsOnVLM/paper-h6k16-xyz-seed0-h2o/evaluation/test_n1_seed0.npz`
- `/projects/torresani-lab/sejoon/runs/baselines/HandsOnVLM/paper-h6k16-xyz-seed0-egopat3d/evaluation/seen_n1_seed0.npz`
- `/projects/torresani-lab/sejoon/runs/baselines/HandsOnVLM/paper-h6k16-xyz-seed0-egopat3d/evaluation/unseen_n1_seed0.npz`

Slurm:

- Corrected training array: `8626160` (both tasks completed).
- Dependency-chained selection/test array: `8626323` (both tasks completed).

Verification after the final loader and loss changes: 93 tests passed, one
optional real-VITRA fixture test skipped.
