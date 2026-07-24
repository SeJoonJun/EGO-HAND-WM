# H2O / EgoPAT3D adapted-baseline contract

These experiments compare model families under one data and metric contract. They
are **adapted baselines**, not unchanged reproductions of the upstream papers.

## Shared task

- Input: six RGB frames and six tracked-wrist positions sampled at 30 Hz.
- Anchor: time zero and all 3-D coordinates are expressed in the last observed
  camera frame.
- Output: 16 future tracked-wrist positions at 30 Hz. The final target is
  0.533 seconds after the anchor.
- Unit: metres for 3-D outputs.
- Splits: official H2O train/val/test and source-native EgoPAT3D
  train/val/test-seen/test-novel manifests.
- Metrics: 3-D ADE/FDE in metres and image-normalized 2-D ADE/FDE. Stochastic
  models additionally report the sampling count and best-of-N selection rule.
- One manifest row represents one tracked hand. A model must not silently change
  this to a bi-hand example.

The canonical manifests are in `data/h6_k16_manifests`. Run the cross-repository
geometry check with:

```bash
cd /home/jun.se/EGO-HAND-WM
PYTHONPATH=src python scripts/audit_h6k16_baseline_geometry.py
```

## Model adaptations

### USST

The upstream USST trajectory network and loss are retained. Its dataset loaders,
temporal lengths, camera anchoring, and reporting are replaced by the shared
H6/K16 contract. Training and test entry points emit paired 3-D and normalized
2-D metrics.

```bash
cd /home/jun.se/EGO-HAND-WM-REF/USST
bash train_h2o_h6k16.sh
bash train_egopat3d_h6k16.sh
```

### MMTwin

The twin diffusion/Mamba-Transformer predictor is retained. Six observed DINOv3
states and observed wrist positions condition diffusion over 16 future targets.
No future RGB is supplied. Metric XYZ is converted to the model's projective
space for training and deterministically converted back for evaluation.

```bash
cd /home/jun.se/EGO-HAND-WM-REF/MMTwin
bash train_h2o_h6k16.sh
bash train_egopat3d_h6k16.sh
```

Formal evaluation uses `traineval_h6k16.py --evaluate-only --resume CHECKPOINT`
with the desired test manifest and `--num-samples N`. Evaluation must run on one
GPU because a distributed sampler would pad the split with duplicate examples.

### HandsOnVLM

The upstream VLM and CVAE design are retained. The adapted head receives an
explicit time-aware encoding of all six observed wrist positions and decodes 16
future metric-XYZ trajectory tokens. Adapted inference samples the CVAE without
receiving future targets.

```bash
cd /home/jun.se/EGO-HAND-WM-REF/HandsOnVLM-release
MODEL_PATH=/projects/torresani-lab/sejoon/checkpoints/handsonvlm/handsonvlm-7b \
MANIFEST=/home/jun.se/EGO-HAND-WM/data/h6_k16_manifests/h2o_train_h6_k16.jsonl \
OUTPUT_DIR=/projects/torresani-lab/sejoon/runs/baselines/HandsOnVLM/h2o-h6k16 \
bash scripts/finetune_h6k16.sh
```

Evaluate a completed adapted checkpoint with `scripts/evaluate_h6k16.sh`, setting
`CHECKPOINT`, `MANIFEST`, `OUTPUT`, and optionally `NUM_SAMPLES`.

## Validation status

- Canonical trajectory tests: passed.
- Cross-adapter geometry audit: zero numerical discrepancy on sampled rows from
  all seven manifests.
- USST: bounded real-GPU train/validation probes passed for both datasets.
- MMTwin: bounded real-GPU train/validation/sampling probes passed for both
  datasets.
- HandsOnVLM: unit-level history-gradient and sampling tests passed; corrected
  full 7.06B-model optimization probes passed for both datasets.

Probe losses and metrics are integration diagnostics only. Final paper numbers
require complete training and evaluation over the untouched test manifests.
