# EGO-HAND-WM

Geometry-first, physically time-conditioned world-action model for joint future camera SE(3),
bilateral wrist SE(3), and full MANO articulation forecasting.

The sibling directories (`VITRA`, `EgoVLA_Release`, `MMTwin`, and others) are upstream reference
repositories. The new implementation lives exclusively in `src/ego_hand_wm`; it does not modify
or import those repositories into the main training process.

## Current runnable gate

The CPU smoke configuration uses synthetic canonical data and lightweight encoders. It exercises
the complete batch contract, structured flow denoiser, masked flow loss, optimizer, sampler, and
checkpoint path without downloading weights or requesting a GPU.

```bash
cd /n/home08/sjmathy/EGO-HAND-WM
PYTHONPATH=src python -m ego_hand_wm.cli.train --config configs/smoke.yaml
PYTHONPATH=src pytest -q
```

Production configurations are intentionally local-files-only for DINOv3 and text weights. A
missing weight cache fails clearly rather than initiating an unreviewed download.

`configs/vitra_pretrain.yaml` is the geometry-first target recipe, not a launch-ready claim. It
requires aligned precomputed context DINO features but keeps the future-visual loss off. Only
after that stage passes should `configs/vitra_pretrain_visual_aux.yaml` enable future-DINO targets
and their one-way auxiliary loss. Both have hard modality guards. The current real-data runnable
path is `configs/vitra_geometry_gate.yaml`.

## Data boundary

All datasets must emit `CanonicalBatch`. The 207-D state order is fixed:

```text
camera9 | left_wrist9 | right_wrist9 | left_MANO(15x6) | right_MANO(15x6)
```

Every pose is anchored to the final observed camera `A=C0`. Dataset-specific coordinates, MANO
conventions, and robot qpos remain outside the shared model.

See [plan.md](plan.md) for the frozen research plan and `configs/` for executable settings.

## What is implemented now

- Fixed 207-D canonical geometry and variable-time batch contracts.
- VITRA world-to-anchor-camera conversion with physical timestamps and supervision masks.
- Structured conditional rectified-flow transformer with separate physical-time and flow-time
  embeddings.
- Camera, bilateral wrist, and full 15-joint MANO heads, plus an optional one-way future-visual
  latent auxiliary head.
- Masked per-stream flow, SO(3), and camera/hand decomposition losses.
- Native PyTorch DDP, atomic checkpoints, resume, CPU smoke tests, and guarded Slurm templates.
- Byte-preserving VITRA annotation sharding and actual-video-PTS indexing scripts.

Still gated before a production GPU run: aligned DINO feature shards, canonical train statistics,
licensed MANO model files for left-hand/FK validation, and the offline EgoVLA robot export adapter.
No preprocessing array or GPU job is launched by this repository setup.
