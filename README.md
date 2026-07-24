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
cd /home/jun.se/EGO-HAND-WM
PYTHONPATH=src python -m ego_hand_wm.cli.train --config configs/smoke.yaml
PYTHONPATH=src pytest -q
```

Production configurations are intentionally local-files-only for DINOv3/DINO.txt weights. A
missing weight cache fails clearly rather than initiating an unreviewed download.

`configs/vitra_pretrain.yaml` is the geometry-only ablation recipe. The finalized production
architecture is `configs/vitra_pretrain_visual_aux.yaml`: it requires globally deduplicated
precomputed DINO.txt features and prompt embeddings and enables the parallel, training-only
17-token future-DINO expert. Both configurations have hard modality guards. The current
real-data runnable path is `configs/vitra_geometry_gate.yaml`.

## Data boundary

All datasets must emit `CanonicalBatch`. The 207-D state order is fixed:

```text
camera9 | left_wrist9 | right_wrist9 | left_MANO(15x6) | right_MANO(15x6)
```

Every pose is anchored to the final observed camera `A=C0`. Dataset-specific coordinates, MANO
conventions, and robot qpos remain outside the shared model.

VITRA production uses both hands. Its released left local rotations remain in the documented
MANO_RIGHT-derived basis (`left_mano_policy: as_stored`) and are represented by left-specific
tokens. Native MANO_LEFT or robot-hand conversion is deferred to the deterministic output adapter.

See [plan.md](plan.md) for the frozen research plan and `configs/` for executable settings.

## Explorer VITRA layout

The checked-in configs and Slurm templates use the Explorer paths below:

```text
/scratch/jun.se/VITRA-1M                                      # released annotations/archives
/scratch/jun.se/EGO-HAND-WM/VITRA_SHARDS                     # derived annotation shards
/scratch/jun.se/EGO-HAND-WM/VITRA_PTS                        # derived video PTS caches
/projects/torresani-lab/sejoon/datasets/VITRA/videos/
  ego4d_undistorted                                           # 2,448 videos
  egoexo4d_undistorted                                        # 4,990 videos
/projects/torresani-lab/datasets/epic-kitchens                # EPIC videos
/projects/torresani-lab/datasets/something-something-v2       # SSv2 videos
```

The Ego4D/EgoExo4D files are already undistorted. Do not run VITRA's undistortion scripts again.
Their annotation intrinsics remain native-resolution intrinsics; the adapter normalizes against
the calibration canvas rather than dividing raw `K` by the downscaled video dimensions.

## What is implemented now

- Fixed 207-D canonical geometry and variable-time batch contracts.
- VITRA world-to-anchor-camera conversion with physical timestamps and supervision masks.
- Thirteen anatomical state entities per time: camera, bilateral wrists, and five three-joint
  MANO chains per hand. The external 207-D storage contract remains unchanged.
- Structured conditional rectified-flow transformer with separate physical-time and flow-time
  embeddings and a single masked softmax over cached context plus future-entity K/V.
- Shared per-layer observed-context K/V cache consumed by independent geometry and future-DINO
  experts. The visual expert is training-only and cannot feed future information into geometry.
- Camera, bilateral wrist, and full 15-joint MANO output projections.
- Masked per-stream flow, SO(3), and camera/hand decomposition losses.
- Native PyTorch DDP, atomic checkpoints, resume, CPU smoke tests, and guarded Slurm templates.
- Byte-preserving VITRA annotation sharding and actual-video-PTS indexing scripts.

Still gated before a production GPU run: aligned DINO feature shards, canonical train statistics,
licensed MANO model files for left-hand/FK validation, and the offline EgoVLA robot export adapter.
No preprocessing array or GPU job is launched by this repository setup.
