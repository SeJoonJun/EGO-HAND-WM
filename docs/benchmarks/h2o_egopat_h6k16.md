# H2O / EgoPAT3D adapted H6-K16 protocol

This protocol is an adapted comparison, not a reproduction of each baseline's
native timing.

- Input: six consecutive RGB/trajectory frames at 30 Hz.
- Anchor: the sixth (last observed) camera frame, at time zero.
- Output: sixteen consecutive 3D hand positions at +1/30 through +16/30 s.
- History span: 5/30 s = 0.167 s.
- Final prediction horizon: 16/30 s = 0.533 s.
- Window length: 22 frames. Clips shorter than 22 are dropped, never padded.
- Train window stride: 16 frames, with one end-aligned tail window.
- Splits: preserve the official H2O split and USST's physical-record
  EgoPAT3D train/val/test-seen/test-novel split.
- Metrics: 3D ADE and 3D FDE in metres, reported separately per dataset and
  for EgoPAT3D seen/novel subsets.

The generated JSONL manifests are the source of truth. A loader may consume
them directly or implement the identical deterministic window rule, but its
split/window counts must match the manifests before training. Each record
contains the exact source video, trajectory file, physical split group, six
history indices, sixteen future indices, and relative timestamps. This avoids
episode leakage and prevents model-specific window selection.

Generate the manifests after the raw datasets are present:

```bash
PYTHONPATH=src python scripts/build_h2o_egopat_h6k16_manifests.py \
  --dataset both \
  --h2o-root /path/to/H2O \
  --egopat3d-root /path/to/EgoPAT3D \
  --output-dir /path/to/H6_K16_MANIFESTS \
  --require-video
```

The summary JSON reports exact retained windows, short clips, and missing
files. The script references existing video files; it does not duplicate RGB.

`TrajectoryWindowDataset` in `ego_hand_wm.benchmarks` is the shared model-side
reader. It converts H2O world coordinates and EgoPAT3D incremental odometry to
the same last-observed-camera frame, returns tensors shaped `[6,3]` and
`[16,3]`, and optionally decodes the six RGB context frames.

## Baseline status

- USST: adapted H2O and EgoPAT3D loaders/configs/launchers are implemented.
  Their split counts exactly match the manifests (8,604/1,830/3,832 for H2O;
  14,280/1,735/3,353/4,686 for EgoPAT3D), and checked metric trajectories
  match this reader to numerical precision. Train from scratch.
- MMTwin: an EgoMAN-style adapter reads both datasets directly, replaces the
  unavailable GLIP/voxel front end with frozen DINOv3 context, and retains the
  twin camera/hand diffusion plus Mamba--Transformer core. Forward, backward,
  and 16-step sampling smoke tests pass. Train from scratch.
- HandsOnVLM: a common-manifest conversation dataset and six-frame visual path
  are integrated. Sixteen special trajectory tokens condition a new 3-D CVAE
  head, following EgoMAN's adapted-baseline principle. Fine-tune the released
  HandsOnVLM checkpoint; the new CVAE head starts from scratch.

Native configs and native checkpoint tensor shapes remain unchanged.
