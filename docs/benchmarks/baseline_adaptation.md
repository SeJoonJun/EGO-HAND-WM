# Adapted trajectory baseline map

All comparisons use the common H6-K16 manifest and evaluate sixteen 3D hand
positions in the last observed camera frame. Results must be labelled
"adapted" because none of the native paper protocols are identical.

## USST

Repository: `/home/jun.se/EGO-HAND-WM-REF/USST`

USST consumes RGB and observed trajectory locations, masks future locations,
and predicts a full sequence with its TransformerSSM. The released H2O setting
uses 64 total frames at a 0.6 observation ratio; EgoPAT3D uses 40 at 0.6.

Implemented adaptation:

- `DATA.max_frames=22`, `history_steps=6`, ratio `6/22`.
- Exact windows with stride 16 and an end-aligned tail window.
- Clips shorter than 22 are dropped rather than padded.
- H2O world positions and EgoPAT3D per-frame positions are transformed into
  the sixth camera frame.
- Native configs/loaders remain the default unless the adapted flags are set.

Configs:

- `config/adapted/h2o_h6_k16_res18_3d.yml`
- `config/adapted/egopat3d_h6_k16_res18_3d.yml`

Launch with `python train.py --config <config> --tag h6_k16` after setting the
config data root. Check its generated sample counts against the common JSONL
manifests before training.

## MMTwin

Repository: `/home/jun.se/EGO-HAND-WM-REF/MMTwin`

MMTwin combines trajectory, GLIP, motion, and voxel inputs with separate hand
trajectory and egomotion diffusion components. The release only implements
EgoPAT3D, not H2O, and defaults to 40 frames with a 0.6 ratio.

Implemented adaptation:

- New CLI flags select 22 total, six observed, ratio `6/22`, stride 16, strict
  windows, tail coverage, and the sixth-camera coordinate frame.
- RGB, GLIP features, motion features, and odometry are sliced with the same
  window offset.
- Voxel cache names include the window offset so incompatible full-clip
  caches cannot be silently reused.

Launch with:

```bash
cd /home/jun.se/EGO-HAND-WM-REF/MMTwin
EGOPAT3D_ROOT=/path/to/EgoPAT3D bash train_egopat3d_h6k16.sh
```

The adapted run must regenerate its voxel caches. An H2O result would require
a new H2O multimodal preprocessing path and should not be claimed from the
current release.

## HandsOnVLM

Repository: `/home/jun.se/EGO-HAND-WM-REF/HandsOnVLM-release`

HandsOnVLM is a VLM trajectory-token baseline. Its native data/collator/model/
trainer path is hardcoded for ten observed images and four future 2D points
(five points in the dataset because it includes the last observation). It
cannot become a 16-step 3D baseline through a config-only change.

Implemented foundation:

- `AdaptedTrajectoryHead3D` accepts `[B,2,16,D]` trajectory-token embeddings,
  predicts `[B,2,16,3]`, and supports per-hand/per-time masks.
- `configs/adapted/h2o_egopat3d_h6_k16.yaml` records the target protocol while
  preserving native checkpoint shapes.

Required integration before its adapted result is publishable:

1. Add an H2O/EgoPAT3D wrapper around the common manifest reader.
2. Replace the EPIC-only ten-frame/100-image collator assertions with six
   observed images and make visual-token length dynamic.
3. Emit sixteen `<hand_traj>` tokens instead of four.
4. Accept anchor-plus-sixteen 3D points from the dataset, remove the anchor,
   and route the sixteen targets to `AdaptedTrajectoryHead3D`.
5. Replace the 2D trajectory positional projection with a 3D projection.
6. Generalize the trainer's fixed `[B,2,4,2]` assertions and report 3D ADE/FDE.
7. Retrain the new head and the affected token/projection layers; a native
   HandsOnVLM checkpoint cannot directly supply these new parameters.

Until those seven items pass an overfit test, HandsOnVLM should be listed as
"adaptation in progress", not as a completed baseline.
