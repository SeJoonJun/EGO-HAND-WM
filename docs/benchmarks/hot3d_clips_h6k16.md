# Controlled HOT3D-Clips Aria hand forecasting

This benchmark uses the official packaged HOT3D-Clips Aria data, not the
custom `LafouCC/hot3d-full` derivative. Each official clip contains 150 frames
(five seconds), Aria RGB images, metric camera poses, and public hand labels
for the released training package.

The official test archives cannot be used for local hand-forecasting metrics:
their hand annotations are withheld. We therefore derive and explicitly name
a **controlled public-train benchmark** rather than claiming official test-set
results.

## Fixed participant split

- train: P0001, P0002, P0003, P0009, P0010, P0014, P0015
- validation: P0011
- test: P0012

This is approximately a 70/15/15 division by official labeled Aria clip count.
No participant, sequence, or clip crosses splits. The builder scans every tar
and retains only hand windows with all 22 required labels.

The completed strict build contains:

- train: 1,059 clips, 84 sequences, 19,062 hand windows;
- validation: 220 clips, 19 sequences, 3,960 hand windows;
- test: 237 clips, 21 sequences, 4,266 hand windows.

All 27,288 windows have complete 22-frame tracks. The local audit reports zero
participant, sequence, clip, or sample overlap across the three splits.

## Temporal and geometric contract

- six observed frames and sixteen strictly future frames at 30 Hz;
- windows start every 16 frames, including the end-aligned tail;
- time zero is the sixth/last observed frame;
- final prediction horizon is 16/30 = 0.533 seconds;
- camera and wrist poses are expressed in the last observed Aria RGB camera;
- RGB stream `214-1` is rotated 90 degrees clockwise before visual encoding;
- training and ADE/FDE use only the selected wrist translation in metres;
- wrist rotation is retained in the canonical state but masked by default.

HOT3D provides a directly usable metric UmeTrack `T_world_from_wrist`. The
packaged MANO `thetas` contain 15 hand-model parameters, not fifteen raw
axis-angle vectors. The adapter masks the canonical MANO streams rather than
inventing an invalid conversion. A later articulated-hand arm must decode the
official model with Meta's hand-tracking toolkit and its required assets.

## Data preparation

The selective downloader fetches the two official manifests and only
`train_aria/*.tar`:

```bash
python scripts/download_hot3d_clips_aria.py \
  --local-dir /projects/torresani-lab/sejoon/datasets/HOT3D-Clips \
  --token-file /projects/torresani-lab/sejoon/.cache/huggingface/token \
  --max-workers 8
```

After the download completes, build strict manifests:

```bash
python scripts/build_hot3d_clips_h6k16_manifests.py \
  --root /projects/torresani-lab/sejoon/datasets/HOT3D-Clips \
  --output-dir data/hot3d_clips_h6_k16 \
  --scan-workers 16
```

Re-run the leakage and real-sample contract audit with:

```bash
python scripts/audit_hot3d_clips_manifests.py \
  --manifest-dir data/hot3d_clips_h6_k16
```

For model training, set `data.kind: hot3d_clips_h6k16`, provide the three
manifest paths, and use `decode_rgb: true`. The existing `trajectory`
validation path then reports deterministic N=1 wrist ADE/FDE and camera error.

## Compatibility status

The task-level contract now matches the adapted H2O/EgoPAT3D comparison:
`H=6`, `K=16`, 30 Hz, one tracked hand per row, last-observed-camera metric
XYZ, and deterministic N=1 ADE/FDE. A real public HOT3D tar passes canonical
collation and a model forward/loss/backward smoke test.

All four model paths now read the same HOT3D tar/manifests:

- EGO-HAND-WM uses `Hot3DClipsForecastDataset`;
- MMTwin uses `data_utils/H6K16ManifestLoader.py`;
- HandsOnVLM uses `handsonvlm/dataset/h6k16_dataset.py`;
- USST uses `src/H6K16ManifestLoader.py` and the
  `config/adapted/hot3d_clips_h6_k16_res18_3d_absolute.yml` configuration.

Every adapted path uses the train-split statistics:

```text
mean = [ 0.27043531, -0.06219469, 0.31847892 ] metres
std  = [ 0.11184065,  0.18661360, 0.10673272 ] metres
```

Standardization is reversible and is not a residual output parameterization.
The supervised target remains the absolute wrist XYZ in the last-observed
camera frame. In particular, no model receives a nonlearned
`last_observed_wrist + predicted_delta` shortcut.

The cross-model audit decodes real samples from every split and compares the
metric target before model-specific normalization:

```bash
PYTHONPATH=src \
  /projects/torresani-lab/sejoon/miniconda3/envs/handsonvlm-h6k16/bin/python \
  scripts/audit_hot3d_baseline_geometry.py \
  --output status/hot3d_baseline_compatibility.json
```

MMTwin and HandsOnVLM match the canonical target exactly. USST differs only by
float32 standardize/unstandardize roundoff (at most `3e-8` metres in the
audit).

Aria uses a fisheye camera, so the USST adapter intentionally reports only
metric 3-D ADE/FDE and does not reuse its pinhole 2-D projection metrics.

## Visual cache

EGO-HAND-WM's VITRA-initialized configuration uses precomputed DINO.txt
features. It needs:

- history features for train, validation, and test;
- future visual targets for train and validation only.

The five cache gates are produced by:

```bash
sbatch slurm/extract_hot3d_clips_dinotxt_array.sbatch
```

The features live under
`/scratch/jun.se/EGO-HAND-WM/TRAJECTORY_DINOTXT/hot3d_clips_aria`. Each cache
is written through a temporary memmap and becomes usable only after its
matching `*.SUCCESS.json` gate validates shape, dtype, feature extractor, and
sample count.

After all five gates exist, run the strict cache/manifests audit:

```bash
PYTHONPATH=src \
  /projects/torresani-lab/sejoon/envs/ego-hand-wm/bin/python \
  scripts/audit_hot3d_feature_cache.py \
  --output status/hot3d_feature_cache_audit.json
```

The EGO-HAND-WM fine-tune is prepared but is not started as part of data
preparation:

```bash
sbatch slurm/train_hot3d_clips_vitra_h6k16.sbatch
```

The corresponding configuration is
`configs/benchmarks/hot3d_clips_vitra_finetune_h6k16.yaml`.

## Adapted baseline execution

All launchers use direct absolute XYZ output. The full runs are intentionally
sequential, and none requests more than two GPUs.

USST (one GPU):

```bash
cd /home/jun.se/EGO-HAND-WM-REF/USST
sbatch slurm/smoke_hot3d_absolute.sbatch
sbatch slurm/train_hot3d_absolute.sbatch
sbatch slurm/eval_hot3d_absolute.sbatch
```

MMTwin (one-GPU smoke/evaluation, two-GPU training):

```bash
cd /home/jun.se/EGO-HAND-WM-REF/MMTwin
sbatch slurm/smoke_hot3d_absolute.sbatch
sbatch slurm/train_hot3d_absolute.sbatch
sbatch slurm/eval_hot3d_absolute.sbatch
```

HandsOnVLM (one-GPU smoke/evaluation, two-GPU training):

```bash
cd /home/jun.se/EGO-HAND-WM-REF/HandsOnVLM-release
sbatch slurm/smoke_hot3d_absolute.sbatch
sbatch slurm/train_hot3d_absolute.sbatch
sbatch slurm/eval_hot3d_absolute.sbatch
```

Each stage advances only after the preceding smoke/training/evaluation process
has exited successfully. HOT3D evaluation reports metric 3-D ADE/FDE only;
projective 2-D metrics are intentionally omitted because the Aria stream is
fisheye and these baseline evaluators otherwise assume a pinhole camera.
