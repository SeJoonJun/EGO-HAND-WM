#!/usr/bin/env bash
#SBATCH --job-name=ehwm-dino-smoke
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
ROOT=/home/jun.se/EGO-HAND-WM
PYTHON=/projects/torresani-lab/sejoon/envs/ego-hand-wm/bin/python

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8

"$PYTHON" scripts/extract_vitra_dinov3.py extract \
  --annotation-shard "$ROOT/.smoke/vitra_episode/vitra-smoke.tar" \
  --output-root "$ROOT/.smoke/features" \
  --pts-root /scratch/jun.se/EGO-HAND-WM/VITRA_PTS \
  --repo-path /home/jun.se/EGO-HAND-WM-REF/dinov3 \
  --weights-path /projects/torresani-lab/sejoon/checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --model-name dinov3_vitl16 \
  --input-size 256 \
  --spatial-grid-size 4 \
  --batch-size 24 \
  --device cuda \
  --force \
  --video-root ego4d_cooking_and_cleaning=/projects/torresani-lab/sejoon/datasets/VITRA/videos/ego4d_undistorted \
  --video-root ego4d_other=/projects/torresani-lab/sejoon/datasets/VITRA/videos/ego4d_undistorted \
  --video-root egoexo4d=/projects/torresani-lab/sejoon/datasets/VITRA/videos/egoexo4d_undistorted \
  --video-root epic=/projects/torresani-lab/datasets/epic-kitchens \
  --video-root ssv2=/projects/torresani-lab/datasets/something-something-v2

"$PYTHON" -c "from pathlib import Path; from ego_hand_wm.data.feature_shards import decode_feature_record; import tarfile; p=Path('$ROOT/.smoke/features/shards/vitra-smoke.tar'); a=tarfile.open(p); m=next(x for x in a if x.isfile()); r=decode_feature_record(a.extractfile(m).read()); print({'member': m.name, 'frames': len(r.frame_ids), 'shape': r.pooled_features.shape, 'dtype': str(r.pooled_features.dtype), 'finite': bool(__import__('numpy').isfinite(r.pooled_features).all())})"
