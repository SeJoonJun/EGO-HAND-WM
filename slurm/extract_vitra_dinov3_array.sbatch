#!/usr/bin/env bash
#SBATCH --job-name=ehwm-dino
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
ROOT=/home/jun.se/EGO-HAND-WM
PYTHON=/projects/torresani-lab/sejoon/envs/ego-hand-wm/bin/python
SHARD_ROOT=/scratch/jun.se/EGO-HAND-WM/VITRA_SHARDS
OUTPUT_ROOT=/scratch/jun.se/EGO-HAND-WM/VITRA_DINO_FEATURES
PTS_ROOT=/scratch/jun.se/EGO-HAND-WM/VITRA_PTS
DINO_REPO=/home/jun.se/EGO-HAND-WM-REF/dinov3
DINO_WEIGHTS=/projects/torresani-lab/sejoon/checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8

mapfile -t SHARDS < <(find "$SHARD_ROOT" -maxdepth 1 -type f -name 'vitra-*.tar' | sort)
if (( ${#SHARDS[@]} != 598 )); then
  echo "Expected 598 annotation shards, found ${#SHARDS[@]}" >&2
  exit 2
fi

# Four persistent workers each load ViT-L once, build each video index once, and process every
# fourth shard. Completed shard manifests are validated and skipped when a worker is restarted.
ASSIGNED=()
for ((index=SLURM_ARRAY_TASK_ID; index<${#SHARDS[@]}; index+=SLURM_ARRAY_TASK_COUNT)); do
  ASSIGNED+=("${SHARDS[$index]}")
done

"$PYTHON" scripts/extract_vitra_dinov3.py extract \
  --annotation-shard "${ASSIGNED[@]}" \
  --output-root "$OUTPUT_ROOT" \
  --pts-root "$PTS_ROOT" \
  --repo-path "$DINO_REPO" \
  --weights-path "$DINO_WEIGHTS" \
  --model-name dinov3_vitl16 \
  --input-size 256 \
  --spatial-grid-size 4 \
  --batch-size 32 \
  --device cuda \
  --video-root ego4d_cooking_and_cleaning=/projects/torresani-lab/sejoon/datasets/VITRA/videos/ego4d_undistorted \
  --video-root ego4d_other=/projects/torresani-lab/sejoon/datasets/VITRA/videos/ego4d_undistorted \
  --video-root egoexo4d=/projects/torresani-lab/sejoon/datasets/VITRA/videos/egoexo4d_undistorted \
  --video-root epic=/projects/torresani-lab/datasets/epic-kitchens \
  --video-root ssv2=/projects/torresani-lab/datasets/something-something-v2
