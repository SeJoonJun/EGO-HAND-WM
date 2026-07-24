#!/usr/bin/env bash
#SBATCH --job-name=ehwm-rgb-stage
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-63%16
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
ROOT=/home/jun.se/EGO-HAND-WM
PYTHON=/projects/torresani-lab/sejoon/envs/ego-hand-wm/bin/python
REQUEST_ROOT=/scratch/jun.se/EGO-HAND-WM/VITRA_FRAME_REQUESTS
OUTPUT_ROOT=/scratch/jun.se/EGO-HAND-WM/VITRA_RGB_256
NUM_WORKERS=${NUM_WORKERS:-$SLURM_ARRAY_TASK_COUNT}

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export EGO_HAND_WM_DECODE_THREADS=4

"$PYTHON" scripts/stage_vitra_rgb.py \
  --request-database "$REQUEST_ROOT/requests.sqlite" \
  --output-root "$OUTPUT_ROOT" \
  --pts-root /scratch/jun.se/EGO-HAND-WM/VITRA_PTS \
  --worker-id "$SLURM_ARRAY_TASK_ID" \
  --num-workers "$NUM_WORKERS" \
  --video-root ego4d_cooking_and_cleaning=/projects/torresani-lab/sejoon/datasets/VITRA/videos/ego4d_undistorted \
  --video-root egoexo4d=/projects/torresani-lab/sejoon/datasets/VITRA/videos/egoexo4d_undistorted \
  --video-root epic=/projects/torresani-lab/datasets/epic-kitchens/required_videos_flat \
  --video-root ssv2=/projects/torresani-lab/datasets/something-something-v2
