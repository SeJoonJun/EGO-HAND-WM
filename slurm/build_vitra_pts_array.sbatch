#!/usr/bin/env bash
#SBATCH --job-name=ehwm-vitra-pts
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --array=0-31%16
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/jun.se/EGO-HAND-WM/logs/%x-%A_%a.out
#SBATCH --error=/home/jun.se/EGO-HAND-WM/logs/%x-%A_%a.err

set -euo pipefail

# Required at submission time:
#   DATASET_NAME: exact VITRA namespace, e.g. ssv2, epic, egoexo4d,
#                 ego4d_cooking_and_cleaning, or ego4d_other
#   VIDEO_ROOT:   root recursively containing that source's processed videos
#
# Examples (create the logs directory before sbatch):
#   sbatch --export=ALL,DATASET_NAME=ssv2,VIDEO_ROOT=/projects/torresani-lab/datasets/something-something-v2 \
#     slurm/build_vitra_pts_array.sbatch
#   sbatch --export=ALL,DATASET_NAME=epic,VIDEO_ROOT=/projects/torresani-lab/datasets/epic-kitchens \
#     slurm/build_vitra_pts_array.sbatch
#   sbatch --export=ALL,DATASET_NAME=ego4d_cooking_and_cleaning,VIDEO_ROOT=/projects/torresani-lab/sejoon/datasets/VITRA/videos/ego4d_undistorted \
#     slurm/build_vitra_pts_array.sbatch
#   sbatch --export=ALL,DATASET_NAME=egoexo4d,VIDEO_ROOT=/projects/torresani-lab/sejoon/datasets/VITRA/videos/egoexo4d_undistorted \
#     slurm/build_vitra_pts_array.sbatch

: "${DATASET_NAME:?Set DATASET_NAME to the exact VITRA dataset namespace}"
: "${VIDEO_ROOT:?Set VIDEO_ROOT to the processed video root}"
: "${SLURM_ARRAY_TASK_ID:?This launcher must run as a Slurm array}"
: "${SLURM_ARRAY_TASK_COUNT:?SLURM_ARRAY_TASK_COUNT is unavailable}"

ROOT=/home/jun.se/EGO-HAND-WM
OUTPUT_ROOT=${OUTPUT_ROOT:-/scratch/jun.se/EGO-HAND-WM/VITRA_PTS}
NUM_SHARDS=${NUM_SHARDS:-$SLURM_ARRAY_TASK_COUNT}
PYTHON_BIN=${PYTHON_BIN:-/projects/torresani-lab/sejoon/miniconda3/bin/python}
PREPROCESS_SITE=${PREPROCESS_SITE:-/home/jun.se/.local/ego-hand-wm-preprocess}

cd "$ROOT"
export PYTHONPATH="$PREPROCESS_SITE:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}

"$PYTHON_BIN" scripts/build_video_pts_index.py \
  --video-root "$VIDEO_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --dataset-name "$DATASET_NAME" \
  --shard-id "$SLURM_ARRAY_TASK_ID" \
  --num-shards "$NUM_SHARDS"
