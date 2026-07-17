#!/usr/bin/env bash
#SBATCH --job-name=ehwm-vitra-pts
#SBATCH --account=ydu_lab
#SBATCH --partition=shared
#SBATCH --array=0-31%16
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=/n/home08/sjmathy/EGO-HAND-WM/logs/%x-%A_%a.out
#SBATCH --error=/n/home08/sjmathy/EGO-HAND-WM/logs/%x-%A_%a.err

set -euo pipefail

# Required at submission time:
#   DATASET_NAME: exact VITRA namespace, e.g. ssv2, epic, egoexo4d,
#                 ego4d_cooking_and_cleaning, or ego4d_other
#   VIDEO_ROOT:   root recursively containing that source's processed videos
#
# Examples (create the logs directory before sbatch):
#   sbatch --export=ALL,DATASET_NAME=ssv2,VIDEO_ROOT=/n/netscratch/ydu_lab/Lab/sjmathy/VITRA_RAW/ssv2 \
#     slurm/build_vitra_pts_array.sbatch
#   sbatch --export=ALL,DATASET_NAME=epic,VIDEO_ROOT=/n/netscratch/ydu_lab/Lab/sjmathy/VITRA_RAW/epic \
#     slurm/build_vitra_pts_array.sbatch

: "${DATASET_NAME:?Set DATASET_NAME to the exact VITRA dataset namespace}"
: "${VIDEO_ROOT:?Set VIDEO_ROOT to the processed video root}"
: "${SLURM_ARRAY_TASK_ID:?This launcher must run as a Slurm array}"
: "${SLURM_ARRAY_TASK_COUNT:?SLURM_ARRAY_TASK_COUNT is unavailable}"

ROOT=/n/home08/sjmathy/EGO-HAND-WM
OUTPUT_ROOT=${OUTPUT_ROOT:-/n/netscratch/ydu_lab/Lab/sjmathy/VITRA_PTS}
NUM_SHARDS=${NUM_SHARDS:-$SLURM_ARRAY_TASK_COUNT}
CONDA_ENV=${CONDA_ENV:-base}

cd "$ROOT"
source /n/home08/sjmathy/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}

python scripts/build_video_pts_index.py \
  --video-root "$VIDEO_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --dataset-name "$DATASET_NAME" \
  --shard-id "$SLURM_ARRAY_TASK_ID" \
  --num-shards "$NUM_SHARDS"
