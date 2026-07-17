#!/usr/bin/env bash
#SBATCH --job-name=ehwm-vitra-shards
#SBATCH --account=ydu_lab
#SBATCH --partition=shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=2-00:00:00
#SBATCH --output=/n/home08/sjmathy/EGO-HAND-WM/logs/%x-%j.out
#SBATCH --error=/n/home08/sjmathy/EGO-HAND-WM/logs/%x-%j.err

set -euo pipefail

ROOT=/n/home08/sjmathy/EGO-HAND-WM
ARCHIVE_ROOT=${ARCHIVE_ROOT:-/n/netscratch/ydu_lab/Lab/sjmathy/VITRA}
OUTPUT_DIR=${OUTPUT_DIR:-/n/netscratch/ydu_lab/Lab/sjmathy/VITRA_SHARDS}
EPISODES_PER_SHARD=${EPISODES_PER_SHARD:-2048}
CONDA_ENV=${CONDA_ENV:-base}

cd "$ROOT"
source /n/home08/sjmathy/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

python scripts/prepare_vitra_shards.py \
  "$ARCHIVE_ROOT/ego4d_cooking_and_cleaning.tar.gz" \
  "$ARCHIVE_ROOT/ego4d_other.tar.gz" \
  "$ARCHIVE_ROOT/epic.tar.gz" \
  "$ARCHIVE_ROOT/egoexo4d.tar.gz" \
  "$ARCHIVE_ROOT/ssv2.tar.gz" \
  --output-dir "$OUTPUT_DIR" \
  --episodes-per-shard "$EPISODES_PER_SHARD"
