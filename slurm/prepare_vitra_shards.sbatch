#!/usr/bin/env bash
#SBATCH --job-name=ehwm-vitra-shards
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/jun.se/EGO-HAND-WM/logs/%x-%j.out
#SBATCH --error=/home/jun.se/EGO-HAND-WM/logs/%x-%j.err

set -euo pipefail

ROOT=/home/jun.se/EGO-HAND-WM
ARCHIVE_ROOT=${ARCHIVE_ROOT:-/scratch/jun.se/VITRA-1M}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/jun.se/EGO-HAND-WM/VITRA_SHARDS}
EPISODES_PER_SHARD=${EPISODES_PER_SHARD:-2048}
PYTHON_BIN=${PYTHON_BIN:-/projects/torresani-lab/sejoon/miniconda3/bin/python}

cd "$ROOT"

"$PYTHON_BIN" scripts/prepare_vitra_shards.py \
  "$ARCHIVE_ROOT/ego4d_cooking_and_cleaning.tar.gz" \
  "$ARCHIVE_ROOT/ego4d_other.tar.gz" \
  "$ARCHIVE_ROOT/epic.tar.gz" \
  "$ARCHIVE_ROOT/egoexo4d.tar.gz" \
  "$ARCHIVE_ROOT/ssv2.tar.gz" \
  --output-dir "$OUTPUT_DIR" \
  --episodes-per-shard "$EPISODES_PER_SHARD"
