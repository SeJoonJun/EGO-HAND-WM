#!/bin/bash
#SBATCH --job-name=hot3d-manifests
#SBATCH --account=l.torresani
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/jun.se/EGO-HAND-WM/logs/hot3d-manifests-%j.out
#SBATCH --error=/home/jun.se/EGO-HAND-WM/logs/hot3d-manifests-%j.err

set -euo pipefail

REPOSITORY_ROOT=/home/jun.se/EGO-HAND-WM
DATA_ROOT=/projects/torresani-lab/sejoon/datasets/HOT3D-Clips
MANIFEST_ROOT=/home/jun.se/EGO-HAND-WM/data/hot3d_clips_h6_k16
PYTHON_BIN=/projects/torresani-lab/sejoon/miniconda3/bin/python

cd "$REPOSITORY_ROOT"

PYTHONPATH=src "$PYTHON_BIN" scripts/build_hot3d_clips_h6k16_manifests.py \
  --root "$DATA_ROOT" \
  --output-dir "$MANIFEST_ROOT" \
  --scan-workers 16
