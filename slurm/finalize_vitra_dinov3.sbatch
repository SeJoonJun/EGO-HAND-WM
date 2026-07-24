#!/usr/bin/env bash
#SBATCH --job-name=ehwm-dino-final
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
ROOT=/home/jun.se/EGO-HAND-WM
PYTHON=/projects/torresani-lab/sejoon/envs/ego-hand-wm/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" scripts/extract_vitra_dinov3.py finalize \
  --annotation-glob '/scratch/jun.se/EGO-HAND-WM/VITRA_SHARDS/vitra-*.tar' \
  --output-root /scratch/jun.se/EGO-HAND-WM/VITRA_DINO_FEATURES
