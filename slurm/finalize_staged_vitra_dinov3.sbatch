#!/usr/bin/env bash
#SBATCH --job-name=ehwm-dino-finalize
#SBATCH --account=l.torresani
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
ROOT=/home/jun.se/EGO-HAND-WM
ENV=/projects/torresani-lab/sejoon/envs/ego-hand-wm
NUM_WORKERS=${NUM_WORKERS:-6}
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$ENV/bin/python" scripts/extract_staged_vitra_dinov3.py finalize \
  --request-database /scratch/jun.se/EGO-HAND-WM/VITRA_FRAME_REQUESTS/requests.sqlite \
  --output-root /scratch/jun.se/EGO-HAND-WM/VITRA_DINOTXT_UNIQUE \
  --num-workers "$NUM_WORKERS"
