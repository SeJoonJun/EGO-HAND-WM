#!/usr/bin/env bash
#SBATCH --job-name=ehwm-rgb-smoke
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
ROOT=/home/jun.se/EGO-HAND-WM
PYTHON=/projects/torresani-lab/sejoon/envs/ego-hand-wm/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export EGO_HAND_WM_DECODE_THREADS=4

"$PYTHON" scripts/stage_vitra_rgb.py \
  --request-database "$ROOT/.smoke/rgb-requests.sqlite" \
  --output-root "$ROOT/.smoke/rgb" \
  --pts-root /scratch/jun.se/EGO-HAND-WM/VITRA_PTS \
  --worker-id 0 \
  --num-workers 1 \
  --video-root ego4d_cooking_and_cleaning=/projects/torresani-lab/sejoon/datasets/VITRA/videos/ego4d_undistorted

"$PYTHON" -c "import numpy as np; p='$ROOT/.smoke/rgb/ego4d_cooking_and_cleaning/46af7819-8dfd-4c3c-9b21-5ad0d9095580.rgb.npy'; x=np.load(p,mmap_mode='r'); print({'shape':x.shape,'dtype':str(x.dtype),'minimum':int(x.min()),'maximum':int(x.max())})"
