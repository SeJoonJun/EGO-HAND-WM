#!/usr/bin/env bash
#SBATCH --job-name=ehwm-dinotxt-bench
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
ROOT=/home/jun.se/EGO-HAND-WM
PYTHON=/projects/torresani-lab/sejoon/envs/ego-hand-wm/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" scripts/benchmark_dinotxt_batches.py \
  --rgb-path /scratch/jun.se/EGO-HAND-WM/VITRA_RGB_256/epic/P01_01.rgb.npy \
  --repo-path /home/jun.se/EGO-HAND-WM-REF/dinov3 \
  --weights-path /projects/torresani-lab/sejoon/checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --dinotxt-weights-path /projects/torresani-lab/sejoon/checkpoints/dinov3/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth \
  --frames 4096 \
  --batches 512 1024 2048
