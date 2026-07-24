#!/usr/bin/env bash
#SBATCH --job-name=ehwm-dinotxt-smoke
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
ROOT=/home/jun.se/EGO-HAND-WM
PYTHON=/projects/torresani-lab/sejoon/envs/ego-hand-wm/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" scripts/smoke_dinotxt_real.py \
  --rgb-root /scratch/jun.se/EGO-HAND-WM/VITRA_RGB_256 \
  --dataset epic \
  --video P01_01 \
  --repo-path /home/jun.se/EGO-HAND-WM-REF/dinov3 \
  --weights-path /projects/torresani-lab/sejoon/checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --dinotxt-weights-path /projects/torresani-lab/sejoon/checkpoints/dinov3/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth \
  --bpe-path /projects/torresani-lab/sejoon/checkpoints/dinov3/bpe_simple_vocab_16e6.txt.gz \
  --frames 2 \
  --device cuda
