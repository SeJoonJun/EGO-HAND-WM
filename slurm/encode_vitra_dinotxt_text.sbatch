#!/usr/bin/env bash
#SBATCH --job-name=ehwm-txt-encode
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
ROOT=/home/jun.se/EGO-HAND-WM
PYTHON=/projects/torresani-lab/sejoon/envs/ego-hand-wm/bin/python

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" scripts/extract_vitra_dinotxt_text.py encode \
  --collection-glob '/scratch/jun.se/EGO-HAND-WM/VITRA_DINOTXT_TEXT_COLLECTIONS/worker-*.json' \
  --output-root /scratch/jun.se/EGO-HAND-WM/VITRA_DINOTXT_TEXT \
  --repo-path /home/jun.se/EGO-HAND-WM-REF/dinov3 \
  --weights-path /projects/torresani-lab/sejoon/checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --dinotxt-weights-path /projects/torresani-lab/sejoon/checkpoints/dinov3/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth \
  --bpe-path /projects/torresani-lab/sejoon/checkpoints/dinov3/bpe_simple_vocab_16e6.txt.gz \
  --batch-size 256 \
  --device cuda
