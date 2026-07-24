#!/usr/bin/env bash
#SBATCH --job-name=ehwm-dinotxt-share
#SBATCH --account=l.torresani
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
ROOT=/home/jun.se/EGO-HAND-WM
ENV=/projects/torresani-lab/sejoon/envs/ego-hand-wm
NUM_WORKERS=${NUM_WORKERS:-2}

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1

"$ENV/bin/python" scripts/extract_staged_vitra_dinov3.py worker \
  --request-database /scratch/jun.se/EGO-HAND-WM/VITRA_FRAME_REQUESTS/requests.sqlite \
  --rgb-root /scratch/jun.se/EGO-HAND-WM/VITRA_RGB_256 \
  --output-root /scratch/jun.se/EGO-HAND-WM/VITRA_DINOTXT_UNIQUE \
  --repo-path /home/jun.se/EGO-HAND-WM-REF/dinov3 \
  --weights-path /projects/torresani-lab/sejoon/checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --dinotxt-weights-path /projects/torresani-lab/sejoon/checkpoints/dinov3/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth \
  --bpe-path /projects/torresani-lab/sejoon/checkpoints/dinov3/bpe_simple_vocab_16e6.txt.gz \
  --model-name dinov3_vitl16 \
  --input-size 256 \
  --spatial-grid-size 4 \
  --feature-dim 1024 \
  --batch-size 1024 \
  --device cuda \
  --worker-id "$SLURM_ARRAY_TASK_ID" \
  --num-workers "$NUM_WORKERS" \
  --assist-base-workers 6 \
  --assist-tail-fraction 0.15 \
  --worker-manifest-prefix assist-worker \
  --max-runtime-seconds 2700 \
  --available-only
