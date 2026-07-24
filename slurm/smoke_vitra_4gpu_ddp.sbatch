#!/usr/bin/env bash
#SBATCH --job-name=ehwm-ddp-gate
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
ROOT=/home/jun.se/EGO-HAND-WM
ENV=/projects/torresani-lab/sejoon/envs/ego-hand-wm
OUTPUT=/scratch/jun.se/EGO-HAND-WM/runs/vitra-ddp-qknorm-validation-v1

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/scratch/jun.se/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export NCCL_ASYNC_ERROR_HANDLING=1

test -f /scratch/jun.se/EGO-HAND-WM/VITRA_SHARDS/_SUCCESS
test -d /scratch/jun.se/EGO-HAND-WM/VITRA_PTS
test -f /scratch/jun.se/EGO-HAND-WM/VITRA_DINOTXT_UNIQUE/_SUCCESS
test -f /scratch/jun.se/EGO-HAND-WM/VITRA_DINOTXT_TEXT/_SUCCESS
test ! -e "$OUTPUT"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"$ENV/bin/torchrun" --standalone --nproc-per-node=4 \
  -m ego_hand_wm.cli.train \
  --config configs/vitra_pretrain_visual_aux.yaml \
  --set training.output_dir="$OUTPUT" \
  --set training.batch_size=16 \
  --set training.num_workers=4 \
  --set training.max_steps=1000 \
  --set training.warmup_steps=50 \
  --set training.gradient_accumulation_steps=4 \
  --set training.log_every=1 \
  --set training.save_every=0 \
  --set validation.every=500 \
  --set validation.ode_steps=2 \
  --set validation.ode_method=euler

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
