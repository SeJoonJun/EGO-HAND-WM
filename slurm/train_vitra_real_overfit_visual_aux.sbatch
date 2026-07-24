#!/usr/bin/env bash
#SBATCH --job-name=ehwm-overfit-aux
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
ROOT=/home/jun.se/EGO-HAND-WM
ENV=/projects/torresani-lab/sejoon/envs/ego-hand-wm

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

test -f "$ROOT/.smoke/vitra_episode/vitra-smoke.tar"
test -f /scratch/jun.se/EGO-HAND-WM/VITRA_DINOTXT_UNIQUE/_SUCCESS
test -f /scratch/jun.se/EGO-HAND-WM/VITRA_DINOTXT_TEXT/_SUCCESS

"$ENV/bin/python" -m ego_hand_wm.cli.train \
  --config configs/vitra_pretrain_visual_aux.yaml \
  --set data.shard_glob="$ROOT/.smoke/vitra_episode/vitra-smoke.tar" \
  --set data.shuffle_buffer=0 \
  --set training.output_dir=/scratch/jun.se/EGO-HAND-WM/runs/vitra-real-overfit-visual-aux-ln-v1 \
  --set training.batch_size=1 \
  --set training.num_workers=0 \
  --set training.drop_last=false \
  --set training.max_steps=200 \
  --set training.warmup_steps=10 \
  --set training.weight_decay=0.01 \
  --set training.gradient_accumulation_steps=1 \
  --set training.log_every=1 \
  --set training.save_every=50
