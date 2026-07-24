#!/usr/bin/env bash
#SBATCH --job-name=ehwm-batch-bench
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --array=0-2%1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

ROOT=/home/jun.se/EGO-HAND-WM
ENV=/projects/torresani-lab/sejoon/envs/ego-hand-wm
BATCH_SIZES=(16 32 64)
ACCUMULATION=(4 2 1)
INDEX=${SLURM_ARRAY_TASK_ID}
BATCH=${BATCH_SIZES[$INDEX]}
ACCUM=${ACCUMULATION[$INDEX]}
OUTPUT=/scratch/jun.se/EGO-HAND-WM/benchmarks/refine-b${BATCH}-a${ACCUM}-${SLURM_ARRAY_JOB_ID}

cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/scratch/jun.se/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

"$ENV/bin/torchrun" \
  --standalone \
  --nproc-per-node=4 \
  -m ego_hand_wm.cli.train \
  --config configs/vitra_geometry_kinematics_refine.yaml \
  --set "training.output_dir=$OUTPUT" \
  --set training.overwrite=true \
  --set training.max_steps=100 \
  --set training.warmup_steps=0 \
  --set "training.batch_size=$BATCH" \
  --set "training.gradient_accumulation_steps=$ACCUM" \
  --set training.save_every=0 \
  --set training.model_only_save_every=0 \
  --set training.save_last=false \
  --set validation.enabled=false
