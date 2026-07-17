#!/usr/bin/env bash
#SBATCH --job-name=ehwm-vitra-mn
#SBATCH --account=ydu_lab
#SBATCH --partition=gpu,gpu_h200,seas_gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
ROOT=/n/home08/sjmathy/EGO-HAND-WM
cd "$ROOT"
source /n/home08/sjmathy/miniconda3/etc/profile.d/conda.sh
conda activate ego-hand-wm
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/n/netscratch/ydu_lab/Lab/sjmathy/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8

test -d /n/netscratch/ydu_lab/Lab/sjmathy/VITRA_SHARDS
test -d /n/netscratch/ydu_lab/Lab/sjmathy/VITRA_PTS
test -f /n/netscratch/ydu_lab/Lab/sjmathy/VITRA_DINO_FEATURES/_SUCCESS

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export MASTER_ADDR MASTER_PORT

srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  bash -c '
    torchrun \
      --nnodes="$SLURM_NNODES" \
      --nproc-per-node=8 \
      --node-rank="$SLURM_PROCID" \
      --master-addr="$MASTER_ADDR" \
      --master-port="$MASTER_PORT" \
      -m ego_hand_wm.cli.train --config configs/vitra_pretrain.yaml
  '
