#!/usr/bin/env bash
#SBATCH --job-name=ehwm-vitra-mn
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
ROOT=/home/jun.se/EGO-HAND-WM
ENV=/projects/torresani-lab/sejoon/envs/ego-hand-wm
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/scratch/jun.se/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8

test -f /scratch/jun.se/EGO-HAND-WM/VITRA_SHARDS/_SUCCESS
test -d /scratch/jun.se/EGO-HAND-WM/VITRA_PTS
test -f /scratch/jun.se/EGO-HAND-WM/VITRA_DINOTXT_UNIQUE/_SUCCESS
test -f /scratch/jun.se/EGO-HAND-WM/VITRA_DINOTXT_TEXT/_SUCCESS

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export MASTER_ADDR MASTER_PORT

srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  bash -c '
    /projects/torresani-lab/sejoon/envs/ego-hand-wm/bin/torchrun \
      --nnodes="$SLURM_NNODES" \
      --nproc-per-node=8 \
      --node-rank="$SLURM_PROCID" \
      --master-addr="$MASTER_ADDR" \
      --master-port="$MASTER_PORT" \
      -m ego_hand_wm.cli.train --config configs/vitra_pretrain_visual_aux.yaml
  '
