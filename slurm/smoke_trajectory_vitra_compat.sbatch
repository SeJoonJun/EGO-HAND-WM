#!/usr/bin/env bash
#SBATCH --job-name=ehwm-traj-compat
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --array=0-1%1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
ROOT=/home/jun.se/EGO-HAND-WM
PYTHON=/projects/torresani-lab/sejoon/envs/ego-hand-wm/bin/python
CONFIGS=(
  configs/benchmarks/h2o_vitra_finetune_h6k16.yaml
  configs/benchmarks/egopat3d_vitra_finetune_h6k16.yaml
)
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HUB_OFFLINE=1
"$PYTHON" scripts/smoke_trajectory_vitra_compat.py \
  --config "${CONFIGS[$SLURM_ARRAY_TASK_ID]}" \
  --checkpoint /scratch/jun.se/EGO-HAND-WM/runs/vitra-pretrain-visual-aux-v2/last.pt

