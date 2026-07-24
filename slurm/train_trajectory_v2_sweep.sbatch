#!/usr/bin/env bash
#SBATCH --job-name=ehwm-traj-v2-sweep
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=5-00:00:00
#SBATCH --array=0-3%2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
ROOT=/home/jun.se/EGO-HAND-WM
PYTHON=/projects/torresani-lab/sejoon/envs/ego-hand-wm/bin/python
CONFIGS=(
  configs/benchmarks/h2o_vitra_finetune_h6k16_trajectory_v2.yaml
  configs/benchmarks/egopat3d_vitra_finetune_h6k16_trajectory_v2.yaml
  configs/benchmarks/h2o_vitra_finetune_h6k16_trajectory_v2_refine.yaml
  configs/benchmarks/egopat3d_vitra_finetune_h6k16_trajectory_v2_refine.yaml
)

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HUB_OFFLINE=1
"$PYTHON" -m ego_hand_wm.cli.train \
  --config "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
