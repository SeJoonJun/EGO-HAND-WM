#!/usr/bin/env bash
#SBATCH --job-name=ehwm-traj-v1-audit
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
ROOT=/home/jun.se/EGO-HAND-WM
PYTHON=/projects/torresani-lab/sejoon/envs/ego-hand-wm/bin/python
DATASETS=(h2o egopat3d)
OUTPUTS=(
  reports/h2o-vitra-h6k16-v1-inference-audit-validation.json
  reports/egopat3d-vitra-h6k16-v1-inference-audit-validation.json
)

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HUB_OFFLINE=1
"$PYTHON" scripts/audit_trajectory_v1_inference.py \
  --dataset "${DATASETS[$SLURM_ARRAY_TASK_ID]}" \
  --batch-size 64 \
  --num-workers 4 \
  --ode-steps 8 16 \
  --noise-scales 0.0 0.5 1.0 \
  --output "${OUTPUTS[$SLURM_ARRAY_TASK_ID]}"
