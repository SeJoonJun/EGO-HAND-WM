#!/usr/bin/env bash
#SBATCH --job-name=ehwm-traj-eval
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2%2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
ROOT=/home/jun.se/EGO-HAND-WM
PYTHON=/projects/torresani-lab/sejoon/envs/ego-hand-wm/bin/python
CONFIGS=(
  configs/benchmarks/h2o_vitra_finetune_h6k16.yaml
  configs/benchmarks/egopat3d_vitra_finetune_h6k16.yaml
  configs/benchmarks/egopat3d_vitra_finetune_h6k16.yaml
)
CHECKPOINTS=(
  /scratch/jun.se/EGO-HAND-WM/runs/h2o-vitra-h6k16-v1/best.pt
  /scratch/jun.se/EGO-HAND-WM/runs/egopat3d-vitra-h6k16-v1/best.pt
  /scratch/jun.se/EGO-HAND-WM/runs/egopat3d-vitra-h6k16-v1/best.pt
)
SPLITS=(test test_seen test_novel)
OUTPUTS=(
  /home/jun.se/EGO-HAND-WM/reports/h2o-vitra-h6k16-test.json
  /home/jun.se/EGO-HAND-WM/reports/egopat3d-vitra-h6k16-test-seen.json
  /home/jun.se/EGO-HAND-WM/reports/egopat3d-vitra-h6k16-test-novel.json
)

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HUB_OFFLINE=1
"$PYTHON" scripts/evaluate_trajectory_model.py \
  --config "${CONFIGS[$SLURM_ARRAY_TASK_ID]}" \
  --checkpoint "${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}" \
  --split "${SPLITS[$SLURM_ARRAY_TASK_ID]}" \
  --batch-size 64 \
  --num-workers 4 \
  --ode-steps 8 \
  --ode-method heun \
  --output "${OUTPUTS[$SLURM_ARRAY_TASK_ID]}"
