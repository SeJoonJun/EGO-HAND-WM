#!/usr/bin/env bash
#SBATCH --job-name=a101-e4-oracle
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
ROOT=/home/jun.se/EGO-HAND-WM
ENV=/projects/torresani-lab/sejoon/envs/ego-hand-wm
RUN_ROOT=/projects/torresani-lab/sejoon/runs/assembly101_e4_oracle
MODES=(
  rgb
  rgb_gt_camera
  rgb_gt_wrist
  rgb_gt_handpose
  rgb_gt_whole_hand
  rgb_gt_camera_wrist
  rgb_gt_camera_handpose
  rgb_gt_camera_whole_hand
)
MODE="${MODES[$SLURM_ARRAY_TASK_ID]}"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8

"$ENV/bin/python" -m ego_hand_wm.cli.train_assembly101_anticipation \
  --config configs/assembly101_e4_anticipation.yaml \
  --set "model.mode=$MODE" \
  --set "training.output_dir=$RUN_ROOT/$MODE"
