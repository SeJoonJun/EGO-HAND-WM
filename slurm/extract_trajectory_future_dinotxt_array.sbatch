#!/usr/bin/env bash
#SBATCH --job-name=ehwm-traj-future
#SBATCH --account=l.torresani
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --array=0-3%2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
ROOT=/home/jun.se/EGO-HAND-WM
PYTHON=/projects/torresani-lab/sejoon/envs/ego-hand-wm/bin/python
MANIFEST_ROOT="$ROOT/data/h6_k16_manifests"
MANIFESTS=(
  h2o_train_h6_k16.jsonl
  h2o_val_h6_k16.jsonl
  egopat3d_train_h6_k16.jsonl
  egopat3d_val_h6_k16.jsonl
)

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HUB_OFFLINE=1

"$PYTHON" scripts/extract_trajectory_dinotxt.py \
  --manifest "$MANIFEST_ROOT/${MANIFESTS[$SLURM_ARRAY_TASK_ID]}" \
  --frame-field future_indices \
  --output-root /scratch/jun.se/EGO-HAND-WM/TRAJECTORY_DINOTXT \
  --repo-path /home/jun.se/EGO-HAND-WM-REF/dinov3 \
  --weights-path /projects/torresani-lab/sejoon/checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --dinotxt-weights-path /projects/torresani-lab/sejoon/checkpoints/dinov3/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth \
  --bpe-path /projects/torresani-lab/sejoon/checkpoints/dinov3/bpe_simple_vocab_16e6.txt.gz \
  --model-name dinov3_vitl16 \
  --input-size 256 \
  --spatial-grid-size 4 \
  --batch-size 128 \
  --device cuda
