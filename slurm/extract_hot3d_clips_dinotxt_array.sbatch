#!/usr/bin/env bash
#SBATCH --job-name=ehwm-hot3d-dinotxt
#SBATCH --account=l.torresani
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-4%2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
ROOT=/home/jun.se/EGO-HAND-WM
PYTHON=/projects/torresani-lab/sejoon/envs/ego-hand-wm/bin/python
MANIFEST_ROOT="$ROOT/data/hot3d_clips_h6_k16"
SPLITS=(train validation test train validation)
SEQUENCES=(history history history future future)

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HUB_OFFLINE=1

SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
SEQUENCE=${SEQUENCES[$SLURM_ARRAY_TASK_ID]}
FRAME_ARGS=()
if [[ "$SEQUENCE" == future ]]; then
  FRAME_ARGS=(--frame-field future_indices)
fi

"$PYTHON" scripts/extract_trajectory_dinotxt.py \
  --manifest "$MANIFEST_ROOT/hot3d_clips_aria_${SPLIT}_h6_k16.jsonl" \
  "${FRAME_ARGS[@]}" \
  --output-root /scratch/jun.se/EGO-HAND-WM/TRAJECTORY_DINOTXT \
  --repo-path /home/jun.se/EGO-HAND-WM-REF/dinov3 \
  --weights-path /projects/torresani-lab/sejoon/checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --dinotxt-weights-path /projects/torresani-lab/sejoon/checkpoints/dinov3/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth \
  --bpe-path /projects/torresani-lab/sejoon/checkpoints/dinov3/bpe_simple_vocab_16e6.txt.gz \
  --model-name dinov3_vitl16 \
  --input-size 256 \
  --spatial-grid-size 4 \
  --batch-size 128 \
  --hot3d-camera-stream 214-1 \
  --device cuda
