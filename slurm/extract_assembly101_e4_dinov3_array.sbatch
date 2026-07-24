#!/usr/bin/env bash
#SBATCH --job-name=a101-e4-dino
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --array=0-355%8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
ROOT=/home/jun.se/EGO-HAND-WM
ENV=/projects/torresani-lab/sejoon/envs/ego-hand-wm
DATA=/projects/torresani-lab/sejoon/datasets/Assembly101
ANNO="$DATA/action_anticipation/official_annotations/CSVs"
OUTPUT="$DATA/derived/e4_anticipation/dinov3_vitl16"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$ENV/bin/python" scripts/extract_assembly101_e4_dinov3.py \
  --recordings-root "$DATA/recordings" \
  --annotations "$ANNO/train.csv" \
  --annotations "$ANNO/validation.csv" \
  --annotations "$ANNO/test.csv" \
  --output-root "$OUTPUT" \
  --repo-path /home/jun.se/EGO-HAND-WM-REF/dinov3 \
  --weights-path /projects/torresani-lab/sejoon/checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --model-name dinov3_vitl16 \
  --input-size 256 \
  --batch-size 32 \
  --index "$SLURM_ARRAY_TASK_ID"

