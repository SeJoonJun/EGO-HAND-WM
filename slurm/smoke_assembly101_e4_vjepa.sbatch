#!/usr/bin/env bash
#SBATCH --job-name=a101-e4-vjepa-smoke
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
ROOT=/home/jun.se/EGO-HAND-WM
ENV=/projects/torresani-lab/sejoon/envs/ego-hand-wm
DATA=/projects/torresani-lab/sejoon/datasets/Assembly101
ANNO="$DATA/action_anticipation/official_annotations/data/CSVs"
OUTPUT="$DATA/derived/e4_anticipation/vjepa2_vitg16_384"
CHECKPOINT=/projects/torresani-lab/sejoon/checkpoints/vjepa2/vjepa2_1_vitG_384.pt
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8

"$ENV/bin/python" scripts/extract_assembly101_e4_vjepa.py \
  --annotations-root "$ANNO" \
  --recordings-root "$DATA/recordings" \
  --output-root "$OUTPUT" \
  --checkpoint "$CHECKPOINT" \
  --vjepa-repository /home/jun.se/EGO-HAND-WM-REF/vjepa2 \
  --batch-size 1 \
  --index 0
