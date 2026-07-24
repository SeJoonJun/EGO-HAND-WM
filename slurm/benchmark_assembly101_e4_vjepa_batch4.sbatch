#!/usr/bin/env bash
#SBATCH --job-name=a101-vjepa-b4
#SBATCH --account=l.torresani
#SBATCH --partition=sharing
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
ROOT=/home/jun.se/EGO-HAND-WM
ENV=/projects/torresani-lab/sejoon/envs/ego-hand-wm
DATA=/projects/torresani-lab/sejoon/datasets/Assembly101
ANNO="$DATA/action_anticipation/official_annotations/data/CSVs"
CHECKPOINT=/projects/torresani-lab/sejoon/checkpoints/vjepa2/vjepa2_1_vitG_384.pt
BATCH_SIZE="${BATCH_SIZE:-4}"
OUTPUT="/tmp/a101-vjepa-b${BATCH_SIZE}-$SLURM_JOB_ID"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8

/usr/bin/time -v "$ENV/bin/python" scripts/extract_assembly101_e4_vjepa.py \
  --annotations-root "$ANNO" \
  --recordings-root "$DATA/recordings" \
  --output-root "$OUTPUT" \
  --checkpoint "$CHECKPOINT" \
  --vjepa-repository /home/jun.se/EGO-HAND-WM-REF/vjepa2 \
  --batch-size "$BATCH_SIZE" \
  --index 0
