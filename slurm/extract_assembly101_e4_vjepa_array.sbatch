#!/usr/bin/env bash
#SBATCH --job-name=a101-e4-vjepa
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --array=0-3%4
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

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
  --batch-size 8 \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --num-shards 4
