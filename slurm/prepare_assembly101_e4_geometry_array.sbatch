#!/usr/bin/env bash
#SBATCH --job-name=a101-e4-geom
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab,sharing
#SBATCH --array=0-15%8
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:59:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
ROOT=/home/jun.se/EGO-HAND-WM
ENV=/projects/torresani-lab/sejoon/envs/ego-hand-wm
DATA=/projects/torresani-lab/sejoon/datasets/Assembly101
ANNO="$DATA/action_anticipation/official_annotations/data/CSVs"
OUTPUT="$DATA/derived/e4_anticipation/geometry_oracle_8hz"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$ENV/bin/python" scripts/prepare_assembly101_e4_geometry.py \
  --poses-zip "$DATA/AssemblyPoses.zip" \
  --annotations "$ANNO/train.csv" \
  --annotations "$ANNO/validation.csv" \
  --output-root "$OUTPUT" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --num-shards 16
