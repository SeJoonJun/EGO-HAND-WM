#!/usr/bin/env bash
#SBATCH --job-name=ehwm-txt-collect
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --array=0-63%32
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
ROOT=/home/jun.se/EGO-HAND-WM
PYTHON=/projects/torresani-lab/sejoon/envs/ego-hand-wm/bin/python
SHARD_ROOT=/scratch/jun.se/EGO-HAND-WM/VITRA_SHARDS
OUTPUT_ROOT=/scratch/jun.se/EGO-HAND-WM/VITRA_DINOTXT_TEXT_COLLECTIONS
WORKERS=${SLURM_ARRAY_TASK_COUNT:-64}

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_ROOT"
mapfile -t ALL_SHARDS < <(find "$SHARD_ROOT" -maxdepth 1 -type f -name '*.tar' | sort)
ASSIGNED=()
for ((INDEX=SLURM_ARRAY_TASK_ID; INDEX<${#ALL_SHARDS[@]}; INDEX+=WORKERS)); do
  ASSIGNED+=("${ALL_SHARDS[$INDEX]}")
done
if ((${#ASSIGNED[@]} == 0)); then
  exit 0
fi
"$PYTHON" scripts/extract_vitra_dinotxt_text.py collect \
  --annotation-shard "${ASSIGNED[@]}" \
  --output-path "$OUTPUT_ROOT/worker-$(printf '%03d' "$SLURM_ARRAY_TASK_ID").json"
