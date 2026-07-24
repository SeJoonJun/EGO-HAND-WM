#!/usr/bin/env bash
#SBATCH --job-name=ehwm-frame-index
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --array=0-31%16
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
ROOT=/home/jun.se/EGO-HAND-WM
PYTHON=/projects/torresani-lab/sejoon/envs/ego-hand-wm/bin/python
SHARD_ROOT=/scratch/jun.se/EGO-HAND-WM/VITRA_SHARDS
OUTPUT_ROOT=/scratch/jun.se/EGO-HAND-WM/VITRA_FRAME_REQUESTS

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_ROOT/parts"
mapfile -t SHARDS < <(find "$SHARD_ROOT" -maxdepth 1 -type f -name 'vitra-*.tar' | sort)
if (( ${#SHARDS[@]} != 598 )); then
  echo "Expected 598 annotation shards, found ${#SHARDS[@]}" >&2
  exit 2
fi
ARGS=()
for shard in "${SHARDS[@]}"; do
  ARGS+=(--annotation-shard "$shard")
done

printf -v OUTPUT '%s/parts/requests-part-%02d.sqlite' "$OUTPUT_ROOT" "$SLURM_ARRAY_TASK_ID"
"$PYTHON" scripts/build_vitra_frame_requests.py partition \
  "${ARGS[@]}" \
  --output "$OUTPUT" \
  --partition-id "$SLURM_ARRAY_TASK_ID" \
  --num-partitions "$SLURM_ARRAY_TASK_COUNT"
