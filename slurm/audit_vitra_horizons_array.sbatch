#!/usr/bin/env bash
#SBATCH --job-name=ehwm-horizon
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
OUTPUT_ROOT=${OUTPUT_ROOT:-/scratch/jun.se/EGO-HAND-WM/VITRA_HORIZON_AUDIT}
NUM_PARTITIONS=${NUM_PARTITIONS:-$SLURM_ARRAY_TASK_COUNT}
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
printf -v OUTPUT '%s/parts/horizon-part-%02d.npz' "$OUTPUT_ROOT" "$SLURM_ARRAY_TASK_ID"

"$PYTHON" scripts/audit_vitra_horizons.py partition \
  "${ARGS[@]}" \
  --output "$OUTPUT" \
  --pts-root /scratch/jun.se/EGO-HAND-WM/VITRA_PTS \
  --pts-alias ego4d_other=ego4d_cooking_and_cleaning \
  --partition-id "$SLURM_ARRAY_TASK_ID" \
  --num-partitions "$NUM_PARTITIONS" \
  --history-steps 6
