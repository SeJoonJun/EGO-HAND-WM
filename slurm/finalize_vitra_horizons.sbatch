#!/usr/bin/env bash
#SBATCH --job-name=ehwm-horizon-merge
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
ROOT=/home/jun.se/EGO-HAND-WM
PYTHON=/projects/torresani-lab/sejoon/envs/ego-hand-wm/bin/python
OUTPUT_ROOT=/scratch/jun.se/EGO-HAND-WM/VITRA_HORIZON_AUDIT
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mapfile -t PARTS < <(find "$OUTPUT_ROOT/parts" -maxdepth 1 -type f -name 'horizon-part-*.npz' | sort)
if (( ${#PARTS[@]} != 32 )); then
  echo "Expected 32 horizon partitions, found ${#PARTS[@]}" >&2
  exit 2
fi
ARGS=()
for part in "${PARTS[@]}"; do
  ARGS+=(--partition "$part")
done
"$PYTHON" scripts/audit_vitra_horizons.py merge \
  "${ARGS[@]}" \
  --output "$OUTPUT_ROOT/summary.json"
