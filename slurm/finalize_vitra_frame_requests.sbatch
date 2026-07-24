#!/usr/bin/env bash
#SBATCH --job-name=ehwm-frame-merge
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
ROOT=/home/jun.se/EGO-HAND-WM
PYTHON=/projects/torresani-lab/sejoon/envs/ego-hand-wm/bin/python
OUTPUT_ROOT=/scratch/jun.se/EGO-HAND-WM/VITRA_FRAME_REQUESTS

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mapfile -t PARTS < <(find "$OUTPUT_ROOT/parts" -maxdepth 1 -type f -name 'requests-part-*.sqlite' | sort)
if (( ${#PARTS[@]} != 32 )); then
  echo "Expected 32 frame-request partitions, found ${#PARTS[@]}" >&2
  exit 2
fi
ARGS=()
for part in "${PARTS[@]}"; do
  ARGS+=(--partition "$part")
done

"$PYTHON" scripts/build_vitra_frame_requests.py merge \
  "${ARGS[@]}" \
  --output "$OUTPUT_ROOT/requests.sqlite" \
  --pts-root /scratch/jun.se/EGO-HAND-WM/VITRA_PTS \
  --pts-alias ego4d_other=ego4d_cooking_and_cleaning
