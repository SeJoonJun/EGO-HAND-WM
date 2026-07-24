#!/usr/bin/env bash
# Run the four last-observed-wrist-relative Assembly101 ablations concurrently.
# Invoke this inside an interactive allocation that exposes four GPUs.

set -uo pipefail

ROOT=/home/jun.se/EGO-HAND-WM
ENV=/projects/torresani-lab/sejoon/envs/ego-hand-wm
RUN_ROOT=/projects/torresani-lab/sejoon/runs/assembly101_e4_oracle_wrist_anchor
MODES=(
  rgb_gt_wrist
  rgb_gt_whole_hand
  rgb_gt_camera_wrist
  rgb_gt_camera_whole_hand
)

cd "$ROOT" || exit 1
mkdir -p logs "$RUN_ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8

IFS=',' read -r -a GPU_DEVICES <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if (( ${#GPU_DEVICES[@]} < 4 )); then
  echo "Expected four visible GPUs, found: ${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 1
fi

RUN_ID="${SLURM_JOB_ID:-manual}"
CHILDREN=()
cleanup() {
  if (( ${#CHILDREN[@]} )); then
    kill "${CHILDREN[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

for GPU_SLOT in 0 1 2 3; do
  MODE="${MODES[$GPU_SLOT]}"
  OUTPUT_DIR="$RUN_ROOT/$MODE"
  if [[ -e "$OUTPUT_DIR/last.pt" ]]; then
    echo "Refusing to overwrite an existing run: $OUTPUT_DIR/last.pt" >&2
    exit 1
  fi
  LOG="logs/a101-e4-wrist-anchor-${RUN_ID}-${MODE}.out"
  echo "launch gpu=${GPU_DEVICES[$GPU_SLOT]} mode=$MODE log=$LOG"
  (
    export CUDA_VISIBLE_DEVICES="${GPU_DEVICES[$GPU_SLOT]}"
    exec "$ENV/bin/python" -m ego_hand_wm.cli.train_assembly101_anticipation \
      --config configs/assembly101_e4_anticipation.yaml \
      --set "data.wrist_reference=last_observed_wrist" \
      --set "model.mode=$MODE" \
      --set "training.output_dir=$OUTPUT_DIR"
  ) >"$LOG" 2>&1 &
  CHILDREN+=("$!")
done

FAILURES=0
for INDEX in 0 1 2 3; do
  if wait "${CHILDREN[$INDEX]}"; then
    echo "complete mode=${MODES[$INDEX]}"
  else
    STATUS=$?
    echo "failed mode=${MODES[$INDEX]} status=$STATUS" >&2
    FAILURES=$((FAILURES + 1))
  fi
done
CHILDREN=()
if (( FAILURES > 0 )); then
  echo "$FAILURES wrist-anchor run(s) failed." >&2
  exit 1
fi

echo "All four last-observed-wrist-relative Assembly101 ablations completed."
