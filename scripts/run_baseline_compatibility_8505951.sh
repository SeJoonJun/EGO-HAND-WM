#!/usr/bin/env bash
# Use the three currently free GPUs in allocation 8505951.  GPU 2 is reserved
# for the already-running HandsOnVLM EgoPAT3D one-epoch integration run.
set -u

ROOT=/home/jun.se/EGO-HAND-WM
LOG_ROOT="$ROOT/logs/baseline-compatibility-8505951"
STATUS_ROOT="$ROOT/status/baseline-compatibility-8505951"
MANIFESTS="$ROOT/data/h6_k16_manifests"
USST=/home/jun.se/EGO-HAND-WM-REF/USST
MMTWIN=/home/jun.se/EGO-HAND-WM-REF/MMTwin
USST_PY=/projects/torresani-lab/sejoon/miniconda3/envs/usst-h6k16/bin/python
MMTWIN_PY=/projects/torresani-lab/sejoon/miniconda3/envs/mmtwin-h6k16/bin/python
DINO_REPO=/home/jun.se/EGO-HAND-WM-REF/dinov3
DINO_WEIGHTS=/projects/torresani-lab/sejoon/checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth

mkdir -p "$LOG_ROOT" "$STATUS_ROOT"
rm -f "$STATUS_ROOT"/*.{complete,failed} 2>/dev/null || true

run_usst_egopat() {
  cd "$USST" || return
  CUDA_VISIBLE_DEVICES=0 "$USST_PY" train.py \
    --config config/adapted/egopat3d_h6_k16_res18_3d.yml \
    --tag compatibility-h6k16-8505951 \
    --num_workers 4
}

run_mmtwin() {
  local gpu=$1
  local dataset=$2
  cd "$MMTWIN" || return
  CUDA_VISIBLE_DEVICES="$gpu" "$MMTWIN_PY" traineval_h6k16.py \
    --train-manifest "$MANIFESTS/${dataset}_train_h6_k16.jsonl" \
    --val-manifest "$MANIFESTS/${dataset}_val_h6_k16.jsonl" \
    --output-dir "/projects/torresani-lab/sejoon/runs/baselines/MMTwin/compatibility-${dataset}-8505951" \
    --vision-model dinov3_vitl16 \
    --vision-repo "$DINO_REPO" \
    --vision-checkpoint "$DINO_WEIGHTS" \
    --batch-size 2 --workers 2 --epochs 1 \
    --max-train-batches 5 --validation-batches 1 --metric-batches 1 \
    --diffusion-steps 100 --timestep-respacing 20 \
    --skip-checkpoint-save
}

launch() {
  local name=$1
  shift
  (
    if "$@"; then
      date --iso-8601=seconds > "$STATUS_ROOT/${name}.complete"
    else
      rc=$?
      printf '%s\n' "$rc" > "$STATUS_ROOT/${name}.failed"
      exit "$rc"
    fi
  ) > "$LOG_ROOT/${name}.log" 2>&1 &
  echo "$! $name"
}

launch usst-egopat run_usst_egopat
launch mmtwin-h2o run_mmtwin 1 h2o
launch mmtwin-egopat run_mmtwin 3 egopat3d

wait

failed=0
for name in usst-egopat mmtwin-h2o mmtwin-egopat; do
  if [[ -f "$STATUS_ROOT/${name}.complete" ]]; then
    echo "$name COMPLETE"
  else
    echo "$name FAILED"
    failed=1
  fi
done
exit "$failed"
