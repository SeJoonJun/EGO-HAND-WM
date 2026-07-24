#!/usr/bin/env bash
#SBATCH --job-name=base-rv2-eval
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --array=0-4%5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/jun.se/EGO-HAND-WM/logs/slurm/%x-%A_%a.out
#SBATCH --error=/home/jun.se/EGO-HAND-WM/logs/slurm/%x-%A_%a.err

set -euo pipefail

USST=/home/jun.se/EGO-HAND-WM-REF/USST
MMTWIN=/home/jun.se/EGO-HAND-WM-REF/MMTwin
MANIFESTS=/home/jun.se/EGO-HAND-WM/data/h6_k16_manifests
DINO_REPO=/home/jun.se/EGO-HAND-WM-REF/dinov3
DINO_WEIGHT=/projects/torresani-lab/sejoon/checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
USST_PY=/projects/torresani-lab/sejoon/miniconda3/envs/usst-h6k16/bin/python
MMTWIN_PY=/projects/torresani-lab/sejoon/miniconda3/envs/mmtwin-h6k16/bin/python
USST_ROOT=/projects/torresani-lab/sejoon/runs/baselines/USST
MMTWIN_ROOT=/projects/torresani-lab/sejoon/runs/baselines/MMTwin

case "$SLURM_ARRAY_TASK_ID" in
  0)
    TAG=residual-v2-paper-h2o-seed0
    CKPT="$USST_ROOT/H2O/$TAG/snapshot/best.pth"
    [[ -s "$CKPT" ]] || { echo "Missing USST H2O best checkpoint: $CKPT" >&2; exit 2; }
    cd "$USST"
    exec "$USST_PY" -u trainval_h2o.py \
      --config config/adapted/h2o_h6_k16_res18_3d.yml \
      --tag "$TAG" --num_workers 6 --eval --checkpoint "$CKPT"
    ;;
  1)
    TAG=residual-v2-paper-egopat3d-seed0
    CKPT="$USST_ROOT/EgoPAT3D/$TAG/snapshot/best.pth"
    [[ -s "$CKPT" ]] || { echo "Missing USST EgoPAT3D best checkpoint: $CKPT" >&2; exit 2; }
    cd "$USST"
    exec "$USST_PY" -u test.py \
      --config config/adapted/egopat3d_h6_k16_res18_3d.yml \
      --tag "$TAG" --num_workers 6 --checkpoint "$CKPT"
    ;;
  2)
    DATASET=h2o
    SPLIT=test
    EVAL_MANIFEST="$MANIFESTS/h2o_test_h6_k16.jsonl"
    ;;
  3)
    DATASET=egopat3d
    SPLIT=seen
    EVAL_MANIFEST="$MANIFESTS/egopat3d_test_seen_h6_k16.jsonl"
    ;;
  4)
    DATASET=egopat3d
    SPLIT=unseen
    EVAL_MANIFEST="$MANIFESTS/egopat3d_test_novel_h6_k16.jsonl"
    ;;
esac

if (( SLURM_ARRAY_TASK_ID >= 2 )); then
  RUN="$MMTWIN_ROOT/residual-v2-paper-${DATASET}-seed0"
  CKPT="$RUN/best.pt"
  METRICS="$RUN/metrics-${SPLIT}-n1.json"
  [[ -s "$CKPT" ]] || { echo "Missing MMTwin best checkpoint: $CKPT" >&2; exit 2; }
  [[ ! -e "$METRICS" ]] || { echo "Refusing to overwrite formal metrics: $METRICS" >&2; exit 2; }
  cd "$MMTWIN"
  exec "$MMTWIN_PY" -u traineval_h6k16.py \
    --train-manifest "$MANIFESTS/${DATASET}_train_h6_k16.jsonl" \
    --val-manifest "$MANIFESTS/${DATASET}_val_h6_k16.jsonl" \
    --eval-manifest "$EVAL_MANIFEST" \
    --output-dir "$RUN" --metrics-output "$METRICS" \
    --resume "$CKPT" --evaluate-only \
    --vision-repo "$DINO_REPO" --vision-checkpoint "$DINO_WEIGHT" \
    --batch-size 4 --workers 6 --max-eval-batches 0 \
    --eval-timestep-respacing 200 --sample-gap 1 --num-samples 1 --seed 0
fi
