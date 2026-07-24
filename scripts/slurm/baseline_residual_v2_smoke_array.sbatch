#!/usr/bin/env bash
#SBATCH --job-name=base-rv2-smoke
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:00:00
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

case "$SLURM_ARRAY_TASK_ID" in
  0)
    cd "$USST"
    exec "$USST_PY" trainval_h2o.py \
      --config config/adapted/h2o_h6_k16_res18_3d.yml \
      --tag residual-v2-smoke-h2o \
      --num_workers 6 --epochs 5 \
      --max_train_batches 10 --max_eval_batches 2
    ;;
  1)
    cd "$USST"
    exec "$USST_PY" train.py \
      --config config/adapted/egopat3d_h6_k16_res18_3d.yml \
      --tag residual-v2-smoke-egopat3d \
      --num_workers 6 --epochs 5 \
      --max_train_batches 10 --max_eval_batches 2
    ;;
  2)
    cd "$MMTWIN"
    exec "$MMTWIN_PY" traineval_h6k16.py \
      --train-manifest "$MANIFESTS/h2o_train_h6_k16.jsonl" \
      --val-manifest "$MANIFESTS/h2o_val_h6_k16.jsonl" \
      --output-dir /projects/torresani-lab/sejoon/runs/baselines/MMTwin/residual-v2-smoke-h2o \
      --vision-model dinov3_vitl16 --vision-repo "$DINO_REPO" \
      --vision-checkpoint "$DINO_WEIGHT" \
      --batch-size 4 --workers 4 --epochs 5 --learning-rate 1e-4 \
      --diffusion-steps 100 --validation-timestep-respacing 10 \
      --max-train-batches 10 --validation-batches 2 --metric-batches 0 \
      --checkpoint-selection ade_3d_m --require-beat-persistence --seed 0
    ;;
  3)
    cd "$MMTWIN"
    exec "$MMTWIN_PY" traineval_h6k16.py \
      --train-manifest "$MANIFESTS/egopat3d_train_h6_k16.jsonl" \
      --val-manifest "$MANIFESTS/egopat3d_val_h6_k16.jsonl" \
      --output-dir /projects/torresani-lab/sejoon/runs/baselines/MMTwin/residual-v2-smoke-egopat3d \
      --vision-model dinov3_vitl16 --vision-repo "$DINO_REPO" \
      --vision-checkpoint "$DINO_WEIGHT" \
      --batch-size 4 --workers 4 --epochs 5 --learning-rate 1e-4 \
      --diffusion-steps 100 --validation-timestep-respacing 10 \
      --max-train-batches 10 --validation-batches 2 --metric-batches 0 \
      --checkpoint-selection ade_3d_m --require-beat-persistence --seed 0
    ;;
esac
