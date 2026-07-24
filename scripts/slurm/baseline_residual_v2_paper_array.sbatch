#!/usr/bin/env bash
#SBATCH --job-name=base-rv2-paper
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --array=0-3%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
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

case "$SLURM_ARRAY_TASK_ID" in
  0)
    RUN=/projects/torresani-lab/sejoon/runs/baselines/USST/H2O/residual-v2-paper-h2o-seed0
    if [[ -e "$RUN" ]]; then
      echo "Refusing to overwrite corrected paper run: $RUN" >&2
      exit 2
    fi
    cd "$USST"
    exec "$USST_PY" -u trainval_h2o.py \
      --config config/adapted/h2o_h6_k16_res18_3d.yml \
      --tag residual-v2-paper-h2o-seed0 --num_workers 6
    ;;
  1)
    RUN=/projects/torresani-lab/sejoon/runs/baselines/USST/EgoPAT3D/residual-v2-paper-egopat3d-seed0
    if [[ -e "$RUN" ]]; then
      echo "Refusing to overwrite corrected paper run: $RUN" >&2
      exit 2
    fi
    cd "$USST"
    exec "$USST_PY" -u train.py \
      --config config/adapted/egopat3d_h6_k16_res18_3d.yml \
      --tag residual-v2-paper-egopat3d-seed0 --num_workers 6
    ;;
  2)
    RUN=/projects/torresani-lab/sejoon/runs/baselines/MMTwin/residual-v2-paper-h2o-seed0
    if [[ -e "$RUN" ]]; then
      echo "Refusing to overwrite corrected paper run: $RUN" >&2
      exit 2
    fi
    cd "$MMTWIN"
    exec "$MMTWIN_PY" -u traineval_h6k16.py \
      --train-manifest "$MANIFESTS/h2o_train_h6_k16.jsonl" \
      --val-manifest "$MANIFESTS/h2o_val_h6_k16.jsonl" \
      --output-dir "$RUN" \
      --vision-model dinov3_vitl16 --vision-repo "$DINO_REPO" \
      --vision-checkpoint "$DINO_WEIGHT" \
      --batch-size 4 --workers 6 --epochs 30 --learning-rate 5e-5 \
      --diffusion-steps 1000 --validation-timestep-respacing 50 \
      --validation-batches 0 --metric-batches 4 --full-metric-interval 5 \
      --checkpoint-selection ade_3d_m --require-beat-persistence --seed 0
    ;;
  3)
    RUN=/projects/torresani-lab/sejoon/runs/baselines/MMTwin/residual-v2-paper-egopat3d-seed0
    if [[ -e "$RUN" ]]; then
      echo "Refusing to overwrite corrected paper run: $RUN" >&2
      exit 2
    fi
    cd "$MMTWIN"
    exec "$MMTWIN_PY" -u traineval_h6k16.py \
      --train-manifest "$MANIFESTS/egopat3d_train_h6_k16.jsonl" \
      --val-manifest "$MANIFESTS/egopat3d_val_h6_k16.jsonl" \
      --output-dir "$RUN" \
      --vision-model dinov3_vitl16 --vision-repo "$DINO_REPO" \
      --vision-checkpoint "$DINO_WEIGHT" \
      --batch-size 4 --workers 6 --epochs 30 --learning-rate 5e-5 \
      --diffusion-steps 1000 --validation-timestep-respacing 50 \
      --validation-batches 0 --metric-batches 4 --full-metric-interval 5 \
      --checkpoint-selection ade_3d_m --require-beat-persistence --seed 0
    ;;
esac
