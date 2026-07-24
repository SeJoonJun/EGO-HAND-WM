#!/usr/bin/env bash
#SBATCH --job-name=mmt-rv2-smoke
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/jun.se/EGO-HAND-WM/logs/slurm/%x-%A_%a.out
#SBATCH --error=/home/jun.se/EGO-HAND-WM/logs/slurm/%x-%A_%a.err

set -euo pipefail

REPO=/home/jun.se/EGO-HAND-WM-REF/MMTwin
PYTHON=/projects/torresani-lab/sejoon/miniconda3/envs/mmtwin-h6k16/bin/python
MANIFESTS=/home/jun.se/EGO-HAND-WM/data/h6_k16_manifests
DINO_REPO=/home/jun.se/EGO-HAND-WM-REF/dinov3
DINO_WEIGHT=/projects/torresani-lab/sejoon/checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth

if [[ "$SLURM_ARRAY_TASK_ID" == 0 ]]; then
  dataset=h2o
else
  dataset=egopat3d
fi

cd "$REPO"
exec "$PYTHON" traineval_h6k16.py \
  --train-manifest "$MANIFESTS/${dataset}_train_h6_k16.jsonl" \
  --val-manifest "$MANIFESTS/${dataset}_val_h6_k16.jsonl" \
  --output-dir "/projects/torresani-lab/sejoon/runs/baselines/MMTwin/residual-v2-extended-smoke-${dataset}" \
  --vision-model dinov3_vitl16 --vision-repo "$DINO_REPO" \
  --vision-checkpoint "$DINO_WEIGHT" \
  --batch-size 4 --workers 4 --epochs 20 --learning-rate 1e-4 \
  --diffusion-steps 100 --validation-timestep-respacing 10 \
  --max-train-batches 25 --validation-batches 2 --metric-batches 0 \
  --checkpoint-selection ade_3d_m --require-beat-persistence --seed 0
