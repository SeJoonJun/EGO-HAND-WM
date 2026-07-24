#!/usr/bin/env bash
#SBATCH --job-name=ehwm-dino-bench
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /home/jun.se/EGO-HAND-WM
/projects/torresani-lab/sejoon/miniconda3/envs/isaaclab/bin/python \
  scripts/benchmark_dinov3.py \
  --repo /home/jun.se/EGO-HAND-WM-REF/dinov3 \
  --weights /projects/torresani-lab/sejoon/checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --image-size 256 \
  --batch-sizes 8 16 32 \
  --warmup 3 \
  --iterations 10
