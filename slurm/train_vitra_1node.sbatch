#!/usr/bin/env bash
#SBATCH --job-name=ehwm-vitra
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
ROOT=/home/jun.se/EGO-HAND-WM
ENV=/projects/torresani-lab/sejoon/envs/ego-hand-wm
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/scratch/jun.se/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

test -f /scratch/jun.se/EGO-HAND-WM/VITRA_SHARDS/_SUCCESS
test -d /scratch/jun.se/EGO-HAND-WM/VITRA_PTS
test -f /scratch/jun.se/EGO-HAND-WM/VITRA_DINOTXT_UNIQUE/_SUCCESS
test -f /scratch/jun.se/EGO-HAND-WM/VITRA_DINOTXT_TEXT/_SUCCESS
test -f /scratch/jun.se/EGO-HAND-WM/VITRA_FRAME_REQUESTS/requests.sqlite
test -f /scratch/jun.se/EGO-HAND-WM/VITRA_FRAME_REQUESTS/requests.sqlite.SUCCESS.json
test ! -e /scratch/jun.se/EGO-HAND-WM/runs/vitra-pretrain-visual-aux-v2

"$ENV/bin/torchrun" --standalone --nproc-per-node=4 \
  -m ego_hand_wm.cli.train --config configs/vitra_pretrain_visual_aux.yaml
