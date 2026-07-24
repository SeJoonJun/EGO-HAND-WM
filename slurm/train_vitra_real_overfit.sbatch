#!/usr/bin/env bash
#SBATCH --job-name=ehwm-real-overfit
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
ROOT=/home/jun.se/EGO-HAND-WM
ENV=/projects/torresani-lab/sejoon/envs/ego-hand-wm

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

test -f "$ROOT/.smoke/features/_SUCCESS"
test -f "$ROOT/.smoke/vitra_episode/vitra-smoke.tar"
test -f /scratch/jun.se/EGO-HAND-WM/VITRA_PTS/ego4d_cooking_and_cleaning/46af7819-8dfd-4c3c-9b21-5ad0d9095580.npy

args=(--config configs/vitra_real_overfit.yaml)
if [[ -n "${EGO_HAND_WM_RESUME:-}" ]]; then
  test -f "$EGO_HAND_WM_RESUME"
  args+=(--set "training.resume=$EGO_HAND_WM_RESUME")
fi

"$ENV/bin/python" -m ego_hand_wm.cli.train "${args[@]}"
