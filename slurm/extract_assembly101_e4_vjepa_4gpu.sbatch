#!/usr/bin/env bash
#SBATCH --job-name=a101-e4-vjepa-4g
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
export ROOT=/home/jun.se/EGO-HAND-WM
export ENV=/projects/torresani-lab/sejoon/envs/ego-hand-wm
export DATA=/projects/torresani-lab/sejoon/datasets/Assembly101
export ANNO="$DATA/action_anticipation/official_annotations/data/CSVs"
export OUTPUT="$DATA/derived/e4_anticipation/vjepa2_vitg16_384"
export CHECKPOINT=/projects/torresani-lab/sejoon/checkpoints/vjepa2/vjepa2_1_vitG_384.pt
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8

srun \
  --ntasks=4 \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  --output="logs/%x-%j_%t.out" \
  --error="logs/%x-%j_%t.err" \
  bash -c '
    set -euo pipefail
    exec "$ENV/bin/python" scripts/extract_assembly101_e4_vjepa.py \
      --annotations-root "$ANNO" \
      --recordings-root "$DATA/recordings" \
      --output-root "$OUTPUT" \
      --checkpoint "$CHECKPOINT" \
      --vjepa-repository /home/jun.se/EGO-HAND-WM-REF/vjepa2 \
      --batch-size 8 \
      --shard-index "$SLURM_PROCID" \
      --num-shards 4
  '
