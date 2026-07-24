#!/usr/bin/env bash
#SBATCH --job-name=hovlm-ego-best-eval
#SBATCH --account=l.torresani
#SBATCH --partition=torresani-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/jun.se/EGO-HAND-WM/logs/%x-%j.out
#SBATCH --error=/home/jun.se/EGO-HAND-WM/logs/%x-%j.err

set -euo pipefail

REPO=/home/jun.se/EGO-HAND-WM-REF/HandsOnVLM-release
ENV=/projects/torresani-lab/sejoon/miniconda3/envs/handsonvlm-h6k16
MANIFESTS=/home/jun.se/EGO-HAND-WM/data/h6_k16_manifests
RUN=/projects/torresani-lab/sejoon/runs/baselines/HandsOnVLM/paper-h6k16-xyz-seed0-egopat3d
CHECKPOINT="$RUN/checkpoint-334"
OUTPUT_DIR="$RUN/evaluation"

test -s "$CHECKPOINT/config.json"
test -s "$CHECKPOINT/pytorch_model.bin.index.json"

export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUTPUT_DIR"
cd "$REPO"

splits=(seen unseen)
manifests=(
  egopat3d_test_seen_h6_k16.jsonl
  egopat3d_test_novel_h6_k16.jsonl
)

for num_samples in 1 5 10; do
  for index in "${!splits[@]}"; do
    split=${splits[$index]}
    manifest=${manifests[$index]}
    output="$OUTPUT_DIR/${split}_n${num_samples}_seed0.npz"
    if [[ -s "$output" ]]; then
      echo "Skipping completed result: $output"
      continue
    fi
    echo "[$(date --iso-8601=seconds)] Evaluating EgoPAT3D best checkpoint: split=$split N=$num_samples"
    CHECKPOINT="$CHECKPOINT" \
    MANIFEST="$MANIFESTS/$manifest" \
    OUTPUT="$output" \
    BATCH_SIZE=2 \
    WORKERS=2 \
    NUM_SAMPLES="$num_samples" \
    SEED=0 \
      bash scripts/evaluate_h6k16.sh
  done
done
