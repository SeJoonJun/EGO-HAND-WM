#!/usr/bin/env bash
#SBATCH --job-name=ehwm-vitra-resume
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
RUN_ROOT=/scratch/jun.se/EGO-HAND-WM/runs/vitra-pretrain-visual-aux-v2
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-"$RUN_ROOT/step-00125000.pt"}

if [[ "${RESUME_LATEST:-0}" == "1" ]]; then
  mapfile -t AVAILABLE_CHECKPOINTS < <(
    find "$RUN_ROOT" -maxdepth 1 -type f -name 'step-*.pt' -print | sort -V
  )
  if ((${#AVAILABLE_CHECKPOINTS[@]} == 0)); then
    echo "No step checkpoint is available under $RUN_ROOT" >&2
    exit 1
  fi
  RESUME_CHECKPOINT=${AVAILABLE_CHECKPOINTS[-1]}
fi

cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/scratch/jun.se/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

test -f /scratch/jun.se/EGO-HAND-WM/VITRA_SHARDS/_SUCCESS
test -d /scratch/jun.se/EGO-HAND-WM/VITRA_PTS
test -f /scratch/jun.se/EGO-HAND-WM/VITRA_DINOTXT_UNIQUE/_SUCCESS
test -f /scratch/jun.se/EGO-HAND-WM/VITRA_DINOTXT_TEXT/_SUCCESS
test -f /scratch/jun.se/EGO-HAND-WM/VITRA_FRAME_REQUESTS/requests.sqlite
test -f /scratch/jun.se/EGO-HAND-WM/VITRA_FRAME_REQUESTS/requests.sqlite.SUCCESS.json
test -f "$RESUME_CHECKPOINT"
test -d "$RUN_ROOT"

RESOURCE_LOG="$ROOT/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.resources.log"
RANK_LOG_DIR="$ROOT/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.ranks"
mkdir -p "$RANK_LOG_DIR"

monitor_resources() {
  while true; do
    {
      echo "=== $(date --iso-8601=seconds) ==="
      cgroup_path=$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)
      cgroup_root="/sys/fs/cgroup${cgroup_path}"
      echo "cgroup=$cgroup_root"
      for field in memory.current memory.peak memory.events memory.events.local; do
        if [[ -r "$cgroup_root/$field" ]]; then
          echo "[$field]"
          cat "$cgroup_root/$field"
        fi
      done
      if [[ -r "$cgroup_root/memory.stat" ]]; then
        echo "[memory.stat selected]"
        grep -E '^(anon|file|shmem|file_mapped|inactive_anon|active_anon|inactive_file|active_file|pgfault|pgmajfault) ' \
          "$cgroup_root/memory.stat"
      fi
      echo "[node memory]"
      free -b
      echo "[gpus]"
      nvidia-smi \
        --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv,noheader,nounits
    } >> "$RESOURCE_LOG" 2>&1
    sleep 60
  done
}

monitor_resources &
MONITOR_PID=$!
TRAIN_PID=""

cleanup_monitor() {
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
}

terminate_training() {
  echo "received termination signal at $(date --iso-8601=seconds)" >> "$RESOURCE_LOG"
  if [[ -n "$TRAIN_PID" ]]; then
    kill -TERM "$TRAIN_PID" 2>/dev/null || true
  fi
  cleanup_monitor
  exit 143
}

trap terminate_training TERM INT HUP
trap cleanup_monitor EXIT

echo "resuming $RESUME_CHECKPOINT on job $SLURM_JOB_ID at $(date --iso-8601=seconds)"

set +e
"$ENV/bin/torchrun" \
  --standalone \
  --nproc-per-node=4 \
  --log-dir "$RANK_LOG_DIR" \
  --redirects 3 \
  --tee 3 \
  -m ego_hand_wm.cli.train \
  --config configs/vitra_pretrain_visual_aux.yaml \
  --set "training.resume=$RESUME_CHECKPOINT" &
TRAIN_PID=$!
wait "$TRAIN_PID"
TRAIN_EXIT=$?
set -e

echo "torchrun exit_code=$TRAIN_EXIT at $(date --iso-8601=seconds)" | tee -a "$RESOURCE_LOG"
exit "$TRAIN_EXIT"
