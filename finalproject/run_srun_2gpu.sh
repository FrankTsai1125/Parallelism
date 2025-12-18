#!/usr/bin/env bash
set +u
set +o nounset 2>/dev/null || true
set -e
set -o pipefail

# Taiwania2 (Lab 3) multi-GPU, single-node scaling (2 GPUs):
# - 1 node has up to 8x V100
# - launch 2 tasks, each task gets 1 GPU (SLURM_LOCALID selects device)
#
# Default behavior (no args):
# - total paths = 10,000,000 split across 2 tasks via --paths-total
# - steps = 1000 (heavier workload; set to 252 if you want "daily")
# - output reduced mean/std trajectory: avg_2330_2gpu.csv (written by proc0)
# - append benchmark summary row: 2gpu.csv (proc0 only; code guards against multi-task race)
#
# Usage:
#   bash run_srun_2gpu.sh
#   bash run_srun_2gpu.sh -- --type european --paths-total 10000000 --steps 1000 --avg-path-csv avg_2330_2gpu.csv

PROJECT_ID="ACD114118"
TIME_MIN="10"
GPUS_PER_NODE="2"
NTASKS_PER_NODE="2"

SRC="mc_pricer.cu"
EXE="mc_pricer"
NVCC_FLAGS=(-O3 -std=c++17 -arch=sm_70)

# Default benchmark settings (override by passing args after `--`).
TOTAL_PATHS="10000000"  # total across 8 tasks (program will split using --paths-total)
STEPS="1000"            # if you want "daily" steps, use 252
T_YEARS="1"
YAHOO_CSV="data/2330_TW.csv"
AVG_OUT="avg_2330_2gpu.csv"
BENCH_OUT="2gpu.csv"

RUN_ARGS=()
if [[ "${1:-}" == "--" ]]; then
  shift
  RUN_ARGS=("$@")
else
  RUN_ARGS=(
    --yahoo-csv "${YAHOO_CSV}"
    --T "${T_YEARS}"
    --steps "${STEPS}"
    --paths-total "${TOTAL_PATHS}"
    --avg-path-csv "${AVG_OUT}"
    --out-csv "${BENCH_OUT}"
    --append-csv
  )
fi

echo "[1/3] Loading CUDA module..."
export SLURM_JOBID="${SLURM_JOBID-}"
module load cuda

echo "[2/3] Compiling ${SRC} -> ${EXE} ..."
nvcc "${NVCC_FLAGS[@]}" "${SRC}" -o "${EXE}"

echo "[3/3] Submitting multi-GPU job via srun (1 node, ${NTASKS_PER_NODE} tasks, ${GPUS_PER_NODE} GPUs)..."

# Perf logs are written on the compute node into the shared filesystem.
RUN_TAG="${RUN_TAG:-2gpu_$(date -u +%Y%m%dT%H%M%SZ)}"
export RUN_TAG

srun -N 1 --ntasks-per-node "${NTASKS_PER_NODE}" --gpus-per-node "${GPUS_PER_NODE}" \
  -A "${PROJECT_ID}" -t "${TIME_MIN}" \
  bash -lc '
    set -euo pipefail
    LOG_ROOT="perf_logs/${RUN_TAG}_job${SLURM_JOB_ID:-unknown}"
    LOG_DIR="${LOG_ROOT}/proc${SLURM_PROCID:-0}"
    mkdir -p "${LOG_DIR}"

    {
      echo "run_tag=${RUN_TAG}"
      echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "hostname=$(hostname)"
      echo "slurm_job_id=${SLURM_JOB_ID:-}"
      echo "slurm_step_id=${SLURM_STEP_ID:-}"
      echo "slurm_procid=${SLURM_PROCID:-}"
      echo "slurm_localid=${SLURM_LOCALID:-}"
      echo "slurm_ntasks=${SLURM_NTASKS:-}"
      echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
      echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
      echo "omp_num_threads=${OMP_NUM_THREADS:-}"
    } > "${LOG_DIR}/meta.txt"

    SMI_PID=""
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi -q > "${LOG_DIR}/nvidia_smi_q.txt" || true
      nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total \
        --format=csv -l 1 > "${LOG_DIR}/nvidia_smi.csv" &
      SMI_PID="$!"
    fi
    cleanup() { if [[ -n "${SMI_PID}" ]]; then kill "${SMI_PID}" 2>/dev/null || true; fi; }
    trap cleanup EXIT

    { /usr/bin/time -v "$@"; } 2> >(tee "${LOG_DIR}/time.txt" >&2) | tee "${LOG_DIR}/stdout.log"
  ' bash "./${EXE}" "${RUN_ARGS[@]}"


