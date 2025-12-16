#!/usr/bin/env bash
set +u
set +o nounset 2>/dev/null || true
set -e
set -o pipefail

# Taiwania2 (Lab 3) multi-GPU, single-node scaling:
# - 1 node has 8x V100
# - launch 8 tasks, each task gets 1 GPU (SLURM_LOCALID selects device)
#
# Default behavior (no args):
# - total paths = 10,000,000 split across 8 tasks via --paths-total
# - steps = 1000 (heavier workload; set to 252 if you want "daily")
# - output reduced mean/std trajectory: avg_2330_8gpu.csv (written by proc0)
# - append benchmark summary row: bench_runs.csv (NOTE: multiple tasks writing same file can race;
#   you can remove --out-csv if you only need avg-path output)
#
# Usage:
#   bash run_srun_8gpu.sh
#   bash run_srun_8gpu.sh -- --type european --paths-total 10000000 --steps 1000 --avg-path-csv avg_2330_8gpu.csv

PROJECT_ID="ACD114118"
TIME_MIN="10"
GPUS_PER_NODE="8"
NTASKS_PER_NODE="8"

SRC="mc_pricer.cu"
EXE="mc_pricer"
NVCC_FLAGS=(-O3 -std=c++17 -arch=sm_70)

# Default benchmark settings (override by passing args after `--`).
TOTAL_PATHS="10000000"  # total across 8 tasks (program will split using --paths-total)
STEPS="1000"            # if you want "daily" steps, use 252
T_YEARS="1"
YAHOO_CSV="data/2330_TW.csv"
AVG_OUT="avg_2330_8gpu.csv"
BENCH_OUT="bench_runs.csv"

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

echo "[3/3] Submitting multi-GPU job via srun (1 node, 8 tasks, 8 GPUs)..."
srun -N 1 --ntasks-per-node "${NTASKS_PER_NODE}" --gpus-per-node "${GPUS_PER_NODE}" \
  -A "${PROJECT_ID}" -t "${TIME_MIN}" "./${EXE}" "${RUN_ARGS[@]}"


