#!/usr/bin/env bash
set +u
set +o nounset 2>/dev/null || true
set -e
set -o pipefail

# Taiwania2 (Lab 3) CPU baseline using OpenMP.
# - Compiles on login node
# - Runs on compute node via srun (NO GPU requested)
#
# Default behavior (no args):
# - paths = 10,000,000
# - steps = 1000 (heavier workload; set to 252 if you want "daily")
# - uses Yahoo CSV calibration (S0/sigma/mu)
# - outputs avg-path mean/std trajectory: avg_2330_cpu.csv
# - appends benchmark row: bench_runs.csv
#
# Usage:
#   bash run_srun_cpu_omp.sh
#   bash run_srun_cpu_omp.sh -- --paths 1000000 --steps 252 --avg-path-csv avg_cpu.csv

PROJECT_ID="ACD114118"
TIME_MIN="10"
# NOTE (Taiwania2 policy):
# On this partition the CPU limit shown by Slurm is 4 CPUs per node for this job type
# (error: "CPU per node specification 32 is out of limit: 1 * 4.").
# So we default to 4 OpenMP threads for the CPU baseline.
CPUS_PER_TASK="4"
GPUS_PER_NODE="1"

SRC="mc_pricer_omp.cpp"
EXE="mc_pricer_omp"
GXX_FLAGS=(-O3 -std=c++17 -fopenmp)

# Default benchmark settings (override by passing args after `--`).
TOTAL_PATHS="10000000"
STEPS="1000"   # if you want "daily" steps, use 252
T_YEARS="1"
YAHOO_CSV="data/2330_TW.csv"
AVG_OUT="avg_2330_cpu.csv"
BENCH_OUT="cpu.csv"

RUN_ARGS=()
if [[ "${1:-}" == "--" ]]; then
  shift
  RUN_ARGS=("$@")
else
  RUN_ARGS=(
    --yahoo-csv "${YAHOO_CSV}"
    --T "${T_YEARS}"
    --steps "${STEPS}"
    --paths "${TOTAL_PATHS}"
    --avg-path-csv "${AVG_OUT}"
    --out-csv "${BENCH_OUT}"
    --append-csv
    --omp-threads "${CPUS_PER_TASK}"
  )
fi

echo "[1/3] (CPU) Compiling ${SRC} -> ${EXE} ..."
g++ "${GXX_FLAGS[@]}" "${SRC}" -o "${EXE}"

echo "[2/3] Submitting CPU job via srun (1 node, 1 task, ${CPUS_PER_TASK} CPUs)..."
# Note (Taiwania2): some partitions enforce GPU allocation even for CPU-only binaries.
# We request 1 GPU to satisfy Slurm, but the program itself is CPU/OpenMP only.
srun -N 1 --ntasks-per-node 1 -c "${CPUS_PER_TASK}" --gpus-per-node "${GPUS_PER_NODE}" \
  -A "${PROJECT_ID}" -t "${TIME_MIN}" "./${EXE}" "${RUN_ARGS[@]}"


