#!/usr/bin/env bash
# Some Taiwania2 login environments propagate `nounset` (`set -u`) via BASHOPTS,
# which makes `module load` (lmod) fail because SLURM_JOBID is unset on login nodes.
# Force-disable nounset here to make the script robust.
set +u
set +o nounset 2>/dev/null || true

set -e
set -o pipefail

# Taiwania2 (Lab 3) workflow:
# - Compile ONLY on login node (ln01.twcc.ai)
# - Submit execution to compute node using srun with GPU request
#
# Usage:
#   bash run_srun.sh
#   bash run_srun.sh -- --type european --paths 5000000 --steps 252
#
# Notes:
# - Do NOT use `set -u` (nounset) here: `module load` scripts may reference
#   SLURM_* env vars (like SLURM_JOBID) that are intentionally unset on login nodes.

PROJECT_ID="ACD114118"
TIME_MIN="10"
GPUS_PER_NODE="1"

SRC="mc_pricer.cu"
EXE="mc_pricer"
NVCC_FLAGS=(-O3 -std=c++17 -arch=sm_70)

# Default benchmark settings (override by passing args after `--`).
TOTAL_PATHS="10000000"
STEPS="1000"   # if you want "daily" steps, use 252
T_YEARS="1"
YAHOO_CSV="data/2330_TW.csv"
AVG_OUT="avg_2330_1gpu.csv"
BENCH_OUT="1gpu.csv"

# Pass everything after `--` to the executable.
RUN_ARGS=()
if [[ "${1:-}" == "--" ]]; then
  shift
  RUN_ARGS=("$@")
else
  # Default: produce mean/std trajectory (avg-path) for plotting + also write one-line benchmark CSV.
  RUN_ARGS=(
    --yahoo-csv "${YAHOO_CSV}"
    --T "${T_YEARS}"
    --steps "${STEPS}"
    --paths "${TOTAL_PATHS}"
    --avg-path-csv "${AVG_OUT}"
    --out-csv "${BENCH_OUT}"
    --append-csv
  )
fi

echo "[1/3] Loading CUDA module..."
# Ensure SLURM_JOBID exists (empty) to avoid lmod scripts tripping under nounset.
export SLURM_JOBID="${SLURM_JOBID-}"
module load cuda

echo "[2/3] Compiling ${SRC} -> ${EXE} ..."
nvcc "${NVCC_FLAGS[@]}" "${SRC}" -o "${EXE}"

echo "[3/3] Submitting job via srun (1 node, 1 task, 1 GPU)..."
srun -N 1 -n 1 --gpus-per-node "${GPUS_PER_NODE}" -A "${PROJECT_ID}" -t "${TIME_MIN}" "./${EXE}" "${RUN_ARGS[@]}"


