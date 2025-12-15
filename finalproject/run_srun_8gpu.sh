#!/usr/bin/env bash
set +u
set +o nounset 2>/dev/null || true
set -e
set -o pipefail

# Taiwania2 (Lab 3) multi-GPU, single-node scaling:
# - 1 node has 8x V100
# - launch 8 tasks, each task gets 1 GPU (SLURM_LOCALID selects device)
#
# Usage:
#   bash run_srun_8gpu.sh
#   bash run_srun_8gpu.sh -- --type basket --assets 16 --rho 0.3 --paths 2000000 --steps 252

PROJECT_ID="ACD114118"
TIME_MIN="1"
GPUS_PER_NODE="8"
NTASKS_PER_NODE="8"

SRC="mc_pricer.cu"
EXE="mc_pricer"
NVCC_FLAGS=(-O3 -std=c++17 -arch=sm_70)

RUN_ARGS=()
if [[ "${1:-}" == "--" ]]; then
  shift
  RUN_ARGS=("$@")
fi

echo "[1/3] Loading CUDA module..."
export SLURM_JOBID="${SLURM_JOBID-}"
module load cuda

echo "[2/3] Compiling ${SRC} -> ${EXE} ..."
nvcc "${NVCC_FLAGS[@]}" "${SRC}" -o "${EXE}"

echo "[3/3] Submitting multi-GPU job via srun (1 node, 8 tasks, 8 GPUs)..."
srun -N 1 --ntasks-per-node "${NTASKS_PER_NODE}" --gpus-per-node "${GPUS_PER_NODE}" \
  -A "${PROJECT_ID}" -t "${TIME_MIN}" "./${EXE}" "${RUN_ARGS[@]}"


