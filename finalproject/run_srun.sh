#!/usr/bin/env bash
set -euo pipefail

# Taiwania2 (Lab 3) workflow:
# - Compile ONLY on login node (ln01.twcc.ai)
# - Submit execution to compute node using srun with GPU request
#
# Usage:
#   bash run_srun.sh
#
# You can tweak TIME/PATHS/etc. below as needed.

PROJECT_ID="ACD114118"
TIME_MIN="1"
GPUS_PER_NODE="1"

echo "[1/3] Loading CUDA module..."
module load cuda

echo "[2/3] Compiling hello_world..."
nvcc -O2 -std=c++17 -arch=sm_70 hello_world.cu -o hello_world

echo "[3/3] Submitting job via srun (1 node, 1 task, 1 GPU)..."
srun -N 1 -n 1 --gpus-per-node "${GPUS_PER_NODE}" -A "${PROJECT_ID}" -t "${TIME_MIN}" ./hello_world


