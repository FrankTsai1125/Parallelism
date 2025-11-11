#!/bin/bash

# HW4 Bitcoin Miner - Functional Test Runner
#  - Compile once
#  - Run each testcase (via srun when available)
#  - Validate correctness
#  - Record wall-clock time per case

set -u
set -o pipefail

CASES=("case00" "case01" "case02" "case03")

SRUN_NODES=1
SRUN_TASKS=1
SRUN_GPUS=1
SRUN_ACCOUNT="ACD114118"
SRUN_TIME=10

OUTPUT_DIR="outputs"
REPORT_DIR="test_reports"
SUMMARY_FILE="${REPORT_DIR}/summary_$(date +%Y%m%d_%H%M%S).txt"
CSV_FILE="${REPORT_DIR}/results_$(date +%Y%m%d_%H%M%S).csv"

printf "==========================================\n"
printf "  HW4 Bitcoin Miner - Functional Tests\n"
printf "==========================================\n\n"
date
printf "\n"

# ------------------------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------------------------
printf "[Step 1] 使用当前 CUDA 环境。\n"
if ! command -v nvcc >/dev/null 2>&1; then
    if ! command -v module >/dev/null 2>&1; then
        for init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash; do
            if [ -f "$init" ]; then
                # shellcheck disable=SC1091
                source "$init"
                break
            fi
        done
    fi

    if command -v module >/dev/null 2>&1; then
        if module load cuda >/dev/null 2>&1; then
            printf "[INFO] 已執行 'module load cuda'.\n"
        fi
    fi
fi

if ! command -v nvcc >/dev/null 2>&1; then
    printf "[ERROR] 找不到 nvcc，請先執行 'module load cuda' 後再嘗試。\n"
    exit 1
fi

printf "[OK] CUDA 編譯器: %s\n\n" "$(nvcc --version | head -n 1)"

printf "[Step 2] 编译程序...\n"
make clean >/dev/null 2>&1 || true
if ! make; then
    printf "[ERROR] 编译失败！\n"
    exit 1
fi
printf "[OK] 编译成功\n\n"

if [ ! -f ./hw4 ] || [ ! -f ./validation ]; then
    printf "[ERROR] 缺少 hw4 或 validation 可执行文件！\n"
    exit 1
fi

# ------------------------------------------------------------------------------
# Prepare output directories
# ------------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR" "$REPORT_DIR"
printf "[Step 3] 结果目录: %s\n" "$OUTPUT_DIR"
printf "          报告目录: %s\n\n" "$REPORT_DIR"

printf "Functional Test Summary - %s\n" "$(date)" > "$SUMMARY_FILE"
printf "Case,Status,Elapsed(s)\n" > "$CSV_FILE"

TOTAL_TIME=0

for case in "${CASES[@]}"; do
    input="testcases/${case}.in"
    output="${OUTPUT_DIR}/${case}.out"
    validation_log="${REPORT_DIR}/${case}_validation.txt"

    printf '%s\n' '-------------------------------------------'
    printf "Case: %s\n" "$case"

    if [ ! -f "$input" ]; then
        printf "  [SKIP] 输入文件不存在: %s\n" "$input"
        printf "Case %s: SKIP (missing input)\n" "$case" >> "$SUMMARY_FILE"
        printf "%s,SKIP,0\n" "$case" >> "$CSV_FILE"
        continue
    fi

    start_ns=$(date +%s%N)
    if command -v srun >/dev/null 2>&1; then
        srun -N $SRUN_NODES -n $SRUN_TASKS --gpus-per-node $SRUN_GPUS -A $SRUN_ACCOUNT -t $SRUN_TIME \
            ./hw4 "$input" "$output"
        run_status=$?
    else
        ./hw4 "$input" "$output"
        run_status=$?
    fi
    end_ns=$(date +%s%N)

    elapsed=$(echo "scale=3; ($end_ns - $start_ns)/1000000000" | bc)

    if [ $run_status -ne 0 ]; then
        printf "  [FAIL] 程序执行失败 (exit %d)\n" $run_status
        printf "Case %s: RUN_FAIL (%.3fs)\n" "$case" "$elapsed" >> "$SUMMARY_FILE"
        printf "%s,RUN_FAIL,%.3f\n" "$case" "$elapsed" >> "$CSV_FILE"
        TOTAL_TIME=$(echo "$TOTAL_TIME + $elapsed" | bc)
        continue
    fi

    ./validation "$input" "$output" > "$validation_log" 2>&1
    val_status=$?

    if [ $val_status -ne 0 ]; then
        printf "  [FAIL] 验证失败，详见 %s\n" "$validation_log"
        printf "Case %s: VALIDATION_FAIL (%.3fs)\n" "$case" "$elapsed" >> "$SUMMARY_FILE"
        printf "%s,VALIDATION_FAIL,%.3f\n" "$case" "$elapsed" >> "$CSV_FILE"
    else
        printf "  [PASS] 验证通过 (%.3fs)\n" "$elapsed"
        printf "Case %s: PASS (%.3fs)\n" "$case" "$elapsed" >> "$SUMMARY_FILE"
        printf "%s,PASS,%.3f\n" "$case" "$elapsed" >> "$CSV_FILE"
    fi

    TOTAL_TIME=$(echo "$TOTAL_TIME + $elapsed" | bc)
    printf '%s\n\n' '-------------------------------------------'
done

printf "所有案例处理完毕。\n"
printf "总时间: %.3f 秒\n" "$TOTAL_TIME"
printf "\n概要已写入:\n  %s\n  %s\n\n" "$SUMMARY_FILE" "$CSV_FILE"
date

