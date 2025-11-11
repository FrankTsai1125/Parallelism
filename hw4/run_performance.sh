#!/bin/bash

# =====================================================================================
# HW4 Bitcoin Miner - GPU Performance Measurement Script
#  - Runs Nsight Systems (nsys) profiling for each testcase
#  - Captures detailed timing via /usr/bin/time -v
#  - Results are stored under performance_reports/
# =====================================================================================

set -u

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SRUN_NODES=1
SRUN_TASKS=1
SRUN_GPUS=1
SRUN_ACCOUNT="ACD114118"
SRUN_TIME=15

REPORT_DIR="performance_reports"
OUTPUT_DIR="outputs"

CASES=("case00" "case01" "case02" "case03")

printf "==========================================\n"
printf "  HW4 Bitcoin Miner - 性能量测脚本\n"
printf "==========================================\n\n"
date
printf "\n"

# -------------------------------------------------------------------------------------
# Step 1: 环境设置
# -------------------------------------------------------------------------------------
printf "${BLUE}Step 1: 加载 CUDA 环境...${NC}\n"
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
            printf "${YELLOW}[INFO]${NC} 已执行 'module load cuda'.\n"
        fi
    fi
fi

if ! command -v nvcc >/dev/null 2>&1; then
    printf "${RED}[ERROR]${NC} 找不到 nvcc，請先執行 'module load cuda' 後再嘗試。\n"
    exit 1
fi

printf "${GREEN}✓${NC} CUDA 編譯器: %s\n" "$(nvcc --version | head -n 1)"
printf "\n"

# -------------------------------------------------------------------------------------
# Step 2: 编译程序
# -------------------------------------------------------------------------------------
printf "${BLUE}Step 2: 编译程序...${NC}\n"
make clean >/dev/null 2>&1 || true
if ! make; then
    printf "${RED}[ERROR]${NC} 编译失败！\n"
    exit 1
fi
printf "${GREEN}✓${NC} 编译成功\n\n"

if [ ! -f "./hw4" ]; then
    printf "${RED}[ERROR]${NC} 找不到 hw4 可执行档！\n"
    exit 1
fi

# -------------------------------------------------------------------------------------
# Step 3: 建立输出目录
# -------------------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"
mkdir -p "$REPORT_DIR"
SUMMARY_FILE="${REPORT_DIR}/perf_summary_$(date +%Y%m%d_%H%M%S).txt"
printf "性能量测摘要: %s\n" "$SUMMARY_FILE"
printf "Performance Summary - %s\n" "$(date)" > "$SUMMARY_FILE"
printf "========================================\n\n" >> "$SUMMARY_FILE"

# -------------------------------------------------------------------------------------
# Step 4: 针对每个案例量测性能
# -------------------------------------------------------------------------------------
for case in "${CASES[@]}"; do
    printf "===========================================\n"
    printf "${CYAN}性能案例: %s${NC}\n" "$case"
    printf "===========================================\n\n"

    input_file="testcases/${case}.in"
    output_file="${OUTPUT_DIR}/${case}_perf.out"

    if [ ! -f "$input_file" ]; then
        printf "${RED}[SKIP]${NC} 输入档不存在: %s\n\n" "$input_file"
        printf "Case %s: SKIPPED (missing input file)\n" "$case" >> "$SUMMARY_FILE"
        printf "----------------------------------------\n" >> "$SUMMARY_FILE"
        continue
    fi

    perf_output="${REPORT_DIR}/${case}_profile"
    nsys_report_txt="${REPORT_DIR}/${case}_nsys.txt"
    time_output="${REPORT_DIR}/${case}_time.txt"

    printf "${YELLOW}[1/2]${NC} Nsight Systems profiling...\n"
    if command -v nsys >/dev/null 2>&1; then
        if command -v srun >/dev/null 2>&1; then
            srun -N $SRUN_NODES -n $SRUN_TASKS --gpus-per-node $SRUN_GPUS -A $SRUN_ACCOUNT -t $SRUN_TIME \
                nsys profile --stats=true --force-overwrite=true -o "$perf_output" \
                ./hw4 "$input_file" "$output_file" > "$nsys_report_txt" 2>&1
        else
            nsys profile --stats=true --force-overwrite=true -o "$perf_output" \
                ./hw4 "$input_file" "$output_file" > "$nsys_report_txt" 2>&1
        fi
        if [ $? -eq 0 ]; then
            printf "${GREEN}✓${NC} nsys 报告已生成: %s.nsys-rep\n" "$perf_output"
        else
            printf "${YELLOW}[WARNING]${NC} nsys 执行失败，详见 %s\n" "$nsys_report_txt"
        fi
    else
        printf "${YELLOW}[WARNING]${NC} 系统未安装 nsys，跳过此步骤。\n"
        echo "Case ${case}: nsys unavailable" >> "$SUMMARY_FILE"
    fi

    printf "${YELLOW}[2/2]${NC} /usr/bin/time -v 量测...\n"
    if command -v srun >/dev/null 2>&1; then
        /usr/bin/time -v srun -N $SRUN_NODES -n $SRUN_TASKS --gpus-per-node $SRUN_GPUS -A $SRUN_ACCOUNT -t $SRUN_TIME \
            ./hw4 "$input_file" "$output_file" > /dev/null 2> "$time_output"
    else
        /usr/bin/time -v ./hw4 "$input_file" "$output_file" > /dev/null 2> "$time_output"
    fi
    if [ $? -eq 0 ]; then
        printf "${GREEN}✓${NC} 时间统计已写入: %s\n" "$time_output"
    else
        printf "${RED}[ERROR]${NC} /usr/bin/time 执行失败。\n"
    fi

    printf "\n关键信息 (来自 /usr/bin/time):\n"
    if [ -f "$time_output" ]; then
        grep -E "Elapsed|Maximum resident|Percent of CPU" "$time_output"
    fi

    {
        printf "Case: %s\n" "$case"
        printf "  nsys report : %s.nsys-rep\n" "$perf_output"
        printf "  nsys stats  : %s\n" "$nsys_report_txt"
        printf "  time report : %s\n" "$time_output"
        if [ -f "$time_output" ]; then
            grep -E "Elapsed|Maximum resident|Percent of CPU" "$time_output" | sed 's/^/    /'
        fi
        printf '%s\n\n' '----------------------------------------'
    } >> "$SUMMARY_FILE"

    printf "\n"
done

printf "==========================================\n"
printf "性能量测完成。摘要输出: %s\n" "$SUMMARY_FILE"
printf "报告目录: %s/\n" "$REPORT_DIR"
printf "==========================================\n\n"
date
printf "\n"
