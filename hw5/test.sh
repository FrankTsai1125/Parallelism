#!/bin/bash

# Check if running inside Slurm allocation
if [ -z "$SLURM_JOB_ID" ]; then
    echo "Detected execution on Login Node."
    echo "Submitting entire test suite to GPU node via srun (waiting for allocation)..."
    exec srun -p normal -N 1 -n 1 --gres=gpu:1 --unbuffered "$0" "$@"
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration
VERBOSE=0
SHOW_OUTPUT=0
PROFILE_MODE=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -o|--show-output)
            SHOW_OUTPUT=1
            shift
            ;;
        -p|--profile)
            PROFILE_MODE=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -v, --verbose       Show detailed output"
            echo "  -o, --show-output   Show actual vs expected output"
            echo "  -p, --profile       Run profiling on b1024 only (SKIP checking)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BOLD}======================================${NC}"
echo "HW5 N-body Simulation Test Suite"
echo "Running on Host: $(hostname)"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo -e "${BOLD}======================================${NC}"

# Compile
echo -e "${BOLD}Compiling hw5...${NC}"
make clean >/dev/null 2>&1
make hw5 2>&1 | head -10
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Compilation failed!${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Compilation successful!${NC}"
echo ""

mkdir -p outputs
mkdir -p reports

# Correctness Test Mode
if [ $PROFILE_MODE -eq 0 ]; then
    declare -A TEST_CASES
    TEST_CASES=(
        ["b20"]=60 ["b30"]=90 ["b40"]=120 ["b50"]=120
        ["b60"]=150 ["b70"]=150 ["b80"]=180 ["b90"]=180
        ["b100"]=200 ["b200"]=300 ["b512"]=600 ["b1024"]=900
    )
    TEST_ORDER=(b20 b30 b40 b50 b60 b70 b80 b90 b100 b200 b512 b1024)
    
    echo -e "${BOLD}Running Correctness Tests...${NC}"
    
    # Debug GPU visibility
    echo "Checking GPU visibility..."
    rocm-smi
    echo "Current Environment Variables related to GPU:"
    env | grep -E "ROCR|HIP|CUDA|SLURM"
    
    # Try to fix "No GPU detected" by explicitly setting device if not set
    if [ -z "$ROCR_VISIBLE_DEVICES" ]; then
        echo "ROCR_VISIBLE_DEVICES not set. Setting to 0..."
        export ROCR_VISIBLE_DEVICES=0
    fi
    if [ -z "$HIP_VISIBLE_DEVICES" ]; then
        echo "HIP_VISIBLE_DEVICES not set. Setting to 0..."
        export HIP_VISIBLE_DEVICES=0
    fi

    printf "%-10s %-10s %-10s %-s\n" "Test" "Time(s)" "Status" "Details"
    echo "--------------------------------------"
    
    passed=0; failed=0; total=0
    
    for test in "${TEST_ORDER[@]}"; do
        total=$((total + 1))
        time_limit=${TEST_CASES[$test]}
        input="testcases/${test}.in"
        output="outputs/${test}.out"
        expect="testcases/${test}.out"
        
        if [ ! -f "$input" ]; then continue; fi
        
        start=$(date +%s.%N)
        timeout ${time_limit} ./hw5 "$input" "$output"
        ret=$?
        end=$(date +%s.%N)
        elapsed=$(echo "$end - $start" | bc)
        
        if [ $ret -eq 124 ]; then
            printf "%-10s %-10.2f ${RED}%-10s${NC} %s\n" "$test" "$elapsed" "[TIMEOUT]" "> ${time_limit}s"
            failed=$((failed + 1))
        elif [ $ret -ne 0 ]; then
            printf "%-10s %-10.2f ${RED}%-10s${NC} %s\n" "$test" "$elapsed" "[CRASH]" "Code $ret"
            failed=$((failed + 1))
        else
            # Validate
            python3 validate.py "$expect" "$output" >/dev/null 2>&1
            if [ $? -eq 0 ]; then
                printf "%-10s %-10.2f ${GREEN}%-10s${NC} %s\n" "$test" "$elapsed" "[PASS]" "✓"
                passed=$((passed + 1))
            else
                printf "%-10s %-10.2f ${RED}%-10s${NC} %s\n" "$test" "$elapsed" "[FAIL]" "Diff mismatch"
                failed=$((failed + 1))
            fi
        fi
    done
    
    echo "======================================"
    echo -e "Passed: ${GREEN}$passed${NC} / $total"
    if [ $total -gt 0 ]; then
        score=$(echo "scale=2; $passed * 100 / $total" | bc)
        echo -e "Score: ${BOLD}${score}%${NC}"
    fi

# Profiling Mode
else
    echo -e "${BOLD}Running Profiling Mode ...${NC}"
    TEST_ORDER=( b70 b80 b90 b100 b200 b512 b1024)
    
    for test in "${TEST_ORDER[@]}"; do
        echo -e "\n${BOLD}Profiling Test Case: $test${NC}"
        input="testcases/${test}.in"
        output="outputs/${test}.out"
        report_dir="reports/${test}_profile"
        mkdir -p "$report_dir"
        
        # Cleanup old files
        rm -f "$report_dir/gpu_usage.csv" "$report_dir"/*.csv "$report_dir"/*.stats.csv
        
        echo "  1. Starting GPU Monitor (rocm-smi)..."
        # Run rocm-smi in a loop to capture usage every ~0.1s
        # We capture a single snapshot (--showuse --showmemuse) and append to csv
        rm -f "$report_dir/gpu_usage.csv"
        (
            echo "timestamp,gpu_use,mem_use" > "$report_dir/gpu_usage.csv"
            while true; do
                # Capture timestamp and rocm-smi output (filtering for values)
                # This is a rough approximation as rocm-smi output format varies
                # A simpler way is just to dump the raw output
                ts=$(date +%s.%N)
                # Filter out warnings by redirecting stderr to /dev/null
                usage=$(rocm-smi --showuse --showmemuse --csv 2>/dev/null | tail -n +2)
                if [ ! -z "$usage" ]; then
                    echo "$ts,$usage" >> "$report_dir/gpu_usage.csv"
                fi
                sleep 0.1
            done
        ) &
        MON_PID=$!
        
        echo "  2. Running Profiler..."
        start=$(date +%s.%N)
        
        # Try using standard rocprof --stats first (Legacy but reliable for stats)
        # This generates results.stats.csv directly
        rocprof --stats -o "$report_dir/results.stats.csv" ./hw5 "$input" "$output" >/dev/null 2>&1
        
        # Check if it worked
        if [ ! -f "$report_dir/results.stats.csv" ]; then
            echo "  Legacy rocprof failed or didn't output stats. Trying rocprofv2..."
            
            # Fallback to rocprofv2
            # We use --plugin file to ensure CSV output
            rocprofv2 --kernel-trace --hip-trace --plugin file -d "$report_dir" -o "results" ./hw5 "$input" "$output" >/dev/null 2>&1
            
            # Check for rocprofv2 output (results_*.csv or kernel_trace_*.csv)
            trace_csv=$(ls -t "$report_dir"/results_*.csv "$report_dir"/kernel_trace_*.csv 2>/dev/null | head -n 1)
            
            if [ -n "$trace_csv" ]; then
                echo "  Parsing trace file: $trace_csv"
                python3 parse_trace.py "$trace_csv" "$report_dir/results.stats.csv"
            fi
        fi
        
        end=$(date +%s.%N)
        elapsed=$(echo "$end - $start" | bc)
        
        if [ -f "$report_dir/results.stats.csv" ]; then
            echo -e "  Report generated: ${report_dir}/results.stats.csv"
            echo "  Top 5 Kernels:"
            head -n 6 "$report_dir/results.stats.csv" | column -t -s, 2>/dev/null || head -n 6 "$report_dir/results.stats.csv"
        else
            echo -e "  ${YELLOW}Warning: No statistics report generated.${NC}"
            echo "  Contents of $report_dir:"
            ls -F "$report_dir"
        fi
        
        kill $MON_PID 2>/dev/null
        
        echo -e "  ${GREEN}Complete!${NC} Time: ${elapsed}s"
        echo -e "  Report: ${report_dir}/results.stats.csv"
        elapsed=$(echo "$end - $start" | bc)
        
        kill $MON_PID 2>/dev/null
        
        echo -e "  ${GREEN}Complete!${NC} Time: ${elapsed}s"
        echo -e "  Report: ${report_dir}/results.stats.csv"
    done
    
    echo -e "\n${BOLD}All Profiling Completed!${NC}"
fi
