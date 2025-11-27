#!/bin/bash

# Configuration
TEST_CASES=(b20 b30 b40 b50 b60 b70 b80 b90 b100 b200 b512 b1024)
ROOT_DIR=$(pwd)
HW5_BIN="$ROOT_DIR/hw5"
REPORT_DIR="$ROOT_DIR/final_reports"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "=========================================="
echo "Generating ROCm Profiling Reports"
echo "=========================================="

# 1. Compile
echo "Compiling..."
make clean >/dev/null 2>&1
make hw5 >/dev/null 2>&1
if [ ! -f "$HW5_BIN" ]; then
    echo -e "${RED}Error: Compilation failed.${NC}"
    exit 1
fi

# 2. Prepare Output Directory
rm -rf "$REPORT_DIR"
mkdir -p "$REPORT_DIR"

# 3. Loop through test cases
for test in "${TEST_CASES[@]}"; do
    echo -n "Profiling $test ... "
    
    INPUT_FILE="$ROOT_DIR/testcases/${test}.in"
    OUTPUT_FILE="$ROOT_DIR/outputs/${test}.out"
    
    # Create a temp directory for this run to avoid file clutter
    TEMP_RUN_DIR="$REPORT_DIR/temp_${test}"
    mkdir -p "$TEMP_RUN_DIR"
    cd "$TEMP_RUN_DIR"
    
    # Run rocprof via srun to ensure it runs on a GPU node
    # We use --stats to get the summary CSV
    echo "    Submitting job to Slurm..."
    srun -p normal -N 1 -n 1 --gres=gpu:1 rocprof --stats "$HW5_BIN" "$INPUT_FILE" "$OUTPUT_FILE" > run.log 2>&1
    
    # Check if results.stats.csv exists
    if [ -f "results.stats.csv" ]; then
        # Copy and rename to final report dir
        cp "results.stats.csv" "$REPORT_DIR/${test}_stats.csv"
        
        # Optional: Extract Top 3 hotspots for quick view
        echo -e "${GREEN}Done${NC}"
        echo "  Top Kernel/Memcpy:"
        # Skip header, sort by TotalDuration (usually col 4 or 5 depending on version), head 1
        # But rocprof stats format: "Name","Calls","TotalDurationNs","AverageNs","Percentage"
        # We use simple head because it's usually sorted by percentage
        cat "results.stats.csv" | head -n 4 | sed 's/^/    /'
    else
        echo -e "${RED}Failed${NC}"
        echo "  Check log: $TEMP_RUN_DIR/run.log"
        # Cat the log if failed to see why
        cat run.log | tail -n 5
    fi
    
    # Cleanup temp dir
    cd "$ROOT_DIR"
    # rm -rf "$TEMP_RUN_DIR" # Keep for debugging if needed
done

echo "=========================================="
echo "All reports saved to: $REPORT_DIR"
echo "You can view a report using: cat final_reports/b1024_stats.csv"

