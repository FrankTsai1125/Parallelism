#!/bin/bash

# Check if running inside Slurm allocation
if [ -z "$SLURM_JOB_ID" ]; then
    echo "Detected execution on Login Node."
    echo "Submitting to GPU node via srun..."
    exec srun -p normal -N 1 -n 1 --gres=gpu:1 --unbuffered "$0" "$@"
fi

# Compile
echo "Compiling hw5..."
hipcc -std=c++17 -O3 -fopenmp --offload-arch=gfx908 -o hw5 hw5.cpp
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi
echo "Compilation successful."

# Test Cases
TEST_CASES=(b20 b30 b40 b50 b60 b70 b80 b90 b100 b200 b512 b1024)

echo "======================================"
printf "%-10s %-10s\n" "Test" "Time(s)"
echo "--------------------------------------"

total_time=0

for test in "${TEST_CASES[@]}"; do
    input="testcases/${test}.in"
    output="outputs/${test}.out"
    
    if [ ! -f "$input" ]; then continue; fi
    
    # Measure time
    start=$(date +%s.%N)
    ./hw5 "$input" "$output"
    ret=$?
    end=$(date +%s.%N)
    elapsed=$(echo "$end - $start" | bc)
    
    # Accumulate total time
    total_time=$(echo "$total_time + $elapsed" | bc)
    
    if [ $ret -ne 0 ]; then
        printf "%-10s %-10.2f [CRASH] Code %d\n" "$test" "$elapsed" "$ret"
    else
        printf "%-10s %-10.2f [DONE]\n" "$test" "$elapsed"
    fi
done
echo "======================================"
echo "Total Execution Time: ${total_time}s"
echo "======================================"
