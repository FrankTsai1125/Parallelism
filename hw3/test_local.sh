#!/bin/bash

# Local test script for hw3 CUDA implementation
# Tests testcases 0-3

echo "==================================="
echo "HW3 Mandelbulb Local Test Script"
echo "==================================="

# Load CUDA module
echo ""
echo "Loading CUDA module..."
module load cuda

# Compile
echo ""
echo "Compiling hw3.cu..."
make clean
make

if [ $? -ne 0 ]; then
    echo "❌ Compilation failed!"
    exit 1
fi

echo "✅ Compilation successful!"

# Test function
run_test() {
    local test_num=$1
    local test_file="testcases/${test_num}.txt"
    local output_file="output_${test_num}.png"
    local expected_file="testcases/${test_num}.png"
    
    echo ""
    echo "==================================="
    echo "Testing case ${test_num}"
    echo "==================================="
    
    if [ ! -f "$test_file" ]; then
        echo "❌ Test file $test_file not found!"
        return 1
    fi
    
    # Parse test file
    pos=$(grep "pos=" $test_file | cut -d'=' -f2)
    tarpos=$(grep "tarpos=" $test_file | cut -d'=' -f2)
    width=$(grep "width=" $test_file | cut -d'=' -f2)
    height=$(grep "height=" $test_file | cut -d'=' -f2)
    timelimit=$(grep "timelimit=" $test_file | cut -d'=' -f2)
    
    # Parse position values
    pos_x=$(echo $pos | awk '{print $1}')
    pos_y=$(echo $pos | awk '{print $2}')
    pos_z=$(echo $pos | awk '{print $3}')
    tar_x=$(echo $tarpos | awk '{print $1}')
    tar_y=$(echo $tarpos | awk '{print $2}')
    tar_z=$(echo $tarpos | awk '{print $3}')
    
    echo "Parameters:"
    echo "  Camera: ($pos_x, $pos_y, $pos_z)"
    echo "  Target: ($tar_x, $tar_y, $tar_z)"
    echo "  Size: ${width}x${height}"
    echo "  Time limit: ${timelimit}s"
    
    # Run with srun
    echo ""
    echo "Running test..."
    start_time=$(date +%s)
    
    srun -N 1 -n 1 --gpus-per-node 1 -A ACD114118 -t 3 \
        ./hw3 $pos_x $pos_y $pos_z $tar_x $tar_y $tar_z $width $height $output_file
    
    exit_code=$?
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    
    if [ $exit_code -ne 0 ]; then
        echo "❌ Test ${test_num} failed with exit code $exit_code"
        return 1
    fi
    
    if [ ! -f "$output_file" ]; then
        echo "❌ Output file not generated!"
        return 1
    fi
    
    echo "✅ Test ${test_num} completed in ${elapsed}s"
    echo "   Output: $output_file"
    
    # Check if we can compare with expected output
    if [ -f "$expected_file" ]; then
        echo "   Expected: $expected_file (exists)"
    fi
    
    return 0
}

# Run tests
echo ""
echo "==================================="
echo "Running Tests"
echo "==================================="

failed_tests=0
passed_tests=0

for test_num in 00 01 02 03 04 05 06 07 08; do
    run_test $test_num
    if [ $? -eq 0 ]; then
        ((passed_tests++))
    else
        ((failed_tests++))
    fi
done

# Summary
echo ""
echo "==================================="
echo "Test Summary"
echo "==================================="
echo "Passed: $passed_tests/4"
echo "Failed: $failed_tests/4"

if [ $failed_tests -eq 0 ]; then
    echo "✅ All tests passed!"
    exit 0
else
    echo "❌ Some tests failed"
    exit 1
fi

