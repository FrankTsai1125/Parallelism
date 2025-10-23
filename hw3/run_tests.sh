#!/bin/bash

# Improved test script for hw3
# Usage: ./run_tests.sh [test_numbers...]
# Example: ./run_tests.sh 00 01 02 03
# Or just: ./run_tests.sh (runs all)

echo "==================================="
echo "HW3 Mandelbulb Test Script"
echo "==================================="

# Default to all tests if no arguments
if [ $# -eq 0 ]; then
    TESTS="00 01 02 03"
else
    TESTS="$@"
fi

# Check if hw3 exists
if [ ! -f "./hw3" ]; then
    echo "❌ hw3 executable not found! Please compile first."
    echo "   Run: make"
    exit 1
fi

# Run each test
passed=0
failed=0

for test_num in $TESTS; do
    echo ""
    echo "==================================="
    echo "Test Case $test_num"
    echo "==================================="
    
    test_file="testcases/${test_num}.txt"
    
    if [ ! -f "$test_file" ]; then
        echo "❌ Test file $test_file not found!"
        ((failed++))
        continue
    fi
    
    # Parse parameters
    pos=$(grep "pos=" $test_file | cut -d'=' -f2)
    tarpos=$(grep "tarpos=" $test_file | cut -d'=' -f2)
    width=$(grep "width=" $test_file | cut -d'=' -f2)
    height=$(grep "height=" $test_file | cut -d'=' -f2)
    
    pos_x=$(echo $pos | awk '{print $1}')
    pos_y=$(echo $pos | awk '{print $2}')
    pos_z=$(echo $pos | awk '{print $3}')
    tar_x=$(echo $tarpos | awk '{print $1}')
    tar_y=$(echo $tarpos | awk '{print $2}')
    tar_z=$(echo $tarpos | awk '{print $3}')
    
    output_file="output_${test_num}.png"
    
    echo "  Camera: ($pos_x, $pos_y, $pos_z)"
    echo "  Target: ($tar_x, $tar_y, $tar_z)"
    echo "  Resolution: ${width}x${height}"
    
    # Run test with timing
    start=$(date +%s.%N)
    srun -N 1 -n 1 --gpus-per-node 1 -A ACD114118 -t 3 \
        ./hw3 $pos_x $pos_y $pos_z $tar_x $tar_y $tar_z $width $height $output_file 2>&1 | grep -v "WARNING\|INFO"
    
    exit_code=$?
    end=$(date +%s.%N)
    elapsed=$(echo "$end - $start" | bc)
    
    if [ $exit_code -ne 0 ] || [ ! -f "$output_file" ]; then
        echo "❌ Test $test_num FAILED"
        ((failed++))
    else
        filesize=$(ls -lh $output_file | awk '{print $5}')
        echo "✅ Test $test_num PASSED"
        echo "   Output: $output_file ($filesize)"
        echo "   Time: ${elapsed}s"
        ((passed++))
    fi
done

# Summary
echo ""
echo "==================================="
echo "Test Summary"
echo "==================================="
echo "Passed: $passed"
echo "Failed: $failed"
echo ""

if [ $failed -eq 0 ]; then
    echo "🎉 All tests passed!"
    exit 0
else
    echo "⚠️  Some tests failed"
    exit 1
fi

