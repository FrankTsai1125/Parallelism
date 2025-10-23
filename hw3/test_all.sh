#!/bin/bash

# Complete test script for hw3 (cases 00-08)
# Usage: ./test_all.sh [test_numbers...]
# Example: ./test_all.sh 00 01 02
# Or: ./test_all.sh (runs all 00-08)

echo "==================================="
echo "HW3 Complete Test Script (00-08)"
echo "==================================="

# Default to all tests if no arguments
if [ $# -eq 0 ]; then
    TESTS="00 01 02 03 04 05 06 07 08"
else
    TESTS="$@"
fi

# Check if hw3 exists
if [ ! -f "./hw3" ]; then
    echo "❌ hw3 executable not found! Compiling..."
    make
    if [ $? -ne 0 ]; then
        echo "❌ Compilation failed!"
        exit 1
    fi
fi

# Run each test
passed=0
failed=0
timeout_count=0

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
    timelimit=$(grep "timelimit=" $test_file | cut -d'=' -f2)
    
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
    echo "  Time limit: ${timelimit}s"
    
    # Calculate srun time limit (add 3 seconds buffer)
    srun_time=$((timelimit + 3))
    
    # Run test with timing
    start=$(date +%s.%N)
    
    # Run with timeout
    timeout ${timelimit}s srun -N 1 -n 1 --gpus-per-node 1 -A ACD114118 -t ${srun_time} \
        ./hw3 $pos_x $pos_y $pos_z $tar_x $tar_y $tar_z $width $height $output_file > /dev/null 2>&1
    
    exit_code=$?
    end=$(date +%s.%N)
    elapsed=$(echo "$end - $start" | bc)
    
    # Check result
    if [ $exit_code -eq 124 ]; then
        # Timeout
        echo "⏱️  Test $test_num TIMEOUT (>${timelimit}s)"
        echo "   Actual time: ${elapsed}s"
        ((failed++))
        ((timeout_count++))
    elif [ $exit_code -ne 0 ] || [ ! -f "$output_file" ]; then
        echo "❌ Test $test_num FAILED (exit code: $exit_code)"
        ((failed++))
    else
        filesize=$(ls -lh $output_file | awk '{print $5}')
        
        # Check if within time limit
        time_ok=$(echo "$elapsed <= $timelimit" | bc)
        if [ $time_ok -eq 1 ]; then
            echo "✅ Test $test_num PASSED"
            echo "   Output: $output_file ($filesize)"
            echo "   Time: ${elapsed}s / ${timelimit}s"
            ((passed++))
        else
            echo "⏱️  Test $test_num TIMEOUT"
            echo "   Output: $output_file ($filesize)"
            echo "   Time: ${elapsed}s / ${timelimit}s (exceeded)"
            ((failed++))
            ((timeout_count++))
        fi
    fi
done

# Summary
echo ""
echo "==================================="
echo "Test Summary"
echo "==================================="
echo "Passed: $passed"
echo "Failed: $failed"
if [ $timeout_count -gt 0 ]; then
    echo "Timeout: $timeout_count"
fi
echo ""

if [ $failed -eq 0 ]; then
    echo "🎉 All tests passed!"
    exit 0
else
    if [ $timeout_count -gt 0 ]; then
        echo "⚠️  $timeout_count test(s) exceeded time limit - optimization needed!"
    fi
    echo "❌ $failed test(s) failed"
    exit 1
fi

