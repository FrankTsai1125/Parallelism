#!/bin/bash

echo "==================================="
echo "HW3 Mandelbulb Local Test Script"
echo "==================================="

NVPROF_METRICS="achieved_occupancy,sm_efficiency,inst_executed,flop_count_sp,dram_read_throughput,dram_write_throughput,branch_efficiency"
NVPROF_KERNEL_FILTER="render_kernel"
PROFILE_DIR="profiles"

echo ""
echo "Loading CUDA module..."
module load cuda

echo ""
echo "Compiling hw3.cu..."
make clean
make

if [ $? -ne 0 ]; then
    echo "❌ Compilation failed!"
    exit 1
fi

echo "✅ Compilation successful!"

mkdir -p "$PROFILE_DIR"

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

    echo "Parameters:"
    echo "  Camera: ($pos_x, $pos_y, $pos_z)"
    echo "  Target: ($tar_x, $tar_y, $tar_z)"
    echo "  Size: ${width}x${height}"
    echo "  Time limit: ${timelimit}s"

    echo ""
    echo "Running test..."
    start_time=$(date +%s)

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

    if [ -f "$expected_file" ]; then
        echo "   Expected: $expected_file (exists)"
    fi

    local profile_metrics_file="${PROFILE_DIR}/nvprof_${test_num}_metrics.csv"
    local profile_trace_file="${PROFILE_DIR}/nvprof_${test_num}_trace.csv"

    echo ""
    echo "Profiling (metrics) -> ${profile_metrics_file}"
    nvprof --csv \
           --log-file "${profile_metrics_file}" \
           --kernels "${NVPROF_KERNEL_FILTER}" \
           --metrics "${NVPROF_METRICS}" \
           ./hw3 $pos_x $pos_y $pos_z $tar_x $tar_y $tar_z $width $height $output_file > /dev/null 2>&1

    if [ $? -ne 0 ]; then
        echo "⚠️  nvprof metrics profiling failed for test ${test_num}"
    else
        echo "   ✅ Metrics report captured"
    fi

    echo "Profiling (GPU trace) -> ${profile_trace_file}"
    nvprof --csv \
           --log-file "${profile_trace_file}" \
           --kernels "${NVPROF_KERNEL_FILTER}" \
           --print-gpu-trace \
           ./hw3 $pos_x $pos_y $pos_z $tar_x $tar_y $tar_z $width $height $output_file > /dev/null 2>&1

    if [ $? -ne 0 ]; then
        echo "⚠️  nvprof GPU trace profiling failed for test ${test_num}"
    else
        echo "   ✅ GPU trace captured"
    fi

    return 0
}

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

echo ""
echo "==================================="
echo "Test Summary"
echo "==================================="
