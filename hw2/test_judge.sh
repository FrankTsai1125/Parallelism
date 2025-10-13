#!/bin/bash

cd /home/p13922006/Parallelism/hw2

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║              HW2 SIFT - Local Judge Testing                       ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

# Load environment
bash -c 'source scripts/env.sh 2>/dev/null'

echo "📦 Compiling..."
bash -c 'source scripts/env.sh 2>/dev/null; make clean > /dev/null 2>&1; make'

if [ $? -ne 0 ]; then
    echo "❌ Compilation failed!"
    exit 1
fi
echo "✓ Compilation successful"
echo ""

mkdir -p results

echo "🧪 Testing with Judge System Parameters..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Judge System Specs:"
echo "  • -N : 1~3 nodes"
echo "  • -n : 1~36 processes"
echo "  • -c : 6 cores per process"
echo "  • Time limit: 45 seconds per test case"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

passed=0
total=0

# Function to run a test
run_test() {
    local testnum=$1
    local nodes=$2
    local procs=$3
    local cores=$4
    local timelimit=$5
    local testfile="testcases/$(printf "%02d" $testnum).jpg"
    local outfile="results/$(printf "%02d" $testnum).jpg"
    local txtfile="results/$(printf "%02d" $testnum).txt"
    local goldenfile="goldens/$(printf "%02d" $testnum).txt"
    local goldenjpg="goldens/$(printf "%02d" $testnum).jpg"
    
    if [ ! -f "$testfile" ]; then
        echo "Test $testnum: ⊘ SKIP (file not found)"
        return
    fi
    
    total=$((total + 1))
    
    printf "Test %02d (N=%d n=%d c=%d t=%s): " "$testnum" "$nodes" "$procs" "$cores" "$timelimit"
    
    # Run with srun (no bash timeout, let srun handle it)
    start_time=$(date +%s.%N)
    
    # Set OMP_NUM_THREADS based on cores
    # Use srun's time limit, not bash timeout (hw2-judge behavior)
    result=$(bash -c "source scripts/env.sh 2>/dev/null; export OMP_NUM_THREADS=$cores; srun -A ACD114118 -N $nodes -n $procs -c $cores --time=00:01:00 ./hw2 $testfile $outfile $txtfile 2>&1")
    exit_code=$?
    
    end_time=$(date +%s.%N)
    elapsed=$(echo "$end_time - $start_time" | bc)
    
    # Extract execution time from output
    exec_time=$(echo "$result" | grep "Execution time" | awk '{print $3}')
    
    # Check if execution time exceeds limit (45s = 45000ms)
    if [ -n "$exec_time" ] && [ $(echo "$exec_time > 45000" | bc 2>/dev/null) -eq 1 ]; then
        echo "❌ TIMEOUT (exec: ${exec_time} ms > 45000 ms)"
    elif [ $exit_code -ne 0 ]; then
        echo "❌ ERROR (exit: $exit_code)"
    elif [ ! -f "$txtfile" ]; then
        echo "❌ NO OUTPUT"
    else
        # Validate output (use Python validator as per PDF requirement)
        if [ -f "$goldenfile" ]; then
            validation=$(bash -c "source scripts/env.sh 2>/dev/null; python3 validate.py $txtfile $goldenfile $outfile $goldenjpg 2>&1")
            
            # Check if validation passed
            if echo "$validation" | grep -q "Pass"; then
                printf "✓ PASS (total: %.2fs, exec: %s ms)\n" $elapsed "$exec_time"
                passed=$((passed + 1))
            else
                echo "❌ WRONG OUTPUT"
            fi
        else
            printf "⚠ DONE (total: %.2fs, exec: %s ms) [no golden]\n" $elapsed "$exec_time"
        fi
    fi
}

# Test cases with different configurations
# Format: run_test <test_num> <nodes> <procs> <cores> <timeout>
# Note: hw2-judge uses 45s time limit per test case

echo "【Small Test Cases】"
run_test 1  1  1  6  "45s"
run_test 2  1  1  6  "45s"
run_test 3  1  1  6  "45s"
echo ""

echo "【Medium Test Cases】"
run_test 4  1  2  6  "45s"
run_test 5  1  2  6  "45s"
run_test 6  1  3  6  "45s"
echo ""

echo "【Large Test Cases】"
run_test 7  2  4  6  "45s"
run_test 8  2  4  6  "45s"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Results: $passed/$total tests passed"
echo ""

if [ $passed -eq $total ]; then
    echo "🎉 All tests passed! Ready for submission!"
elif [ $passed -gt 0 ]; then
    echo "⚠️  Some tests passed. Consider optimizations for failed tests."
else
    echo "❌ No tests passed. Check implementation."
fi

echo ""
echo "💡 Tips:"
echo "  • Judge system: 45s execution time limit (excludes queue time)"
echo "  • Resource range: -N 1~3, -n 1~36, -c 6"
echo "  • This script now measures execution time like hw2-judge"
echo "  • Your program includes MPI support (rank 0 does all work)"
echo "  • For better performance: Distribute work across MPI ranks"
echo ""
echo "🚀 To submit to judge system:"
echo "   cd /home/p13922006/Parallelism/hw2"
echo "   hw2-judge"
echo ""

