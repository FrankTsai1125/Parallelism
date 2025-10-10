#!/bin/bash

cd /home/p13922006/Parallelism/lab2/mpi_pr_lab

echo "╔═══════════════════════════════════════════════════════╗"
echo "║         MPI Lab2 - Circle Pixel Calculator            ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Load environment
source env.sh 2>/dev/null

echo "📦 Compiling..."
make clean > /dev/null 2>&1
make

if [ $? -ne 0 ]; then
    echo "❌ Compilation failed!"
    exit 1
fi
echo "✓ Compilation successful"
echo ""

echo "🧪 Running Sample Tests:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Function to run a test
run_test() {
    local testnum=$1
    local nproc=$2
    local r=$3
    local k=$4
    local expected=$5
    
    printf "Test %02d (n=%d, r=%-11s k=%-11s): " "$testnum" "$nproc" "$r" "$k"
    
    result=$(srun -n$nproc -A ACD114118 ./lab2 $r $k 2>&1 | grep -oE '^[0-9]+$' | head -1)
    
    if [ "$result" = "$expected" ]; then
        echo "✓ PASS (answer: $result)"
    else
        echo "❌ FAIL (got: $result, expected: $expected)"
    fi
}

# Sample tests
run_test 1  1 5          100        88
run_test 2  2 5          21         4
run_test 3  3 214        214        24
run_test 4  4 2147       2147       2048
run_test 5  5 21474      21474      11608

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ Testing completed!"
echo ""
echo "📝 To test manually:"
echo "   source env.sh"
echo "   srun -n<nproc> -A ACD114118 ./lab2 <r> <k>"
echo ""
echo "📂 Test all cases:"
echo "   Run tests from testcases/ directory manually"

