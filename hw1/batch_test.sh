#!/bin/bash

# 批次測試腳本（在單個 srun 任務中測試所有案例）
cd /home/p13922006/Parallelism/hw1

echo "批次測試（6 threads, 在單個 srun 任務中執行）"
echo "========================================"

# 編譯
echo -n "編譯 hw1.cpp... "
g++ -std=c++17 -O3 -pthread -fopenmp hw1.cpp -o hw1 2>compile.log
if [ $? -eq 0 ]; then
    echo "完成"
else
    echo "失敗"
    cat compile.log
    exit 1
fi

# 建立測試腳本（使用 home 目录，计算节点可以访问）
cat > /home/p13922006/Parallelism/hw1/run_tests_$$.sh << 'EOF'
#!/bin/bash
export OMP_NUM_THREADS=6
cd /home/p13922006/Parallelism/hw1

# 尝试加载模块（可能失败，不影响后续执行）
module load python/3.12.2 2>/dev/null || true

passed=0
failed=0
timeout_count=0

for i in {01..25}; do
    echo -n "測試 $i.txt: "
    
    start_time=$(date +%s.%N)
    timeout 30 ./hw1 "samples/$i.txt" > "/tmp/answer_$i.txt" 2>&1
    exit_code=$?
    end_time=$(date +%s.%N)
    elapsed=$(echo "$end_time - $start_time" | bc -l)
    
    if [ $exit_code -eq 124 ]; then
        printf "❌ TIMEOUT (>30s)\n"
        ((timeout_count++))
        ((failed++))
    elif [ $exit_code -ne 0 ]; then
        printf "❌ ERROR (exit code: $exit_code)\n"
        ((failed++))
    else
        python3 validate.py "samples/$i.txt" "/tmp/answer_$i.txt" > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            steps=$(cat "/tmp/answer_$i.txt" | tr -d '\n' | wc -c)
            printf "✅ PASS (%.2fs, %d steps)\n" "$elapsed" "$steps"
            ((passed++))
        else
            printf "❌ WRONG ANSWER\n"
            ((failed++))
        fi
    fi
done

echo "========================================"
echo "總結："
echo "  通過:  $passed / 25"
echo "  失敗:  $failed / 25"
echo "  超時:  $timeout_count / 25"
EOF

chmod +x /home/p13922006/Parallelism/hw1/run_tests_$$.sh

# 使用單個 srun 任務執行所有測試
echo "開始測試（這會在單個 srun 任務中執行所有測試）..."
srun -A ACD114118 -n1 -c6 --time=00:15:00 bash /home/p13922006/Parallelism/hw1/run_tests_$$.sh

rm -f /home/p13922006/Parallelism/hw1/run_tests_$$.sh

