#!/bin/bash

# 測試所有樣本的腳本（符合助教要求：6個執行緒，30秒限制）
cd /home/p13922006/Parallelism/hw1

echo "開始測試所有樣本（6 threads, 30s timeout）..."
echo "================================"

# 先取消所有現有的 srun 任務，避免達到提交限制
echo "檢查並取消現有任務..."
scancel -u $(whoami) 2>/dev/null
sleep 2
echo "清理完成"
echo ""

# 載入所需模組並編譯執行檔
module load gcc/11.4.0
module load python/3.12.2

echo -n "編譯 hw1.cpp... "
if g++ -std=c++17 -O3 -pthread -fopenmp hw1.cpp -o hw1 2> compile.log; then
    echo "完成"
else
    echo "失敗"
    cat compile.log
    exit 1
fi

passed=0
failed=0

for i in {01..25}; do
    echo "測試樣本 $i.txt:"
    
    # 執行求解器（6個執行緒，30秒超時）
    echo -n "  求解中... "
    tmp_out=$(mktemp)
    tmp_err=$(mktemp)
    start_time=$(date +%s.%N)
    
    # srun 自帶超時機制
    srun -A ACD114118 -n1 -c6 --time=00:00:30 ./hw1 "samples/$i.txt" >"$tmp_out" 2>"$tmp_err"
    status=$?
    end_time=$(date +%s.%N)

    if [ $status -eq 0 ]; then
        result=$(cat "$tmp_out")
        execution_time=$(echo "$end_time - $start_time" | bc -l)
        
        # 驗證結果
        echo "$result" | python3 validate.py samples/$i.txt - > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "✅ 正確 (${execution_time}s, ${#result} 步)"
            ((passed++))
        else
            echo "❌ 錯誤解答"
            ((failed++))
        fi
    else
        # 檢查錯誤類型
        if grep -q "TIME LIMIT\|TIMEOUT\|CANCELLED" "$tmp_err" 2>/dev/null; then
            echo "❌ TIMEOUT (>30s)"
        elif grep -q "Out Of Memory\|OOM\|oom-kill" "$tmp_err" 2>/dev/null; then
            echo "❌ OUT OF MEMORY"
        else
            echo "❌ 執行失敗 (exit code: $status)"
        fi
        ((failed++))
        
        # 顯示錯誤訊息（過濾 srun 資訊）
        if [ -s "$tmp_err" ]; then
            grep -v '^srun:' "$tmp_err" | head -3
        fi
    fi

    rm -f "$tmp_out" "$tmp_err"
    
    # 稍微等待，避免提交太快
    sleep 0.5
done

echo "================================"
echo "測試完成！"
echo "通過: $passed / 25"
echo "失敗: $failed / 25"
