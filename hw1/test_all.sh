#!/bin/bash

# 測試所有樣本的腳本
cd /home/p13922006/Parallelism/hw1

echo "開始測試所有樣本..."
echo "================================"

# 載入所需模組並編譯執行檔
module load gcc/11.4.0
module load openmpi/4.1.6
module load python/3.12.2

echo -n "編譯 hw1.cpp... "
if mpic++ -O2 -std=c++17 hw1.cpp -o hw1 2> compile.log; then
    echo "完成"
else
    echo "失敗"
    cat compile.log
    exit 1
fi

for i in {01..25}; do
    echo "測試樣本 $i.txt:"
    
    # 執行求解器
    echo -n "  求解中... "
    tmp_out=$(mktemp)
    tmp_err=$(mktemp)
    start_time=$(date +%s.%N)
    srun -A ACD114118 -N 1 -n 1 ./hw1 "samples/$i.txt" >"$tmp_out" 2>"$tmp_err"
    status=$?
    end_time=$(date +%s.%N)

    if [ $status -eq 0 ]; then
        result=$(cat "$tmp_out")
        filtered_err=$(grep -v '^srun:' "$tmp_err" || true)
        # 計算執行時間
        execution_time=$(echo "$end_time - $start_time" | bc -l)
        
        # 驗證結果
        echo "$result" | python3 validate.py samples/$i.txt - > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "✅ 正確 (${execution_time}s, ${#result} 步)"
            if [ -n "$filtered_err" ]; then
                echo "    (srun 訊息)"
                echo "$filtered_err"
            fi
        else
            echo "❌ 錯誤解答"
            echo "    輸出: $result"
            if [ -n "$filtered_err" ]; then
                echo "    (srun 訊息)"
                echo "$filtered_err"
            fi
        fi
    else
        echo "❌ 執行失敗"
        if [ -s "$tmp_out" ]; then
            echo "    輸出:"
            cat "$tmp_out"
        fi
        if [ -s "$tmp_err" ]; then
            echo "    錯誤:"
            cat "$tmp_err"
        fi
    fi

    rm -f "$tmp_out" "$tmp_err"
done

echo "================================"
echo "測試完成！"
