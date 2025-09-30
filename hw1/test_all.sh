#!/bin/bash

# 測試所有樣本的腳本
cd /home/p13922006/Parallelism/hw1

echo "開始測試所有樣本..."
echo "================================"

# 載入 Python 模組
module load python/3.12.2

for i in {01..25}; do
    echo "測試樣本 $i.txt:"
    
    # 執行求解器
    echo -n "  求解中... "
    start_time=$(date +%s.%N)
    result=$(./hw1 samples/$i.txt 2>&1)
    end_time=$(date +%s.%N)
    
    if [ $? -eq 0 ]; then
        # 計算執行時間
        execution_time=$(echo "$end_time - $start_time" | bc -l)
        
        # 驗證結果
        echo "$result" | python3 validate.py samples/$i.txt - > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "✅ 正確 (${execution_time}s, ${#result} 步)"
        else
            echo "❌ 錯誤解答"
            echo "    輸出: $result"
        fi
    else
        echo "❌ 執行失敗"
        echo "    錯誤: $result"
    fi
done

echo "================================"
echo "測試完成！"
