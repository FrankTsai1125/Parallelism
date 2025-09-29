#!/bin/bash

# ==================== SBATCH 參數區 ====================
#SBATCH --account=ACD114118          # 您的專案帳號
#SBATCH --job-name=hello_test        # 工作名稱（自行命名）
#SBATCH --nodes=1                    # 使用的節點數
#SBATCH --ntasks=4                   # MPI 任務數（或執行緒數）
#SBATCH --time=00:10:00              # 最長執行時間
#SBATCH --output=job-%j.out          # ← 這裡指定標準輸出檔案
#SBATCH --error=job-%j.err           # ← 這裡指定錯誤輸出檔案
# =====================================================

srun lab1

