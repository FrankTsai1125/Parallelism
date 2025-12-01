 Implementation Details & Questions

 Implementation:

 What is your parallelism strategy?
平行化策略：
我採用了針對不同問題規模 (Problem Size) 的混合平行策略：

1.  針對極小規模 (N <= 64)：
    -   使用 CPU (OpenMP) 進行運算。因為在粒子數量極少時，將資料傳輸到 GPU 以及啟動 Kernel 的 Overhead 會大於實際運算時間。
2.  針對大規模 (N > 64)：
    -   使用 GPU (HIP) 進行加速。
    -   Thread Parallelism：採用「One Thread Per Particle」策略（對應程式碼中的 `compute_accel_per_particle` Kernel）。每個 Thread 負責計算一個粒子受到的總合力。
    -   Instruction Level Parallelism (ILP)：在計算力的迴圈中，利用 GPU 的大量暫存器與 CUDA Core 進行平行運算。
    -   Task Parallelism：在 Problem 3 (最佳化問題) 中，利用 OpenMP 將不同的候選裝置模擬任務 (Task) 分配給不同的 GPU 執行。

 Optimization techniques?
優化技巧：

1.  Shared Memory Reduction (區塊縮減)：
    -   在 `compute_accel_per_particle` Kernel 中，為了避免所有 Thread 同時寫入全域記憶體造成競爭，我利用 Shared Memory (`s_fx`, `s_fy`, `s_fz`) 在 Block 內部先行加總部分力，最後再由一個 Thread 寫回 Global Memory。這大幅減少了 Global Memory 的存取衝突。
2.  Structure of Arrays (SoA)：
    -   將粒子的位置 (qx, qy, qz)、速度 (vx, vy, vz) 等屬性分開儲存為獨立的陣列，而非 Array of Structures (AoS)。這確保了 GPU 在讀取資料時能達成 Coalesced Access (合併存取)，最大化記憶體頻寬利用率。
3.  Batch Status Check (批次狀態檢查)：
    -   為了減少 CPU 與 GPU 之間的 PCIe 傳輸開銷，我不是在每一個 Time Step 都將資料傳回 CPU 檢查是否有撞擊。而是每 500 個 Steps (Batch Size) 才同步一次 `stop_flag`。在 Kernel 內部若發生撞擊，會直接設定 Device 端的 Flag，下一次同步時 CPU 就會知道並終止模擬。
4.  Fast Math:
    -   使用 `rsqrt` (Reciprocal Square Root) 取代標準的 `1.0/sqrt(x)`，利用 GPU 硬體指令加速距離倒數的計算。

 How do you manage the 2-GPU resources?
雙 GPU 資源管理：

在 Problem 3 中，我們需要從多個候選的重力裝置中找出成本最低者。這意味著需要執行多次獨立的 N-Body 模擬。
-   我使用 OpenMP (`pragma omp parallel for`) 建立與系統 GPU 數量相等的 CPU Threads。
-   每個 CPU Thread 根據其 ID (`omp_get_thread_num()`) 透過 `hipSetDevice()` 綁定到特定的 GPU。
-   迴圈會自動將不同的測試案例 (Candidate Device ID) 分配給不同的 Thread (即不同的 GPU)。
-   結果是：GPU 0 計算 Case A, Case C... 而 GPU 1 同時計算 Case B, Case D...，達成 Task-Level Parallelism，理論上能使搜尋速度提升約 2 倍。

 If there are 4 GPUs instead of 2, what would you do to maximize the performance?
若有 4 顆 GPU 的效能最佳化：

由於我的程式架構已經將 Problem 3 的模擬任務視為獨立的 Task 並透過 OpenMP 動態分配：
-   我只需要確保 OpenMP 的 `num_threads` 設定為 GPU 數量 (或是讓 OpenMP 自動偵測)。
-   程式中的 `gpu_id = tid % num_gpus` 邏輯會自動將任務平均分配到 0, 1, 2, 3 號 GPU。
-   這將使得吞吐量 (Throughput) 提升為原來的 4 倍，顯著減少尋找最佳解的時間。

 If there are 5 gravity devices, is it necessary to simulate 5 n-body simulations independently for this problem?
是否需要獨立模擬 5 次：

是，必須獨立模擬。
因為 N-Body 問題是一個非線性且對初始條件高度敏感的系統 (混沌系統)。
-   當我們選擇啟動第 $i$ 個重力裝置時，該裝置的質量會隨時間變化 ($m(t)$)，這會改變該裝置對周圍所有粒子的引力。
-   這個引力的改變會導致粒子的軌跡發生偏移，進而改變粒子在下一時刻的位置，再次影響所有粒子間的交互作用力。
-   因此，啟動裝置 A 的模擬結果 (所有粒子的軌跡) 與啟動裝置 B 的結果會完全不同，且無法透過線性疊加或局部修正來預測。我們必須對每一種情形完整執行一次模擬才能得到正確的結果。

