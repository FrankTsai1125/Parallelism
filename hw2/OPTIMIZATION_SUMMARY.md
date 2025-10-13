# HW2 SIFT - 方向一優化總結

## 優化目標
實現 MPI + OpenMP 混合平行化的 SIFT 演算法，重點在於**優化 Pyramid 建構策略**，避免所有 ranks 重複建構完整金字塔。

## 已完成的優化

### 1. MPI 分散式處理架構 ✅
- **圖像廣播** (`mpi_broadcast_image`): Rank 0 讀取圖像並廣播給所有 ranks
- **Octave 分配** (`compute_octave_partition`): 將 8 個 octaves 平均分配給不同 processes
- **結果收集** (`mpi_gather_keypoints`): 收集所有 ranks 的 keypoints 到 rank 0

### 2. OpenMP 執行緒平行化 ✅
優化的函數包括：

#### `generate_dog_pyramid` (DoG 金字塔生成)
- 使用 `#pragma omp parallel for schedule(static)` 平行處理不同 octaves
- 使用 `#pragma omp simd` 向量化像素級運算

#### `find_keypoints` (關鍵點檢測)
- **兩階段平行化**：
  - Phase 1: 平行檢測候選極值點，使用 `collapse(2)` 和 `schedule(dynamic)`
  - Phase 2: 平行優化和驗證關鍵點，使用 `schedule(dynamic)` 處理負載不平衡
- 使用 thread-local 緩衝區減少 critical section 開銷

#### `generate_gradient_pyramid` (梯度金字塔)
- 外層 octaves 使用 `schedule(static)`
- 內層像素使用 `collapse(2)` 和 `schedule(static)`

#### `find_keypoints_and_descriptors` (描述符計算)
- 使用 `schedule(dynamic)` 平行計算不同關鍵點的描述符
- Thread-local 緩衝區收集結果

#### `gaussian_blur` (高斯模糊)
- 垂直和水平卷積都使用 `collapse(2)` 和 `schedule(static)`
- 這是最耗時的操作之一，優化效果顯著

### 3. `find_keypoints_range` - MPI 分散式處理核心 ✅
- 每個 rank 只處理分配給它的 octaves 範圍
- 結合 OpenMP 在每個 rank 內進行執行緒平行化
- 兩階段處理：候選點檢測 → 描述符計算

## 測試結果

### 測試環境
- 測試案例：testcases/01.jpg
- 配置：6 cores per process
- 驗證：使用 validate.py 對照 golden files

### 性能數據
| 配置 | 執行時間 | Keypoints | 驗證結果 | 加速比 |
|------|---------|-----------|---------|--------|
| N=1, n=1, c=6 | 17921.8 ms | 45730 | Pass | 1.00x |
| N=1, n=2, c=6 | 17031.1 ms | 45730 | Pass | 1.05x |

### 觀察與分析
1. ✅ **正確性**: 兩種配置都通過驗證，找到相同數量的 keypoints
2. ✅ **初步加速**: 使用 2 個 processes 已經有約 5% 的加速
3. 📊 **改進空間**: 目前所有 ranks 仍建構完整金字塔，這是下一步優化重點

## 當前架構說明

### hw2.cpp 主流程
```cpp
1. MPI_Init - 初始化 MPI 環境
2. Rank 0 讀取圖像並轉為灰階
3. mpi_broadcast_image - 廣播圖像到所有 ranks
4. compute_octave_partition - 計算每個 rank 的 octave 分配
5. 每個 rank:
   - 建構完整的 Gaussian/DoG/Gradient pyramids (待優化)
   - 只處理分配給自己的 octaves (已優化)
6. mpi_gather_keypoints - 收集所有 keypoints 到 rank 0
7. Rank 0 寫入結果並輸出時間
8. MPI_Finalize
```

### 調度策略
- **Static scheduling**: 用於負載均衡的任務 (金字塔生成、梯度計算)
- **Dynamic scheduling**: 用於負載不均的任務 (關鍵點檢測、描述符計算)
- **Collapse(2)**: 增加平行粒度，減少排程開銷

## 下一步優化方向

### 方向 1 進階優化 (Pyramid 建構優化)
目前狀態：**所有 ranks 建構完整金字塔，然後只處理部分 octaves**

優化策略：
1. **選項 A**: Rank 0 建構金字塔後使用 MPI_Scatterv 分發對應 octaves
2. **選項 B**: 每個 rank 只建構自己需要的 octaves
3. **選項 C**: Pipeline 處理 - 重疊計算和通訊

### 方向 2: 負載平衡
- 基於圖像大小和關鍵點密度的動態分配
- 考慮不同 octaves 的實際工作量

### 方向 3: 通訊優化
- 使用非阻塞 MPI 操作 (MPI_Ibcast, MPI_Igatherv)
- 減少數據傳輸量

### 方向 4: 演算法優化
- SIMD 向量化更多計算密集部分
- 改善 cache locality
- 減少 critical sections

## 編譯和測試命令

### 編譯
```bash
cd /home/p13922006/Parallelism/hw2
bash compile_test.sh
```

### 快速測試
```bash
bash quick_test.sh
```

### 完整測試
```bash
bash test_judge.sh
```

## 關鍵文件說明
- `hw2.cpp`: MPI 主程式，處理分散式協調
- `sift.hpp/cpp`: SIFT 演算法實現，包含 MPI 輔助函數
- `image.hpp/cpp`: 圖像處理函數，包含高斯模糊等
- `compile_test.sh`: 編譯腳本
- `quick_test.sh`: 快速測試腳本

## 作業要求檢查清單
- ✅ 使用 MPI 進行分散式處理
- ✅ 使用 OpenMP 進行執行緒平行化
- ✅ 混合 Static 和 Dynamic scheduling
- ✅ 考慮負載平衡（使用 dynamic scheduling）
- ✅ 不修改 SIFT 參數（保持算法正確性）
- ✅ 不修改驗證相關代碼
- ✅ 輸出正確且通過驗證

## 總結
**方向一的優化已成功實現基礎架構**：
1. ✅ MPI 分散式處理框架完整
2. ✅ OpenMP 多執行緒優化到位
3. ✅ 程式正確性驗證通過
4. ✅ 已有初步加速效果
5. 📈 後續優化空間明確

下一步可以繼續優化 Pyramid 建構策略，或進行其他方向的優化來進一步提升性能。

