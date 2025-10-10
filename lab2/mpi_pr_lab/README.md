# MPI Lab2 - Circle Pixel Calculator

## 問題描述

計算半徑為 `r` 的圓形在 2D 螢幕上會填滿多少像素。

- 圓心位於 4 個像素的邊界上
- 只要圓形的任何部分與像素重疊就填滿該像素
- 輸出結果對 `k` 取模

### 算法

計算 1/4 圓的像素數，然後乘以 4：

```
對於 x = 0 到 r-1：
    y = ceil(√(r² - x²))
    累加 y
結果 = (累加值 × 4) % k
```

## 實作說明

### 並行策略

使用 **MPI** 將工作分配給多個進程：

1. **工作分配**：將 `x ∈ [0, r)` 均勻分配給各個 MPI 進程
2. **Local 計算**：每個進程獨立計算自己負責的 x 範圍
3. **結果匯總**：使用 `MPI_Reduce` 將所有局部結果加總到 rank 0

### 負載平衡

為了確保工作均勻分配，當 `r` 不能被 `world_size` 整除時：
- 前 `remainder` 個進程各多處理 1 個元素
- 剩餘進程處理標準的 `chunk_size` 個元素

### 核心代碼

```cpp
// 工作分配
unsigned long long chunk_size = r / world_size;
unsigned long long remainder = r % world_size;

if (rank < remainder) {
    start = rank * (chunk_size + 1);
    end = start + chunk_size + 1;
} else {
    start = remainder * (chunk_size + 1) + (rank - remainder) * chunk_size;
    end = start + chunk_size;
}

// Local 計算
unsigned long long local_pixels = 0;
for (unsigned long long x = start; x < end; x++) {
    unsigned long long y = ceil(sqrtl(r*r - x*x));
    local_pixels += y;
    local_pixels %= k;
}

// 結果匯總
unsigned long long total_pixels = 0;
MPI_Reduce(&local_pixels, &total_pixels, 1, MPI_UNSIGNED_LONG_LONG, 
           MPI_SUM, 0, MPI_COMM_WORLD);

// Rank 0 輸出結果
if (rank == 0) {
    printf("%llu\n", (4 * total_pixels) % k);
}
```

## 編譯與執行

### 環境設定

```bash
module load gcc/13
module load openmpi
```

或使用：

```bash
source env.sh
```

### 編譯

```bash
make
```

### 執行

```bash
# 基本執行：n 個 MPI 進程
srun -n<nproc> -A ACD114118 ./lab2 <r> <k>

# 範例
srun -n1 -A ACD114118 ./lab2 5 100
# 輸出: 88

srun -n4 -A ACD114118 ./lab2 2147 2147
# 輸出: 2048
```

### 進階選項

```bash
# 每個進程分配 4 個 CPU
srun -n5 -c4 -A ACD114118 ./lab2 21474 21474

# Debug 模式：顯示進程 rank
srun -n3 -l -A ACD114118 ./lab2 214 214
```

## 測試結果

已驗證所有測試案例（手動測試）：

| Test | nproc | r          | k          | Expected      | Result | Status |
|------|-------|------------|------------|---------------|--------|--------|
| 01   | 1     | 5          | 100        | 88            | 88     | ✓      |
| 02   | 2     | 5          | 21         | 4             | 4      | ✓      |
| 03   | 3     | 214        | 214        | 24            | 24     | ✓      |
| 04   | 4     | 2147       | 2147       | 2048          | 2048   | ✓      |
| 05   | 5     | 21474      | 21474      | 11608         | 11608  | ✓      |
| 06   | 6     | 214748     | 214748     | 157656        | 157656 | ✓      |
| 07   | 7     | 2147483    | 2147483    | 1748568       | 1748568| ✓      |
| 08   | 8     | 21474836   | 21474836   | 300000        | 300000 | ✓      |
| 09   | 9     | 214748364  | 214748364  | 153006692     | 153006692 | ✓  |
| 10   | 10    | 2147483647 | 2147483647 | 256357661     | 256357661 | ✓  |

**所有測試通過！** ✅

## 效能要求

根據 Lab 要求：
- 使用 `n` 個進程時，速度至少要是 sequential 的 **n/2 倍**
- 本實作採用均勻分配策略，理論上可達到接近線性加速

## 文件說明

- `lab2.cpp` - MPI 並行版本的主程式
- `sample/lab2.cpp` - Sequential 版本（參考用）
- `Makefile` - 編譯腳本
- `env.sh` - 環境設定腳本
- `demo.sh` - Demo 測試腳本
- `testcases/*.txt` - 測試案例

## 實作重點

1. ✅ 正確的工作分配（處理不整除情況）
2. ✅ 使用 `unsigned long long` 處理大數值
3. ✅ 使用 `MPI_UNSIGNED_LONG_LONG` 進行 Reduce
4. ✅ 局部取模避免溢位（`local_pixels %= k`）
5. ✅ 只有 rank 0 輸出結果

## 關鍵優化

- **負載平衡**：均勻分配工作給所有進程
- **數值穩定**：使用 `sqrtl()` 提高精度
- **溢位處理**：在累加過程中進行取模運算

