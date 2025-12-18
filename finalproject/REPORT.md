### Introduction

本專案以 **Monte Carlo (MC) 模擬**實作金融衍生品（期權）相關的價格模擬與定價，並在 NCHC Taiwania2 GPU cluster 上進行 **CPU（OpenMP）/ Single‑GPU（CUDA）/ Multi‑GPU（Slurm multi‑task）** 的效能比較。  
MC 方法的核心優點是彈性高、可自然延伸到路徑相依（Asian option）或多資產（multi‑asset）情境；缺點是需要大量路徑與時間步長（steps）才能收斂，因此非常適合用 GPU 的 massive parallelism 來加速。

依照 proposal 與課程作業目標（可參考你們的投影片/文件）：我們希望展示
- **正確的數學模型與 MC 估計流程**
- **從 CPU → GPU → Multi‑GPU 的平行化策略**
- **可重現的 benchmark 與圖表（runtime / speedup / threads）**

（參考：你們的投影片 `Parallelism.pdf` 與提案 `proposal-1.pdf`。）  

---

### Problem Description & Objective

- **背景動機**：MC 是最通用的定價工具之一，特別適用於路徑相依或高維度（multi‑asset）期權，但計算成本高；CPU serial/弱平行在大 N 下延遲過高。  
- **目標**：最大化 simulation throughput，並探索 CUDA/GPU 在 computational finance 的加速成效。  
- **驗證/評估方向（proposal）**：
  - **single & multi‑asset** MC model
  - **baseline（CPU）** 與 **GPU speedup**
  - multi‑asset 相關常態（correlated Gaussian）生成（Cholesky）
  - benchmark N（paths）、T（steps/tenor）、M（assets）對效能的影響

---

### Mathematical Model

#### Asset Price Dynamics（GBM）

我們假設標的價格滿足幾何布朗運動（Geometric Brownian Motion, GBM）：
\[
dS_t = \mu S_t dt + \sigma S_t dW_t
\]

離散化後，每一步更新為：
\[
S_{t+\Delta t} = S_t \cdot \exp\left(\left(\alpha - \frac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t}\,Z\right),\quad Z\sim \mathcal{N}(0,1)
\]

其中 \(\alpha\) 依情境而定：
- **risk‑neutral（定價）**：\(\alpha = r\)（無套利假設下的風險中性漂移）
- **forecast / trajectory（路徑輸出）**：可使用 \(\alpha = \mu\)（由歷史 log return 估計的平均漂移），或用 `--no-yahoo-mu` 強制回到 \(r\)

#### Payoff Functions（本專案）

本 repo 的 `mc_pricer.cu` 實作了三種 option payoff（call）：

- **European call**
\[
\text{payoff} = \max(S_T - K, 0)
\]

- **Asian arithmetic‑average call**
\[
\text{payoff} = \max\left(\bar S - K, 0\right),\quad \bar S = \frac{1}{N}\sum_{i=1}^{N} S_{t_i}
\]

- **Basket (multi‑asset) European call**（terminal basket arithmetic mean）
\[
B_T = \frac{1}{M}\sum_{j=1}^{M} S_T^{(j)},\quad \text{payoff}=\max(B_T-K,0)
\]

#### Monte Carlo Estimator（Price / Std Error）

定價（risk‑neutral）下，期權公平價格：
\[
C = e^{-rT}\mathbb{E}[\text{payoff}]
\]

MC 以 \(N\) 條路徑估計：
\[
\hat C = \frac{1}{N}\sum_{i=1}^{N} e^{-rT}\,\text{payoff}^{(i)}
\]

並以樣本二階矩估計方差與標準誤（standard error）：
\[
\widehat{\mathrm{SE}} = \sqrt{\frac{\widehat{\mathrm{Var}}}{N}}
\]
（程式會同時累積 \(\sum \text{payoff}\) 與 \(\sum \text{payoff}^2\) 來計算 mean / variance / standard error。）

---

### Multi‑Asset Model（Correlation / Cholesky）

在 multi‑asset 模式下，每個時間步需要產生 \(M\) 維的 correlated Gaussian：
- 先生成獨立 \(z\sim\mathcal{N}(0,I)\)
- 給定相關矩陣（或共變異矩陣）\(\Sigma\)，做 Cholesky 分解 \(\Sigma = LL^\top\)
- 令 \(y = Lz\)，即可得到 \(y\sim\mathcal{N}(0,\Sigma)\)

本 repo 目前採用 **equicorrelation**（所有非對角線相關係數皆為 \(\rho\)），並在 `mc_pricer.cu` 內部生成矩陣後做 Cholesky。  
（proposal 的「從檔案載入 correlation matrix」可作為後續延伸。）

---

### Implementation Details (Code‑Level)

#### CPU Baseline：OpenMP（`mc_pricer_omp.cpp`）

- **核心想法**：以 OpenMP 平行化「路徑」迴圈，每個 thread 累積自己的 `sum[step]` / `sumsq[step]`，最後 reduction 合併。
- **輸出**：`--avg-path-csv` 產生每個 step 的 mean/std trajectory。

#### GPU：CUDA + cuRAND（`mc_pricer.cu`）

- **RNG**：使用 cuRAND **Philox** counter‑based generator，適合大量 threads 產生獨立序列；single‑asset 模式使用 `curand_normal4` 提升 throughput。
- **Option pricing kernel**：grid‑stride over paths，每個 thread 生成路徑、算 payoff、乘上貼現，最後累積 \(\sum\) 與 \(\sum^2\)（`atomicAdd(double)` on sm_70）。
- **avg‑path 模式**：
  - kernel 計算每個 step 的 `sumS[t]`、`sumSqS[t]`
  - 以 shared memory 先做 block‑level reduction，再 `atomicAdd` 到 global
  - 最終輸出 `avg_2330_*.csv`（step,t_years,mean,std）
- **Multi‑GPU（multi‑task）合併**：
  - Slurm 多 task 時，依 `--paths-total` 分配每個 task 的 `local_paths`
  - 每個 task 輸出 partial bin（`*.partial.<proc>.bin`），由 `proc0` 等待並合併後寫出最終 CSV
  - 為避免 race，只有 `proc0` 會寫 `--out-csv` 的一行 summary

---

### Data & Calibration（Yahoo 2330.TW）

本 repo 使用 `tools/download_2330_yahoo.py` 下載 `2330.TW` 歷史價格（CSV），並以 log returns 估計：
- **年化波動度**：\(\sigma_{\text{annual}} = \sigma_{\text{daily}}\sqrt{252}\)
- **年化平均漂移**：以 log return 的平均年化（供 trajectory 模式使用）

下載後的資料儲存在 `data/2330_TW.csv`，執行時用 `--yahoo-csv` 自動帶入 `S0/sigma/mu`。

---

### Experiments

#### Environment（Taiwania2）

- **Scheduler**：Slurm（`srun`）
- **GPU**：NVIDIA Tesla V100 (sm_70)
- **執行腳本**
  - `run_srun_cpu_omp.sh`：CPU baseline（OpenMP）
  - `run_srun.sh` / `run_srun_1gpu.sh`：1 GPU
  - `run_srun_2gpu.sh`：2 GPU（2 tasks / 2 GPUs）

#### Experiment Setup（avg‑path benchmark）

- **paths**：10,000,000
- **steps**：1000
- **輸出**
  - avg‑path CSV：`avg_2330_cpu.csv`, `avg_2330_1gpu.csv`, `avg_2330_2gpu.csv`
  - benchmark CSV：`cpu.csv`, `1gpu.csv`, `2gpu.csv`（含 `total_threads`）
  - 圖表：`perf_2330_runtime.png`, `perf_2330_speedup.png`

#### Results（以 `perf_2330_summary.csv` 為準）

（單位：seconds；speedup relative to CPU）

- **CPU（OpenMP, total_threads=4）**：約 **126.66 s**
- **1 GPU（total_threads≈10000128）**：約 **0.235 s**（**~538×**）
- **2 GPU（total_threads≈10000384）**：約 **0.245 s**（**~517×**）

圖檔：
- `perf_2330_runtime.png`（柱頂含 time；x 軸含 threads）
- `perf_2330_speedup.png`（柱頂含 speedup；x 軸含 threads）

#### Discussion（重要觀察）

- **GPU 的 massive parallelism** 對於 MC 這種「path‑independent」工作負載非常有效，出現數百倍加速屬合理結果。
- **2 GPU 未必比 1 GPU 更快**（本次 avg‑path case 中 2 GPU 略慢）可能原因包含：
  - multi‑task 的 partial I/O 與等待合併成本（proc0 等待其他 task 的 partial 檔案）
  - avg‑path kernel 的瓶頸可能偏向 atomic / reduction / memory traffic，而非純算力
  - paths 分配與 block rounding（threads 以 block_size 對齊）造成的額外工作

---

### Limitations & Future Work

- **Black–Scholes closed‑form 驗證**：proposal 提到與 BS closed‑form 對照（European）以驗證正確性；目前 repo 以工程/效能 pipeline 為主，BS 對照可作為補強。
- **CPU multi‑asset baseline**：proposal 期望 CPU 也能做 multi‑asset correlated simulation；目前 `mc_pricer_omp.cpp` 以 single‑asset avg‑path 為主。
- **Correlation matrix from file**：目前 multi‑asset 使用 equicorrelation（\(\rho\)）；若要更接近真實市場，需要支援從多檔股票 returns 建立 correlation/covariance matrix 並讀檔。
- **Profiling（Nsight）**：proposal 提到 Nsight Compute；可針對 kernel occupancy、memory throughput、atomic contention 進一步分析瓶頸。

---

### Work Distribution

（依 proposal 名單，以下用「可對照 repo 實作」的方式描述。可依實際情況調整。）

- **David Lu (T14902116)**：
  - **數學模型/文件**：GBM、risk‑neutral、payoff、MC estimator/SE 的整理
  - **報告統整**：章節架構與結果解讀
- **蔡琦皇 (P13922006)**：
  - **Cluster workflow**：Taiwania2 上 Slurm `srun` 腳本（CPU/1GPU/2GPU）與 reproducibility
  - **Benchmark pipeline**：`cpu.csv/1gpu.csv/2gpu.csv`、threads 標註、perf 圖表生成
- **楊翊廷 (R14944018)**：
  - **CUDA 實作與優化**：cuRAND、kernel 設計、avg‑path 統計與 multi‑task reduce
  - **Multi‑GPU**：task mapping、seed decorrelation、proc0 合併邏輯
- **詹淯翔 (B13201026)**：
  - **資料工具**：Yahoo 下載/估計 sigma/mu（`tools/download_2330_yahoo.py`）
  - **視覺化**：avg‑path 圖/輔助 plotting scripts

---

### References

- [Parallelism.pdf（投影片/報告草稿）](file:///c%3A/Users/admin/Documents/GitHub/Parallelism/finalproject/Parallelism.pdf)
- [proposal-1.pdf（提案）](file:///c%3A/Users/admin/Documents/GitHub/Parallelism/finalproject/proposal-1.pdf)
