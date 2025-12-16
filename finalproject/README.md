## Monte Carlo option pricing (CUDA C++ sample)

This folder contains a minimal **GPU Monte Carlo** pricer for:

- **European call** payoff: \(\max(S_T - K, 0)\)
- **Asian arithmetic-average call** payoff: \(\max(\bar S - K, 0)\)

It is designed to match your `machine.txt` environment (Tesla V100, **compute capability 7.0**).

### Taiwania2 (NCHC) Lab 3 GPU cluster submission (srun)

From `Lab 3 GPU Cluster.pdf`: login node is `ln01.twcc.ai`, **compile and submit on login node**, and run via:

```bash
srun -N 1 -n 1 --gpus-per-node 1 -A ACD114118 -t 1 ./your_executable [args]
```

This repo includes a helper script:

- `run_srun.sh` (loads `cuda` module, compiles `mc_pricer.cu`, and submits with the required `srun` flags)
- `run_srun_8gpu.sh` (1 node uses 8 tasks + 8 GPUs; each task auto-selects GPU by `SLURM_LOCALID`)

### Build (Linux / cluster node with CUDA)

```bash
nvcc -O3 -std=c++17 -arch=sm_70 mc_pricer.cu -o mc_pricer
```

### Run

European call (default):

```bash
./mc_pricer --type european --paths 5000000 --steps 252
```

Asian call:

```bash
./mc_pricer --type asian --paths 5000000 --steps 252
```

Basket (multi-asset, correlated) call (heavier workload):

```bash
./mc_pricer --type basket --assets 16 --rho 0.3 --paths 2000000 --steps 252
```

Optional CPU baseline check (slow for large `--paths`):

```bash
./mc_pricer --type european --paths 200000 --steps 252 --cpu
```

### Taiwania2 recommended experiments (to show GPU scaling)

Single GPU, heavier compute:

```bash
./run_srun.sh -- --type basket --assets 32 --rho 0.3 --paths 2000000 --steps 252
```

Single node, 8 GPUs (8 tasks, one GPU per task):

```bash
chmod +x run_srun_8gpu.sh
./run_srun_8gpu.sh -- --type basket --assets 16 --rho 0.3 --paths 2000000 --steps 252
```

### Parameters

- `--S0` initial price (default 100)
- `--K` strike (default 100)
- `--r` risk-free rate (default 0.05)
- `--sigma` volatility (default 0.2)
- `--T` maturity in years (default 1.0)
- `--steps` time steps per path (default 252)
- `--paths` number of Monte Carlo paths (default 5,000,000)
- `--seed` RNG seed (default 1234)
- `--assets` number of assets for basket (default 1)
- `--rho` equicorrelation for basket (default 0.0). Must satisfy \(\rho \in (-1/(M-1), 1)\).
- `--block-size` CUDA block size (default 256)
- `--blocks-per-sm` blocks per SM (default 8)
- `--device` CUDA device id override (default: auto from `SLURM_LOCALID` if present)

### Download 2330 historical prices from Yahoo (CSV) + estimate sigma

This repo includes a small helper script that **always downloads `2330.TW`** daily history from Yahoo Finance
and optionally prints an estimated annualized volatility (sigma) from log returns.

Prereqs (Python 3.9+ recommended):

```bash
pip install -r requirements.txt
```

Download the last 5 years (default):

```bash
python tools/download_2330_yahoo.py
```

Download the last N years:

```bash
python tools/download_2330_yahoo.py --years 3
```

Or specify an explicit date range:

```bash
python tools/download_2330_yahoo.py --start 2020-01-01 --end 2025-12-16
```

Output is saved to `data/2330_TW.csv` by default. Yahoo typically allows **as much daily history as it has**
for the ticker (often decades). Practically, 1–5 years is common for estimating a stable recent sigma,
but you can increase it if you want longer-term volatility.

### One-shot: download 5y data -> mc_pricer loads it -> write result CSV

1) Download last 5 years (default) to `data/2330_TW.csv`:

```bash
python tools/download_2330_yahoo.py --years 5
```

2) Compile `mc_pricer` (on a CUDA machine / Taiwania2 login node):

```bash
nvcc -O3 -std=c++17 -arch=sm_70 mc_pricer.cu -o mc_pricer
```

3) Run with `--yahoo-csv` so `mc_pricer` automatically sets `S0` (latest close) and annualized `sigma`
from the CSV, then writes the run summary to `results_2330.csv`:

```bash
./mc_pricer --type european --paths 5000000 --steps 252 \
  --yahoo-csv data/2330_TW.csv --out-csv results_2330.csv
```

If you want to append multiple runs into the same CSV (e.g., different `--paths`):

```bash
./mc_pricer --type european --paths 1000000 --steps 252 --yahoo-csv data/2330_TW.csv --out-csv results_2330.csv --append-csv
./mc_pricer --type european --paths 5000000 --steps 252 --yahoo-csv data/2330_TW.csv --out-csv results_2330.csv --append-csv
```

### Forecast-style output: dump 1-year daily price paths (CSV)

If what you want is **"future 1-year daily trajectories"** (not an option price), use `--dump-paths-csv`.
This writes `dump_paths * (steps+1)` rows:

- columns: `path_id, step, t_years, price`
- default `steps=252` (trading days), default `dump_paths=100`

Example:

```bash
./mc_pricer --yahoo-csv data/2330_TW.csv --T 1 --steps 252 \
  --dump-paths 100 --dump-paths-csv paths_2330_1y.csv
```

By default, the drift for simulation uses **mu estimated from the Yahoo CSV** (annualized mean of log returns).
If you want to disable that and use `r` instead:

```bash
./mc_pricer --yahoo-csv data/2330_TW.csv --T 1 --steps 252 \
  --no-yahoo-mu --dump-paths 100 --dump-paths-csv paths_2330_1y.csv
```

### Aggregate many paths into mean/std band (CSV)

If you simulate many paths (e.g. 1000) and want **daily mean / std / percentile bands**:

1) Dump paths (example: 1000 paths):

```bash
./mc_pricer --yahoo-csv data/2330_TW.csv --T 1 --steps 252 \
  --dump-paths 1000 --dump-paths-csv paths_2330_1y_1000.csv
```

2) Aggregate into bands (default percentiles: 5/50/95):

```bash
python tools/aggregate_paths_band.py --in paths_2330_1y_1000.csv --out band_2330_1y_1000.csv
```

Output columns include: `step,t_years,mean,std,p5,p50,p95`


