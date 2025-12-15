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


