import argparse
import math
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Add total_threads column to cpu.csv/1gpu.csv/2gpu.csv for avg_path runs."
    )
    p.add_argument("--cpu", default="cpu.csv", help="CPU bench CSV (default: cpu.csv)")
    p.add_argument("--gpu1", default="1gpu.csv", help="1GPU bench CSV (default: 1gpu.csv)")
    p.add_argument("--gpu2", default="2gpu.csv", help="2GPU bench CSV (default: 2gpu.csv)")
    p.add_argument("--cpu-threads", type=int, default=4, help="OpenMP thread count used for CPU run (default: 4)")
    p.add_argument("--gpu1-ntasks", type=int, default=1, help="Slurm ntasks for 1GPU run (default: 1)")
    p.add_argument("--gpu2-ntasks", type=int, default=2, help="Slurm ntasks for 2GPU run (default: 2)")
    return p.parse_args()


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _gpu_total_threads(total_paths: int, ntasks: int, block_size: int) -> int:
    # Mirror mc_pricer.cu path split:
    base = total_paths // ntasks
    rem = total_paths % ntasks
    tot = 0
    for pid in range(ntasks):
        local = base + (1 if pid < rem else 0)
        blocks = _ceil_div(local, block_size)
        tot += blocks * block_size
    return tot


def _upgrade_cpu(path: Path, cpu_threads: int) -> None:
    df = pd.read_csv(path)
    if "total_threads" in df.columns:
        return
    df["total_threads"] = int(cpu_threads)
    df.to_csv(path, index=False)


def _upgrade_gpu(path: Path, ntasks: int) -> None:
    df = pd.read_csv(path)
    if "total_threads" in df.columns:
        return
    if "paths" not in df.columns:
        raise SystemExit(f"{path}: missing 'paths' column")
    # avg-path GPU rows should include block_size; fall back to 256 if blank.
    bs = 256
    if "block_size" in df.columns:
        v = pd.to_numeric(df["block_size"], errors="coerce").dropna()
        if not v.empty:
            bs = int(v.iloc[0])

    paths_vals = pd.to_numeric(df["paths"], errors="coerce")
    if paths_vals.isna().all():
        raise SystemExit(f"{path}: could not parse paths")
    total_paths = int(paths_vals.dropna().iloc[0])
    df["total_threads"] = _gpu_total_threads(total_paths=total_paths, ntasks=int(ntasks), block_size=int(bs))
    df.to_csv(path, index=False)


def main() -> int:
    args = parse_args()
    _upgrade_cpu(Path(args.cpu), args.cpu_threads)
    _upgrade_gpu(Path(args.gpu1), args.gpu1_ntasks)
    _upgrade_gpu(Path(args.gpu2), args.gpu2_ntasks)
    print("Upgraded: total_threads column added (if it was missing).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


