import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot runtime + speedup from bench CSVs produced by run_srun*.sh (cpu.csv/1gpu.csv/2gpu.csv)."
    )
    p.add_argument("--cpu", default="cpu.csv", help="CPU bench CSV (default: cpu.csv)")
    p.add_argument("--gpu1", default="1gpu.csv", help="1GPU bench CSV (default: 1gpu.csv)")
    p.add_argument("--gpu2", default="2gpu.csv", help="2GPU bench CSV (default: 2gpu.csv)")
    p.add_argument("--out-prefix", default="perf_2330", help="Output prefix (default: perf_2330)")
    p.add_argument("--title", default="Performance comparison (avg-path)", help="Figure title")
    p.add_argument("--logy", action="store_true", help="Use log scale on runtime axis (recommended when CPU is much slower)")
    return p.parse_args()


def _load(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    df = pd.read_csv(path)
    df["config"] = label
    return df


def _infer_time_ms(df: pd.DataFrame) -> pd.Series:
    # Our schema keeps both columns; one side is blank depending on CPU/GPU.
    gpu = pd.to_numeric(df.get("gpu_time_ms"), errors="coerce")
    cpu = pd.to_numeric(df.get("cpu_time_ms"), errors="coerce")
    t = gpu.fillna(cpu)
    if t.isna().all():
        raise SystemExit("Could not infer time_ms from gpu_time_ms/cpu_time_ms (all NaN)")
    return t


def _fmt_seconds(s: float) -> str:
    if s >= 10:
        return f"{s:.1f}s"
    if s >= 1:
        return f"{s:.2f}s"
    return f"{s:.3f}s"


def _fmt_speedup(x: float) -> str:
    if x >= 10:
        return f"{x:.1f}×"
    return f"{x:.2f}×"


def main() -> int:
    args = parse_args()
    frames = [
        _load(Path(args.cpu), "CPU"),
        _load(Path(args.gpu1), "1 GPU"),
        _load(Path(args.gpu2), "2 GPU"),
    ]
    df = pd.concat(frames, ignore_index=True)

    # Normalize numeric fields used for grouping/labels.
    df["steps"] = pd.to_numeric(df.get("steps"), errors="coerce")
    df["paths"] = pd.to_numeric(df.get("paths"), errors="coerce")
    df["time_ms"] = _infer_time_ms(df)
    df["time_s"] = df["time_ms"] / 1000.0

    # Filter to avg_path rows if present.
    if "type" in df.columns:
        df = df[df["type"].astype(str) == "avg_path"].copy()

    # Keep only valid rows.
    df = df.dropna(subset=["steps", "paths", "time_s"])
    if df.empty:
        raise SystemExit("No usable rows after filtering (need steps/paths/time_ms).")

    # Aggregate: if multiple runs exist, use median time per config for a single (steps,paths) group.
    # If multiple distinct (steps,paths) exist, pick the most common group.
    df["steps"] = df["steps"].astype(int)
    df["paths"] = df["paths"].astype(int)

    grp_counts = (
        df.groupby(["steps", "paths"], as_index=False)
        .size()
        .sort_values("size", ascending=False)
        .reset_index(drop=True)
    )
    steps0 = int(grp_counts.loc[0, "steps"])
    paths0 = int(grp_counts.loc[0, "paths"])
    sub = df[(df["steps"] == steps0) & (df["paths"] == paths0)].copy()

    agg = (
        sub.groupby("config", as_index=False)
        .agg(time_s=("time_s", "median"))
        .sort_values("config")
        .reset_index(drop=True)
    )
    # Stable order.
    order = ["CPU", "1 GPU", "2 GPU"]
    agg["config"] = pd.Categorical(agg["config"], categories=order, ordered=True)
    agg = agg.sort_values("config").reset_index(drop=True)

    cpu_time = float(agg.loc[agg["config"] == "CPU", "time_s"].iloc[0])
    agg["speedup_vs_cpu"] = cpu_time / agg["time_s"]

    out_prefix = Path(args.out_prefix)
    out_runtime = out_prefix.with_name(out_prefix.name + "_runtime.png")
    out_speedup = out_prefix.with_name(out_prefix.name + "_speedup.png")
    out_summary = out_prefix.with_name(out_prefix.name + "_summary.csv")

    # Write summary table (for report tables).
    summary = agg.copy()
    summary.insert(1, "steps", steps0)
    summary.insert(2, "paths", paths0)
    summary.to_csv(out_summary, index=False)

    # Runtime bar
    plt.figure(figsize=(7.5, 4.6), dpi=180)
    ax = plt.gca()
    bars = ax.bar(
        summary["config"].astype(str),
        summary["time_s"],
        color=["#222222", "#1f77b4", "#ff7f0e"],
    )
    plt.title(f"{args.title}\n(paths={paths0:,}, steps={steps0})")
    plt.ylabel("time (s)")
    plt.grid(True, axis="y", alpha=0.25)
    if args.logy:
        plt.yscale("log")
        plt.ylabel("time (s, log scale)")
    ax.bar_label(bars, labels=[_fmt_seconds(float(v)) for v in summary["time_s"]], padding=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_runtime, bbox_inches="tight")

    # Speedup bar
    plt.figure(figsize=(7.5, 4.6), dpi=180)
    ax = plt.gca()
    bars = ax.bar(
        summary["config"].astype(str),
        summary["speedup_vs_cpu"],
        color=["#222222", "#1f77b4", "#ff7f0e"],
    )
    plt.title(f"Speedup vs CPU\n(paths={paths0:,}, steps={steps0})")
    plt.ylabel("speedup (×)")
    plt.grid(True, axis="y", alpha=0.25)
    ax.bar_label(bars, labels=[_fmt_speedup(float(v)) for v in summary["speedup_vs_cpu"]], padding=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_speedup, bbox_inches="tight")

    print(f"Wrote: {out_runtime}")
    print(f"Wrote: {out_speedup}")
    print(f"Wrote: {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


