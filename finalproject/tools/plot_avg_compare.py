import argparse
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import pandas as pd


class Series(NamedTuple):
    label: str
    path: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Overlay avg-path CSVs (step,t_years,mean,std) for CPU vs GPU comparisons."
    )
    p.add_argument("--cpu", default="avg_2330_cpu.csv", help="CPU avg-path CSV (default: avg_2330_cpu.csv)")
    p.add_argument("--gpu1", default="avg_2330_1gpu.csv", help="1-GPU avg-path CSV (default: avg_2330_1gpu.csv)")
    p.add_argument("--gpu2", default="avg_2330_2gpu.csv", help="2-GPU avg-path CSV (default: avg_2330_2gpu.csv)")
    p.add_argument("--out", default="avg_2330_compare.png", help="Output PNG path (default: avg_2330_compare.png)")
    p.add_argument("--title", default="2330 avg-path (CPU vs 1GPU vs 2GPU)", help="Plot title")
    p.add_argument("--x-months", action="store_true", help="Use months (0..12) on x-axis instead of years")
    p.add_argument(
        "--show-std",
        action="store_true",
        help="Draw mean±1σ band (shaded) for each series (can look busy).",
    )
    p.add_argument("--std-alpha", type=float, default=0.10, help="Alpha for std band fill (default: 0.10)")
    return p.parse_args()


def _load_avg(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("step", "t_years", "mean", "std"):
        if col not in df.columns:
            raise SystemExit(f"{path}: missing required column '{col}'")
    df = df[["step", "t_years", "mean", "std"]].copy()
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["t_years"] = pd.to_numeric(df["t_years"], errors="coerce")
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
    df["std"] = pd.to_numeric(df["std"], errors="coerce")
    if df.isna().any().any():
        raise SystemExit(f"{path}: contains NaNs after numeric conversion")
    return df


def main() -> int:
    args = parse_args()

    series = [
        Series("CPU (OpenMP)", Path(args.cpu)),
        Series("1 GPU", Path(args.gpu1)),
        Series("2 GPU", Path(args.gpu2)),
    ]

    loaded: list[tuple[Series, pd.DataFrame]] = []
    for s in series:
        if not s.path.exists():
            raise SystemExit(f"File not found: {s.path}")
        loaded.append((s, _load_avg(s.path)))

    # Align on step to ensure we plot comparable x values.
    base_steps = loaded[0][1]["step"].to_numpy()
    for s, df in loaded[1:]:
        if df["step"].to_numpy().shape != base_steps.shape or (df["step"].to_numpy() != base_steps).any():
            raise SystemExit(f"Step grid mismatch vs CPU file: {s.path}")

    t_years = loaded[0][1]["t_years"].to_numpy()
    x = (t_years * 12.0) if args.x_months else t_years

    plt.figure(figsize=(10, 5.5), dpi=180)
    colors = {
        "CPU (OpenMP)": "#222222",
        "1 GPU": "#1f77b4",
        "2 GPU": "#ff7f0e",
    }

    for s, df in loaded:
        mean = df["mean"].to_numpy()
        std = df["std"].to_numpy()
        c = colors.get(s.label, None)
        plt.plot(x, mean, linewidth=2.0, label=s.label, color=c)
        if args.show_std:
            plt.fill_between(x, mean - std, mean + std, alpha=args.std_alpha, color=c)

    plt.title(args.title)
    plt.xlabel("t (months)" if args.x_months else "t (years)")
    plt.ylabel("price")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    print(f"Wrote plot -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


