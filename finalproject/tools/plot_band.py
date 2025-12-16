import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot mean line and percentile band from band_*.csv")
    p.add_argument("--in", dest="inp", required=True, help="Input band CSV (step,t_years,mean,std,p5,p50,p95,...)")
    p.add_argument("--out", dest="out", default="", help="Output PNG path (default: <in>.png)")
    p.add_argument("--title", default="2330 1Y Monte Carlo Band", help="Plot title")
    p.add_argument("--show-std", action="store_true", help="Also draw mean ± 1 std band")
    p.add_argument("--x-months", action="store_true", help="Use months (0..12) on x-axis instead of years")
    p.add_argument("--annotate-final", action="store_true", help="Annotate year-end interval on the plot")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    inp = Path(args.inp)
    out = Path(args.out) if args.out else inp.with_suffix(".png")

    df = pd.read_csv(inp)
    for col in ("t_years", "mean"):
        if col not in df.columns:
            raise SystemExit(f"missing required column: {col}")

    t_years = df["t_years"].to_numpy()
    x = (t_years * 12.0) if args.x_months else t_years
    mean = df["mean"].to_numpy()

    plt.figure(figsize=(10, 5.5), dpi=160)

    # Percentile band (prefer p5/p95 if present)
    if "p5" in df.columns and "p95" in df.columns:
        p5 = df["p5"].to_numpy()
        p95 = df["p95"].to_numpy()
        plt.fill_between(x, p5, p95, alpha=0.25, label="p5–p95")

    # Optional std band
    if args.show_std and "std" in df.columns:
        std = df["std"].to_numpy()
        plt.fill_between(x, mean - std, mean + std, alpha=0.15, label="mean±1σ")

    # Median if present
    if "p50" in df.columns:
        plt.plot(x, df["p50"].to_numpy(), linewidth=1.2, label="median (p50)")

    plt.plot(x, mean, linewidth=2.0, label="mean")

    plt.title(args.title)
    plt.xlabel("t (months)" if args.x_months else "t (years)")
    plt.ylabel("price")
    plt.grid(True, alpha=0.25)
    plt.legend()

    if args.annotate_final:
        last = df.iloc[-1]
        parts = []
        if "p5" in df.columns and "p95" in df.columns:
            parts.append(f"Year-end p5–p95: [{last['p5']:.1f}, {last['p95']:.1f}]")
        if "p50" in df.columns:
            parts.append(f"median(p50): {last['p50']:.1f}")
        parts.append(f"mean: {last['mean']:.1f}")
        if "std" in df.columns:
            parts.append(f"std: {last['std']:.1f}")
        txt = "\n".join(parts)

        plt.gca().text(
            0.02,
            0.98,
            txt,
            transform=plt.gca().transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="0.7"),
        )
    plt.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    print(f"Wrote plot -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


