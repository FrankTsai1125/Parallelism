import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot a single avg-path CSV (step,t_years,mean,std) to PNG.")
    p.add_argument("--in", dest="inp", required=True, help="Input avg-path CSV (step,t_years,mean,std)")
    p.add_argument("--out", dest="out", default="", help="Output PNG path (default: <in>.png)")
    p.add_argument("--title", default="", help="Plot title (default: derived from filename)")
    p.add_argument("--x-months", action="store_true", help="Use months (0..12) on x-axis instead of years")
    p.add_argument("--show-std", action="store_true", help="Draw mean±1σ band (shaded)")
    p.add_argument("--std-alpha", type=float, default=0.12, help="Alpha for std band fill (default: 0.12)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    inp = Path(args.inp)
    if not inp.exists():
        raise SystemExit(f"File not found: {inp}")
    out = Path(args.out) if args.out else inp.with_suffix(".png")

    df = pd.read_csv(inp)
    for col in ("step", "t_years", "mean", "std"):
        if col not in df.columns:
            raise SystemExit(f"{inp}: missing required column '{col}'")

    t_years = pd.to_numeric(df["t_years"], errors="coerce")
    mean = pd.to_numeric(df["mean"], errors="coerce")
    std = pd.to_numeric(df["std"], errors="coerce")
    if t_years.isna().any() or mean.isna().any() or std.isna().any():
        raise SystemExit(f"{inp}: contains NaNs after numeric conversion")

    x = (t_years.to_numpy() * 12.0) if args.x_months else t_years.to_numpy()

    title = args.title.strip() if args.title.strip() else inp.stem
    plt.figure(figsize=(10, 5.5), dpi=180)
    plt.plot(x, mean.to_numpy(), linewidth=2.0, label="mean")
    if args.show_std:
        m = mean.to_numpy()
        s = std.to_numpy()
        plt.fill_between(x, m - s, m + s, alpha=args.std_alpha, label="mean±1σ")
    plt.title(title)
    plt.xlabel("t (months)" if args.x_months else "t (years)")
    plt.ylabel("price")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    print(f"Wrote plot -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


