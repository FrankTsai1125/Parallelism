import argparse

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate Monte Carlo path CSV (path_id,step,t_years,price) into daily mean/std and percentile bands."
    )
    p.add_argument("--in", dest="inp", required=True, help="Input paths CSV, e.g. paths_2330_1y.csv")
    p.add_argument("--out", dest="out", required=True, help="Output band CSV, e.g. band_2330_1y.csv")
    p.add_argument(
        "--percentiles",
        default="5,50,95",
        help="Comma-separated percentiles to output (default: 5,50,95)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ps = [float(x.strip()) for x in args.percentiles.split(",") if x.strip()]
    if any(p < 0 or p > 100 for p in ps):
        raise SystemExit("percentiles must be between 0 and 100")

    df = pd.read_csv(args.inp)
    for col in ("step", "t_years", "price"):
        if col not in df.columns:
            raise SystemExit(f"missing required column: {col}")

    g = df.groupby("step", sort=True)
    out = pd.DataFrame(
        {
            "step": g["step"].first(),
            "t_years": g["t_years"].first(),
            "mean": g["price"].mean(),
            "std": g["price"].std(ddof=1),
        }
    ).reset_index(drop=True)

    for p in ps:
        key = f"p{int(p) if float(p).is_integer() else str(p).replace('.', '_')}"
        out[key] = g["price"].quantile(p / 100.0).values

    out.to_csv(args.out, index=False)
    print(f"Wrote band CSV -> {args.out}  (rows={len(out)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


