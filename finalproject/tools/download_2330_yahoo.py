import argparse
import datetime as dt
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf


TICKER = "2330.TW"  # 固定查詢台積電（台股）


@dataclass(frozen=True)
class VolResult:
    sigma_daily: float
    sigma_annualized: float
    n_returns: int


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download Yahoo Finance daily history for 2330.TW and save CSV; optionally estimate annualized volatility.",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--years", type=float, default=5.0, help="Lookback years from today (default: 5).")
    g.add_argument("--start", type=str, help="Start date YYYY-MM-DD (overrides --years).")
    p.add_argument("--end", type=str, help="End date YYYY-MM-DD (default: today).")
    p.add_argument("--out", type=str, default=os.path.join("data", "2330_TW.csv"), help="Output CSV path.")
    p.add_argument(
        "--adj",
        action="store_true",
        help="Use Adj Close for volatility estimation (default: Close).",
    )
    p.add_argument(
        "--trading-days",
        type=int,
        default=252,
        help="Trading days per year for annualization (default: 252).",
    )
    p.add_argument(
        "--no-vol",
        action="store_true",
        help="Skip volatility estimation; only download CSV.",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=8,
        help="Retry count when Yahoo rate-limits (default: 8).",
    )
    p.add_argument(
        "--retry-sleep",
        type=float,
        default=10.0,
        help="Initial sleep seconds between retries (exponential backoff) (default: 10).",
    )
    return p.parse_args()


def _parse_date(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()


def _date_range(args: argparse.Namespace) -> tuple[dt.date, dt.date]:
    end = _parse_date(args.end) if args.end else dt.date.today()
    if args.start:
        start = _parse_date(args.start)
    else:
        # years can be float; convert to days approximation.
        days = int(round(float(args.years) * 365.25))
        start = end - dt.timedelta(days=days)
    if start >= end:
        raise ValueError(f"start ({start}) must be earlier than end ({end})")
    return start, end


def download_2330_daily(start: dt.date, end: dt.date) -> pd.DataFrame:
    # yfinance end is exclusive for some intervals; add +1 day to include end date.
    end_inclusive = end + dt.timedelta(days=1)
    df = yf.download(
        TICKER,
        start=start.isoformat(),
        end=end_inclusive.isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if df is None or df.empty:
        raise RuntimeError("No data returned from Yahoo Finance. Check network / ticker / date range.")

    # Normalize index and columns.
    if isinstance(df.columns, pd.MultiIndex):
        # Newer yfinance sometimes returns (Field, Ticker) columns.
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df


def _looks_rate_limited(err: BaseException) -> bool:
    s = repr(err)
    return ("YFRateLimitError" in s) or ("Too Many Requests" in s) or ("rate limit" in s.lower())


def estimate_volatility(df: pd.DataFrame, use_adj: bool, trading_days: int) -> VolResult:
    col = "Adj Close" if use_adj and "Adj Close" in df.columns else "Close"
    if col not in df.columns:
        raise ValueError(f"Expected column '{col}' in downloaded data.")
    px = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(px) < 2:
        raise ValueError("Not enough price points to compute returns.")
    r = np.log(px).diff().dropna()
    sigma_daily = float(r.std(ddof=1))
    sigma_annual = float(sigma_daily * np.sqrt(trading_days))
    return VolResult(sigma_daily=sigma_daily, sigma_annualized=sigma_annual, n_returns=int(r.shape[0]))


def main() -> int:
    args = _parse_args()
    start, end = _date_range(args)

    attempt = 0
    sleep_s = float(args.retry_sleep)
    last_err: Exception | None = None
    while True:
        try:
            df = download_2330_daily(start, end)
            break
        except Exception as e:
            last_err = e
            attempt += 1
            if attempt > int(args.retries) or not _looks_rate_limited(e):
                raise
            print(f"Yahoo rate limited; retrying in {sleep_s:.1f}s (attempt {attempt}/{args.retries})...")
            time.sleep(sleep_s)
            sleep_s = min(sleep_s * 2.0, 300.0)

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Downloaded {TICKER} daily data: rows={len(df)}  range={df['Date'].min()}..{df['Date'].max()}")
    print(f"Saved CSV -> {out_path}")

    if not args.no_vol:
        v = estimate_volatility(df, use_adj=args.adj, trading_days=args.trading_days)
        basis = "Adj Close" if args.adj else "Close"
        print(f"Volatility estimate ({basis}, log returns):")
        print(f"  sigma_daily      = {v.sigma_daily:.6f}")
        print(f"  sigma_annualized = {v.sigma_annualized:.6f}  (trading_days={args.trading_days}, n_returns={v.n_returns})")
        print("Tip: you can pass sigma_annualized into mc_pricer as --sigma.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


