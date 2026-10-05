"""Drawdown analysis of a daily price series.

Pipeline: load a market-data CSV -> compute drawdown metrics -> save a plot.

The drawdown at day t is the percentage loss from the highest close seen
up to t (the running peak). The *maximum* drawdown is the worst such loss
over the whole period: the largest peak-to-trough decline an investor
holding the asset would have lived through.

Run from anywhere:
    python src/analysis.py
    python src/analysis.py --input data/raw/prices.csv --output outputs/plot.png
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Paths resolve from the repo root, so the script works whatever the current directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "prices.csv"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "plot.png"

REQUIRED_COLUMNS = {"date", "close"}


@dataclass(frozen=True)
class DrawdownStats:
    """Summary of the worst peak-to-trough episode."""

    max_drawdown: float  # negative fraction, e.g. -0.12 for -12%
    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    recovery_date: pd.Timestamp | None  # None if the old peak was never regained
    total_return: float


def load_prices(path: Path) -> pd.DataFrame:
    """Read the CSV, validate it and return rows sorted by date."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path, parse_dates=["date"])

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing column(s): {sorted(missing)}")

    df = (
        df.dropna(subset=["date", "close"])
        .drop_duplicates(subset="date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    if len(df) < 2:
        raise ValueError("Need at least two valid rows to compute returns.")
    if (df["close"] <= 0).any():
        raise ValueError("Close prices must be strictly positive.")
    return df


def compute_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily return, cumulative return, running peak and drawdown columns."""
    out = df.copy()
    out["daily_return"] = out["close"].pct_change()
    out["cum_return"] = out["close"] / out["close"].iloc[0] - 1
    out["running_peak"] = out["close"].cummax()
    out["drawdown"] = out["close"] / out["running_peak"] - 1  # <= 0 by construction
    return out


def summarise(df: pd.DataFrame) -> DrawdownStats:
    """Locate the maximum drawdown and the peak / trough / recovery dates around it."""
    trough_idx = df["drawdown"].idxmin()
    # The peak is the last date before the trough where the price sat at its running high.
    peak_idx = df.loc[:trough_idx, "close"].idxmax()
    peak_price = df.at[peak_idx, "close"]

    after_trough = df.loc[trough_idx:]
    recovered = after_trough[after_trough["close"] >= peak_price]
    recovery_date = recovered["date"].iloc[0] if not recovered.empty else None

    return DrawdownStats(
        max_drawdown=float(df.at[trough_idx, "drawdown"]),
        peak_date=df.at[peak_idx, "date"],
        trough_date=df.at[trough_idx, "date"],
        recovery_date=recovery_date,
        total_return=float(df["cum_return"].iloc[-1]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute and plot the drawdown of a price series.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="CSV with 'date' and 'close' columns (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Where to save the chart (default: %(default)s)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prices = load_prices(args.input)
    metrics = compute_drawdown(prices)
    stats = summarise(metrics)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output.with_name("drawdown.csv")
    metrics.to_csv(metrics_path, index=False)

    recovery = f"{stats.recovery_date:%Y-%m-%d}" if stats.recovery_date else "not recovered"
    print(f"Rows analysed : {len(metrics)} ({metrics['date'].iloc[0]:%Y-%m-%d} -> "
          f"{metrics['date'].iloc[-1]:%Y-%m-%d})")
    print(f"Total return  : {stats.total_return:+.2%}")
    print(f"Max drawdown  : {stats.max_drawdown:.2%}  "
          f"(peak {stats.peak_date:%Y-%m-%d}, trough {stats.trough_date:%Y-%m-%d}, "
          f"recovery {recovery})")
    print(f"Saved metrics : {metrics_path}")


if __name__ == "__main__":
    main()
