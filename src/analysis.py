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

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: write to file, never open a window
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

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


def plot(df: pd.DataFrame, stats: DrawdownStats, out_path: Path) -> None:
    """Two panels: price vs running peak on top, 'underwater' drawdown chart below."""
    fig, (ax_price, ax_dd) = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    # Top: close and running peak, with the worst episode shaded.
    ax_price.plot(df["date"], df["close"], color="#1D2A4B", linewidth=1.3, label="Close")
    ax_price.plot(
        df["date"], df["running_peak"], color="#7CB9E8", linewidth=1.3,
        linestyle="--", label="Running peak",
    )
    episode_end = stats.recovery_date or df["date"].iloc[-1]
    ax_price.axvspan(stats.peak_date, episode_end, color="#C0392B", alpha=0.08,
                     label="Max-drawdown episode")
    ax_price.set_ylabel("Price")
    ax_price.set_title("Daily close and drawdown from running peak")
    ax_price.legend(loc="upper left", frameon=False)
    ax_price.grid(True, alpha=0.3)

    # Bottom: underwater chart with the trough annotated.
    ax_dd.fill_between(df["date"], df["drawdown"], 0, color="#C0392B", alpha=0.35)
    ax_dd.plot(df["date"], df["drawdown"], color="#C0392B", linewidth=1)
    ax_dd.annotate(
        f"Max drawdown {stats.max_drawdown:.1%}\n{stats.trough_date:%d %b %Y}",
        xy=(stats.trough_date, stats.max_drawdown),
        xytext=(45, -18), textcoords="offset points", fontsize=9, va="center",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#CCCCCC"},
        arrowprops={"arrowstyle": "->", "color": "#555555"},
    )
    ax_dd.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax_dd.set_ylim(stats.max_drawdown * 1.5, 0.005)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_xlabel("Date")
    ax_dd.grid(True, alpha=0.3)
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


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

    plot(metrics, stats, args.output)
    metrics_path = args.output.with_name("drawdown.csv")
    metrics.to_csv(metrics_path, index=False)

    recovery = f"{stats.recovery_date:%Y-%m-%d}" if stats.recovery_date else "not recovered"
    print(f"Rows analysed : {len(metrics)} ({metrics['date'].iloc[0]:%Y-%m-%d} -> "
          f"{metrics['date'].iloc[-1]:%Y-%m-%d})")
    print(f"Total return  : {stats.total_return:+.2%}")
    print(f"Max drawdown  : {stats.max_drawdown:.2%}  "
          f"(peak {stats.peak_date:%Y-%m-%d}, trough {stats.trough_date:%Y-%m-%d}, "
          f"recovery {recovery})")
    print(f"Saved plot    : {args.output}")
    print(f"Saved metrics : {metrics_path}")


if __name__ == "__main__":
    main()
