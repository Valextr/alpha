#!/usr/bin/env python
"""Lightweight IC comparison: CS vs TS signals on 2025 data.

Only computes required features to avoid OOM.
"""

import polars as pl
from datetime import date
from src.data.config import get_config
from src.signals.base import rank_ic, compute_forward_returns

def load_2025_data():
    """Load just 2025 gold data for original 5 tickers."""
    config = get_config()
    gold_dir = config.data_dir / "gold" / "daily"
    files = sorted(gold_dir.rglob("*.parquet"))
    small_files = [f for f in files if any(t in str(f) for t in ["AAPL", "MSFT", "GOOGL", "AMZN", "META"])]
    df = pl.concat([pl.read_parquet(str(f)) for f in small_files], how="vertical_relaxed")
    df = df.filter((pl.col("date") >= date(2025, 1, 1)) & (pl.col("date") <= date(2025, 12, 31)))
    return df.sort(["ticker", "date"])

# Load data
print("Loading 2025 data...")
df = load_2025_data()
print(f"Data: {len(df)} rows, {df['ticker'].n_unique()} tickers")

# Compute only required features for TS signals
print("Computing minimal features...")
df = df.with_columns(
    (pl.col("close").shift(1).over("ticker")).alias("_prev_close")
)
df = df.with_columns(
    ((pl.col("close") / pl.col("_prev_close") - 1)).alias("return_1d")
)
df = df.with_columns(
    (pl.col("close").log() - pl.col("close").log().shift(1).over("ticker")).alias("log_return_1d")
)
df = df.with_columns(
    (pl.col("close").shift(21).over("ticker")).alias("_prev_21d")
)
df = df.with_columns(
    ((pl.col("close") / pl.col("_prev_21d") - 1)).alias("return_21d")
)
df = df.with_columns(
    (pl.col("close").shift(63).over("ticker")).alias("_prev_63d")
)
df = df.with_columns(
    ((pl.col("close") / pl.col("_prev_63d") - 1)).alias("return_63d")
)
df = df.with_columns(
    pl.col("return_1d").rolling_std(window_size=21, min_samples=10).over("ticker").alias("vol_21d")
)
df = df.with_columns(
    pl.col("return_1d").rolling_std(window_size=63, min_samples=30).over("ticker").alias("vol_63d")
)
df = df.drop(["_prev_close", "_prev_21d", "_prev_63d"])

# Generate TS signals directly
print("Generating TS signals...")
from src.signals.time_series import (
    generate_mean_reversion_ts_21d,
    generate_mean_reversion_ts_63d,
    generate_momentum_ts_21d,
    generate_momentum_ts_63d,
)

df = generate_mean_reversion_ts_21d(df)
df = generate_mean_reversion_ts_63d(df)
df = generate_momentum_ts_21d(df)
df = generate_momentum_ts_63d(df)

# Generate CS signals for comparison
print("Generating CS signals...")
from src.signals.mean_reversion import generate_mean_reversion_21d, generate_mean_reversion_63d
from src.signals.momentum import generate_momentum_21d, generate_momentum_63d

df = generate_mean_reversion_21d(df)
df = generate_mean_reversion_63d(df)
df = generate_momentum_21d(df)
df = generate_momentum_63d(df)

# Compute forward returns
print("Computing forward returns...")
df = compute_forward_returns(df, horizons=[5])

# Compare IC
cs_signals = ["signal_mean_reversion_21d", "signal_mean_reversion_63d",
              "signal_momentum_21d", "signal_momentum_63d"]
ts_signals = ["signal_mean_reversion_ts_21d", "signal_mean_reversion_ts_63d",
              "signal_momentum_ts_21d", "signal_momentum_ts_63d"]

print("\nIC comparison (5-day horizon):")
print(f"{'Signal':<35} {'IC':>8}")
print("-" * 45)

cs_ics = []
for sig in cs_signals:
    ic = rank_ic(sig, "forward_return_5", df)
    cs_ics.append(ic)
    print(f"{sig:<35} {ic:>8.4f} (CS)")

ts_ics = []
for sig in ts_signals:
    ic = rank_ic(sig, "forward_return_5", df)
    ts_ics.append(ic)
    print(f"{sig:<35} {ic:>8.4f} (TS)")

print(f"\nMean CS IC: {sum(cs_ics)/len(cs_ics):.4f}")
print(f"Mean TS IC: {sum(ts_ics)/len(ts_ics):.4f}")
print(f"Improvement: {(sum(ts_ics)/len(ts_ics) - sum(cs_ics)/len(cs_ics)) / abs(sum(cs_ics)/len(cs_ics)) * 100:+.1f}%")