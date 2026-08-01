"""Signal matrix tester.

Tests all signal combinations across all timeframes, horizons, and tickers.
Outputs a comprehensive IC matrix for deliberation.

Usage:
    uv run python scripts/test_signal_matrix.py
"""

from __future__ import annotations

import glob
import logging
from datetime import date

import polars as pl
from scipy.stats import spearmanr

from src.features.intraday import compute_intraday_features
from src.signals.base import compute_forward_returns
from src.signals.vwap_reversion import generate_vwap_mean_reversion_60b
from src.signals.vwap_bands import generate_vwap_bands
from src.signals.opening_range import generate_opening_range_breakout

log = logging.getLogger(__name__)

# Configuration
TICKERS = ["MSFT", "GOOGL", "AAPL", "AMZN", "META", "QQQ", "INTC"]
TIMEFRAMES = ["15m", "30m", "1h"]
HORIZONS = [5, 20]
START_DATE = date(2025, 1, 1)
END_DATE = date(2025, 12, 31)


def load_and_resample(ticker: str, timeframe: str) -> pl.DataFrame:
    """Load 1-min data for a ticker and resample to target timeframe."""
    data_dir = f"data/intraday/1min/ticker={ticker}"
    dfs = []
    for f in glob.glob(f"{data_dir}/year=2025/month=*/*.parquet"):
        df = pl.read_parquet(f)
        dfs.append(df)

    if not dfs:
        return pl.DataFrame()

    all_data = pl.concat(dfs, how="vertical_relaxed").sort("datetime")

    # Filter date range
    all_data = all_data.filter(
        (pl.col("datetime").dt.date() >= START_DATE)
        & (pl.col("datetime").dt.date() <= END_DATE)
    )

    # Filter regular session (9:30 AM - 4:00 PM ET = 14:30 - 21:00 UTC)
    all_data = all_data.with_columns(
        pl.col("datetime").dt.hour().alias("hour"),
        pl.col("datetime").dt.minute().alias("minute"),
    ).filter(
        ((pl.col("hour") >= 14) & (pl.col("hour") < 21))
        | ((pl.col("hour") == 14) & (pl.col("minute") >= 30))
    )

    # Resample
    bars = all_data.group_by_dynamic(
        "datetime", every=timeframe, include_boundaries=True
    ).agg(
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
        pl.col("volume").sum(),
        pl.col("ticker").first(),
    )

    return bars


def generate_signals(enriched: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Generate all individual signals and their combinations."""
    signals = {}

    # Individual signals
    try:
        vwap_mr = generate_vwap_mean_reversion_60b(enriched.clone())
        signals["vwap_mr"] = vwap_mr
    except Exception as e:
        log.warning(f"VWAP MR failed: {e}")

    try:
        vwap_bands = generate_vwap_bands(enriched.clone())
        signals["vwap_bands"] = vwap_bands
    except Exception as e:
        log.warning(f"VWAP Bands failed: {e}")

    try:
        orb = generate_opening_range_breakout(enriched.clone())
        signals["orb"] = orb
    except Exception as e:
        log.warning(f"ORB failed: {e}")

    # Pair combinations (equal-weight average of normalized signals)
    if "vwap_mr" in signals and "vwap_bands" in signals:
        s1 = signals["vwap_mr"]["signal_vwap_mean_reversion_60b"]
        s2 = signals["vwap_bands"]["signal_vwap_bands"]
        signals["vwap_mr+vwap_bands"] = signals["vwap_mr"].with_columns(
            ((s1 + s2) / 2).alias("signal_combined")
        )

    if "vwap_mr" in signals and "orb" in signals:
        s1 = signals["vwap_mr"]["signal_vwap_mean_reversion_60b"]
        s2 = signals["orb"]["signal_opening_range_breakout"]
        signals["vwap_mr+orb"] = signals["vwap_mr"].with_columns(
            ((s1 + s2) / 2).alias("signal_combined")
        )

    if "vwap_bands" in signals and "orb" in signals:
        s1 = signals["vwap_bands"]["signal_vwap_bands"]
        s2 = signals["orb"]["signal_opening_range_breakout"]
        signals["vwap_bands+orb"] = signals["vwap_bands"].with_columns(
            ((s1 + s2) / 2).alias("signal_combined")
        )

    # Triple combination
    if all(k in signals for k in ["vwap_mr", "vwap_bands", "orb"]):
        s1 = signals["vwap_mr"]["signal_vwap_mean_reversion_60b"]
        s2 = signals["vwap_bands"]["signal_vwap_bands"]
        s3 = signals["orb"]["signal_opening_range_breakout"]
        signals["all_three"] = signals["vwap_mr"].with_columns(
            ((s1 + s2 + s3) / 3).alias("signal_combined")
        )

    return signals


def compute_ic(df: pl.DataFrame, signal_col: str, horizon: int) -> tuple[float, float]:
    """Compute IC (Spearman rank correlation) between signal and forward returns."""
    fwd_col = f"forward_return_{horizon}"
    if fwd_col not in df.columns:
        return float("nan"), float("nan")

    valid = df.drop_nulls(subset=[signal_col, fwd_col])
    if len(valid) < 100:
        return float("nan"), float("nan")

    ic, p = spearmanr(valid[signal_col].to_list(), valid[fwd_col].to_list())
    return float(ic), float(p)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Map signal names to their column names
    signal_cols = {
        "vwap_mr": "signal_vwap_mean_reversion_60b",
        "vwap_bands": "signal_vwap_bands",
        "orb": "signal_opening_range_breakout",
        "vwap_mr+vwap_bands": "signal_combined",
        "vwap_mr+orb": "signal_combined",
        "vwap_bands+orb": "signal_combined",
        "all_three": "signal_combined",
    }

    # Results storage
    results = {}

    for ticker in TICKERS:
        print(f"\n{'=' * 60}")
        print(f"  {ticker}")
        print(f"{'=' * 60}")

        for tf in TIMEFRAMES:
            bars = load_and_resample(ticker, tf)
            if bars.is_empty:
                continue

            enriched = compute_intraday_features(bars)
            signals = generate_signals(enriched)

            for sig_name, sig_df in signals.items():
                sig_col = signal_cols[sig_name]
                if sig_col not in sig_df.columns:
                    continue

                # Compute forward returns
                with_fwd = compute_forward_returns(sig_df, horizons=HORIZONS)

                for horizon in HORIZONS:
                    ic, p = compute_ic(with_fwd, sig_col, horizon)
                    key = (ticker, tf, sig_name, horizon)
                    results[key] = (ic, p)

    # Print summary tables
    print(f"\n{'#' * 70}")
    print(f"  SIGNAL MATRIX RESULTS — Full Year 2025")
    print(f"{'#' * 70}")

    # Group by timeframe
    for tf in TIMEFRAMES:
        print(f"\n{'=' * 70}")
        print(f"  {tf} bars")
        print(f"{'=' * 70}")

        for horizon in HORIZONS:
            print(f"\n  {horizon}-bar horizon:")
            print(f"  {'Ticker':<8} {'Signal':<22} {'IC':>10} {'p-value':>10}")
            print(f"  {'-' * 52}")

            for ticker in TICKERS:
                for sig_name in signal_cols:
                    key = (ticker, tf, sig_name, horizon)
                    if key in results:
                        ic, p = results[key]
                        print(f"  {ticker:<8} {sig_name:<22} {ic:>+10.4f} {p:>10.4f}")

    # Find best combinations
    print(f"\n{'#' * 70}")
    print(f"  BEST COMBINATIONS (by average IC across all tickers)")
    print(f"{'#' * 70}")

    for tf in TIMEFRAMES:
        for horizon in HORIZONS:
            print(f"\n  {tf} / {horizon}-bar:")
            avg_ics = []
            for sig_name in signal_cols:
                ics = [
                    results[(t, tf, sig_name, horizon)][0]
                    for t in TICKERS
                    if (t, tf, sig_name, horizon) in results
                ]
                if ics:
                    avg = sum(ics) / len(ics)
                    avg_ics.append((sig_name, avg))
            avg_ics.sort(key=lambda x: x[1], reverse=True)
            for sig_name, avg in avg_ics:
                print(f"    {sig_name:<22} avg IC = {avg:>+10.4f}")


if __name__ == "__main__":
    main()