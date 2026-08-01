"""Lightweight validation for regime_conditional signal.

This script:
1. Loads gold layer data (5 tickers, 2022-2025)
2. Computes features
3. Generates the new signal
4. Computes IC (information coefficient) against forward returns
5. Compares to existing signals

Usage:
    cd ~/working/alpha && uv run python test_regime_conditional.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

# Add project root to path
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

from src.features.pipeline import compute_features
from src.signals.base import compute_forward_returns, rank_ic
from src.signals.pipeline import generate_all


def load_gold_sample(tickers=None, year="2022"):
    """Load gold layer data for specified tickers."""
    gold_dir = root / "data" / "gold" / "daily" / f"year={year}"
    files = list(gold_dir.glob("ticker=*/part-0.parquet"))

    if tickers:
        files = [f for f in files if any(t in str(f) for t in tickers)]

    frames = []
    for f in sorted(files):
        df = pl.read_parquet(str(f))
        frames.append(df)

    if not frames:
        return pl.DataFrame()

    return pl.concat(frames, how="vertical_relaxed").sort(["ticker", "date"])


def main():
    print("=" * 60)
    print("Regime-Conditional Signal Validation")
    print("=" * 60)

    # Load data (5 tickers, 2022-2025)
    tickers = ["AAPL", "AMZN", "GOOGL", "META", "MSFT"]
    df = load_gold_sample(tickers=tickers, year="2022")

    if df.is_empty():
        print("No data available")
        return

    print(f"Loaded {len(df)} rows, {df['ticker'].n_unique()} tickers")

    # Compute features
    df = compute_features(df)
    print(f"Features computed: {len(df.columns)} columns")

    # Generate all signals
    df = generate_all(df)
    print(f"Signals generated: {[c for c in df.columns if c.startswith('signal_')]}")

    # Compute forward returns
    df = compute_forward_returns(df, horizons=[1, 5, 21])

    # Compute IC for each signal
    signal_cols = [c for c in df.columns if c.startswith("signal_")]
    print("\n" + "=" * 60)
    print("IC (Information Coefficient) by Signal")
    print("=" * 60)

    for signal_col in signal_cols:
        ic_1d = rank_ic(signal_col, "forward_return_1", df)
        ic_5d = rank_ic(signal_col, "forward_return_5", df)
        ic_21d = rank_ic(signal_col, "forward_return_21", df)

        print(f"{signal_col:40s} IC(1d): {ic_1d:+.4f}  IC(5d): {ic_5d:+.4f}  IC(21d): {ic_21d:+.4f}")

    # Highlight the new signal
    new_signal = "signal_regime_conditional_mr_21d"
    if new_signal in signal_cols:
        ic_1d = rank_ic(new_signal, "forward_return_1", df)
        ic_5d = rank_ic(new_signal, "forward_return_5", df)
        ic_21d = rank_ic(new_signal, "forward_return_21", df)

        print("\n" + "=" * 60)
        print(f"New Signal: {new_signal}")
        print("=" * 60)
        print(f"IC(1d):  {ic_1d:+.4f}")
        print(f"IC(5d):  {ic_5d:+.4f}")
        print(f"IC(21d): {ic_21d:+.4f}")

        # Compare to existing mean reversion signal
        existing_signal = "signal_mean_reversion_21d"
        if existing_signal in signal_cols:
            ic_existing_1d = rank_ic(existing_signal, "forward_return_1", df)
            ic_existing_5d = rank_ic(existing_signal, "forward_return_5", df)
            ic_existing_21d = rank_ic(existing_signal, "forward_return_21", df)

            print(f"\nComparison to {existing_signal}:")
            print(f"IC(1d):  {ic_existing_1d:+.4f} -> {ic_1d:+.4f} ({ic_1d - ic_existing_1d:+.4f})")
            print(f"IC(5d):  {ic_existing_5d:+.4f} -> {ic_5d:+.4f} ({ic_5d - ic_existing_5d:+.4f})")
            print(f"IC(21d): {ic_existing_21d:+.4f} -> {ic_21d:+.4f} ({ic_21d - ic_existing_21d:+.4f})")


if __name__ == "__main__":
    main()