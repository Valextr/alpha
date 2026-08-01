"""Test regime_conditional signal in the ensemble pipeline.

This script:
1. Loads gold layer data (5 tickers, 2022-2025)
2. Computes features
3. Generates all signals
4. Computes forward returns
5. Computes IC for each signal
6. Computes ensemble IC with and without the new signal
7. Compares the results

Usage:
    cd ~/working/alpha && uv run python test_ensemble_ic.py
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
from src.ensemble.ic_weighted import compute_rolling_ic, ICWeightedEnsemble
from src.ensemble.base import EnsembleConfig


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
    print("Ensemble IC Test: regime_conditional_mr_21d")
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

    # Define signal sets
    existing_signals = [
        "signal_mean_reversion_21d",
        "signal_mean_reversion_63d",
        "signal_momentum_21d",
        "signal_momentum_63d",
    ]

    new_signal = "signal_regime_conditional_mr_21d"

    # Check if the new signal exists
    if new_signal not in df.columns:
        print(f"\nNew signal {new_signal} not found in DataFrame")
        return

    # Compute IC for each signal
    print("\n" + "=" * 60)
    print("IC (Information Coefficient) by Signal")
    print("=" * 60)

    for sig in existing_signals + [new_signal]:
        ic_1d = rank_ic(sig, "forward_return_1", df)
        ic_5d = rank_ic(sig, "forward_return_5", df)
        ic_21d = rank_ic(sig, "forward_return_21", df)

        print(f"{sig:40s} IC(1d): {ic_1d:+.4f}  IC(5d): {ic_5d:+.4f}  IC(21d): {ic_21d:+.4f}")

    # Compute ensemble IC
    print("\n" + "=" * 60)
    print("Ensemble IC Test")
    print("=" * 60)

    # Existing ensemble
    ensemble_existing = ICWeightedEnsemble(ic_window=63)
    df_existing = ensemble_existing.transform(df, existing_signals, "forward_return_1")
    ic_existing = rank_ic("ensemble_score", "forward_return_1", df_existing)

    # New ensemble (with regime_conditional signal)
    ensemble_new = ICWeightedEnsemble(ic_window=63)
    df_new = ensemble_new.transform(df, existing_signals + [new_signal], "forward_return_1")
    ic_new = rank_ic("ensemble_score", "forward_return_1", df_new)

    print(f"\nExisting ensemble IC: {ic_existing:+.4f}")
    print(f"New ensemble IC:      {ic_new:+.4f}")
    print(f"Improvement:          {ic_new - ic_existing:+.4f} ({(ic_new - ic_existing) / abs(ic_existing) * 100:+.1f}%)")

    # Check signal weights
    print("\n" + "=" * 60)
    print("Signal Weights (last rebalance)")
    print("=" * 60)

    for sig in existing_signals + [new_signal]:
        wcol = f"w_{sig}"
        if wcol in df_new.columns:
            last_weight = df_new[wcol].drop_nulls()[-1]
            print(f"{sig:40s} weight: {last_weight:.4f}")


if __name__ == "__main__":
    main()