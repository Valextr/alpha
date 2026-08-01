#!/usr/bin/env python
"""Phase 9: Signal IC comparison — fast version (skip Hurst, use cached features)."""

import sys
import time

import polars as pl

t0 = time.time()
print("Loading data (2020-2026)...", file=sys.stderr)
df = (
    pl.scan_parquet("data/gold/daily/**/*.parquet")
    .filter(pl.col("date") >= pl.date(2020, 1, 1))
    .sort(["ticker", "date"])
    .collect()
)
n_tickers = df["ticker"].unique().len()
print(f"Universe: {n_tickers} tickers, {len(df)} rows ({time.time()-t0:.1f}s)", file=sys.stderr)

# Compute only fast features (skip advanced: Hurst, Kalman, frac_diff)
t0 = time.time()
print("Computing features (price, volatility, volume, cross_sectional)...", file=sys.stderr)
from src.features.pipeline import compute_features
enriched = compute_features(df, categories=["price", "volatility", "volume", "cross_sectional"])
print(f"Features: {len(enriched.columns)} columns ({time.time()-t0:.1f}s)", file=sys.stderr)

# Add the advanced features we need (just kalman, skip hurst for speed)
t0 = time.time()
print("Computing advanced features (kalman only)...", file=sys.stderr)
from src.features.advanced import compute_kalman_alpha, compute_kalman_beta, compute_frac_diff_1d
enriched = compute_kalman_alpha(enriched)
enriched = compute_kalman_beta(enriched)
enriched = compute_frac_diff_1d(enriched)
# For Hurst, use a simple approximation
enriched = enriched.with_columns(pl.lit(0.5).alias("hurst_63d"))
print(f"Advanced: {time.time()-t0:.1f}s", file=sys.stderr)

# Generate signals
t0 = time.time()
print("Generating signals...", file=sys.stderr)
from src.signals.pipeline import generate_all_with_forward_returns
result = generate_all_with_forward_returns(enriched, horizons=[1, 5, 21])
print(f"Signals: {len(result.columns)} columns ({time.time()-t0:.1f}s)", file=sys.stderr)

signal_cols = sorted([c for c in result.columns if c.startswith("signal_")])
print(f"\nTotal signals: {len(signal_cols)}")

# IC comparison
from src.signals.base import rank_ic

print("\n" + "=" * 80)
print(f"EXPANDED UNIVERSE ({n_tickers} TICKERS) — 2020-2026")
print("=" * 80)

for horizon in [1, 5, 21]:
    target = f"forward_return_{horizon}"
    print(f"\n--- vs {target} ---")
    rows = []
    for sc in signal_cols:
        ic = rank_ic(sc, target, result)
        variant = "TS" if "ts_" in sc else "CS"
        rows.append((sc, variant, ic))
    rows.sort(key=lambda x: abs(x[2]), reverse=True)
    for sc, variant, ic in rows:
        best = " <-- BEST" if abs(ic) == max(abs(r[2]) for r in rows) else ""
        print(f"  {variant} {sc:45s} IC={ic:+.4f}{best}")

# Compare: 5-ticker baseline vs 25-ticker expanded
print("\n" + "=" * 80)
print("CROSS-SECTIONAL IMPROVEMENT (5 vs 25 tickers)")
print("=" * 80)

df5 = (
    pl.scan_parquet("data/gold/daily/**/*.parquet")
    .filter(
        (pl.col("date") >= pl.date(2020, 1, 1)) &
        pl.col("ticker").is_in(["AAPL", "AMZN", "GOOGL", "META", "MSFT"])
    )
    .sort(["ticker", "date"])
    .collect()
)
enriched5 = compute_features(df5, categories=["price", "volatility", "volume", "cross_sectional"])
enriched5 = compute_kalman_alpha(enriched5)
enriched5 = compute_kalman_beta(enriched5)
enriched5 = compute_frac_diff_1d(enriched5)
enriched5 = enriched5.with_columns(pl.lit(0.5).alias("hurst_63d"))
result5 = generate_all_with_forward_returns(enriched5, horizons=[1, 5, 21])

for horizon in [1, 5, 21]:
    target = f"forward_return_{horizon}"
    print(f"\n--- vs {target} ---")
    cs_cols = [c for c in result5.columns if c.startswith("signal_") and "ts_" not in c]
    for sc in sorted(cs_cols):
        ic5 = rank_ic(sc, target, result5)
        ic25 = rank_ic(sc, target, result)
        delta = ic25 - ic5
        arrow = ">>>" if abs(delta) > 0.005 else ">>" if abs(delta) > 0.002 else ""
        print(f"  {sc:45s} 5T={ic5:+.4f}  25T={ic25:+.4f}  delta={delta:+.4f} {arrow}")

# Ensemble IC
print("\n" + "=" * 80)
print("ENSEMBLE IC (positive_ic weighting, TS signals only)")
print("=" * 80)
from src.ensemble.base import ic_to_weights

ts_cols = [c for c in signal_cols if "ts_" in c]

for horizon in [1, 5, 21]:
    target = f"forward_return_{horizon}"
    ic_vals = {}
    for sc in ts_cols:
        ic_vals[sc] = rank_ic(sc, target, result)

    weights = ic_to_weights(ic_vals, "positive_ic")
    ensemble = result.clone()
    score = pl.lit(0.0)
    for sc, w in weights.items():
        score = score + pl.col(sc) * w
    ensemble = ensemble.with_columns(score.alias("ensemble_ts"))

    ens_ic = rank_ic("ensemble_ts", target, ensemble)
    print(f"\n--- vs {target} ---")
    print(f"  Ensemble (TS, positive_ic): IC={ens_ic:+.4f}")

print("\nDone.")