#!/usr/bin/env python
"""Phase 9: Deep signal analysis across regimes and universes."""

import sys
import time
import numpy as np

import polars as pl
from src.features.pipeline import compute_features
from src.features.advanced import compute_kalman_alpha, compute_kalman_beta, compute_frac_diff_1d
from src.signals.pipeline import generate_all_with_forward_returns
from src.signals.base import rank_ic
from src.ensemble.base import ic_to_weights

def run_analysis(df, label, tickers_info=""):
    """Run full signal analysis on a DataFrame."""
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"{label} {tickers_info}", file=sys.stderr)
    print(f"  {df['ticker'].unique().len()} tickers, {len(df)} rows", file=sys.stderr)

    t0 = time.time()
    enriched = compute_features(df, categories=["price", "volatility", "volume", "cross_sectional"])
    enriched = compute_kalman_alpha(enriched)
    enriched = compute_kalman_beta(enriched)
    enriched = compute_frac_diff_1d(enriched)
    enriched = enriched.with_columns(pl.lit(0.5).alias("hurst_63d"))
    print(f"  Features: {time.time()-t0:.1f}s", file=sys.stderr)

    t0 = time.time()
    result = generate_all_with_forward_returns(enriched, horizons=[1, 5, 21])
    print(f"  Signals: {time.time()-t0:.1f}s", file=sys.stderr)

    return result

# Load data
t0 = time.time()
print("Loading data...", file=sys.stderr)

df_full = (
    pl.scan_parquet("data/gold/daily/**/*.parquet")
    .sort(["ticker", "date"])
    .collect()
)

df_5t_full = df_full.filter(pl.col("ticker").is_in(["AAPL", "AMZN", "GOOGL", "META", "MSFT"]))
df_25t_full = df_full  # All 25 tickers

df_5t_recent = df_5t_full.filter(pl.col("date") >= pl.date(2020, 1, 1))
df_25t_recent = df_full.filter(pl.col("date") >= pl.date(2020, 1, 1))

print(f"Loaded in {time.time()-t0:.1f}s", file=sys.stderr)

# Run analyses
result_5t_full = run_analysis(df_5t_full, "5 TICKERS", "FULL HISTORY (2014-2026)")
result_25t_full = run_analysis(df_25t_full, "25 TICKERS", "FULL HISTORY (2014-2026)")
result_5t_recent = run_analysis(df_5t_recent, "5 TICKERS", "RECENT (2020-2026)")
result_25t_recent = run_analysis(df_25t_recent, "25 TICKERS", "RECENT (2020-2026)")

# Analysis 1: CS signal IC across universes and time periods
print("\n" + "=" * 80)
print("CROSS-SECTIONAL SIGNAL IC COMPARISON")
print("=" * 80)

cs_cols = sorted([c for c in result_5t_full.columns if c.startswith("signal_") and "ts_" not in c])

for horizon in [5]:
    target = f"forward_return_{horizon}"
    print(f"\n--- vs {target} ---")
    print(f"{'Signal':<45s} {'5T Full':>8s} {'25T Full':>8s} {'5T Recent':>10s} {'25T Recent':>11s}", file=sys.stdout)
    for sc in cs_cols:
        ic_5f = rank_ic(sc, target, result_5t_full)
        ic_25f = rank_ic(sc, target, result_25t_full)
        ic_5r = rank_ic(sc, target, result_5t_recent)
        ic_25r = rank_ic(sc, target, result_25t_recent)
        print(f"{sc:<45s} {ic_5f:+.4f} {ic_25f:+.4f} {ic_5r:+.4f} {ic_25r:+.4f}")

# Analysis 2: TS signal IC across universes and time periods
print("\n" + "=" * 80)
print("TIME-SERIES SIGNAL IC COMPARISON")
print("=" * 80)

ts_cols = sorted([c for c in result_5t_full.columns if "ts_" in c])

for horizon in [5]:
    target = f"forward_return_{horizon}"
    print(f"\n--- vs {target} ---")
    print(f"{'Signal':<45s} {'5T Full':>8s} {'25T Full':>8s} {'5T Recent':>10s} {'25T Recent':>11s}", file=sys.stdout)
    for sc in ts_cols:
        ic_5f = rank_ic(sc, target, result_5t_full)
        ic_25f = rank_ic(sc, target, result_25t_full)
        ic_5r = rank_ic(sc, target, result_5t_recent)
        ic_25r = rank_ic(sc, target, result_25t_recent)
        print(f"{sc:<45s} {ic_5f:+.4f} {ic_25f:+.4f} {ic_5r:+.4f} {ic_25r:+.4f}")

# Analysis 3: Ensemble IC
print("\n" + "=" * 80)
print("ENSEMBLE IC COMPARISON (positive_ic weighting)")
print("=" * 80)

for label, result, sig_cols in [
    ("5T Full CS", result_5t_full, cs_cols),
    ("25T Full CS", result_25t_full, cs_cols),
    ("5T Recent CS", result_5t_recent, cs_cols),
    ("25T Recent CS", result_25t_recent, cs_cols),
    ("5T Full TS", result_5t_full, ts_cols),
    ("25T Full TS", result_25t_full, ts_cols),
    ("5T Recent TS", result_5t_recent, ts_cols),
    ("25T Recent TS", result_25t_recent, ts_cols),
]:
    for horizon in [5]:
        target = f"forward_return_{horizon}"
        ic_vals = {sc: rank_ic(sc, target, result) for sc in sig_cols}
        weights = ic_to_weights(ic_vals, "positive_ic")
        ensemble = result.clone()
        score = pl.lit(0.0)
        for sc, w in weights.items():
            score = score + pl.col(sc) * w
        ensemble = ensemble.with_columns(score.alias("_ens"))
        ens_ic = rank_ic("_ens", target, ensemble)
        n_pos = sum(1 for v in ic_vals.values() if v > 0)
        print(f"  {label:20s} IC={ens_ic:+.4f}  ({n_pos}/{len(sig_cols)} positive)")

# Analysis 4: What if we use ALL signals (CS + TS) in the ensemble?
print("\n" + "=" * 80)
print("FULL ENSEMBLE (CS + TS signals combined)")
print("=" * 80)

all_cols = sorted([c for c in result_25t_full.columns if c.startswith("signal_")])
for horizon in [5]:
    target = f"forward_return_{horizon}"
    ic_vals = {sc: rank_ic(sc, target, result_25t_full) for sc in all_cols}
    weights = ic_to_weights(ic_vals, "positive_ic")
    ensemble = result_25t_full.clone()
    score = pl.lit(0.0)
    for sc, w in weights.items():
        score = score + pl.col(sc) * w
    ensemble = ensemble.with_columns(score.alias("_ens"))
    ens_ic = rank_ic("_ens", target, ensemble)
    n_pos = sum(1 for v in ic_vals.values() if v > 0)
    print(f"\n  25T Full ALL (CS+TS): IC={ens_ic:+.4f}  ({n_pos}/{len(all_cols)} positive)")
    print(f"  IC values: {dict(sorted(ic_vals.items(), key=lambda x: x[1], reverse=True))}")
    print(f"  Weights:   {dict(sorted(weights.items(), key=lambda x: x[1], reverse=True))}")

print("\nDone.")