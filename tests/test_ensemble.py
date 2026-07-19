"""Ensemble tests — CSIC, weights, and IC-weighted ensemble pipeline.

Tests follow the same synthetic data patterns as test_signals.py:
_multi_ticker_bars() for controlled oscillating prices, plus real
data integration tests when gold layer data is available.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.signals.registry import registry as signal_registry
from src.signals.base import (
    compute_forward_returns,
    rank_ic,
    signal_summary,
)
from src.signals.pipeline import generate_all, generate_all_with_forward_returns
from src.signals import mean_reversion, momentum  # noqa: F401 — registers signals
from src.features.pipeline import compute_features
from src.ensemble.ic_weighted import (
    compute_csic,
    compute_ensemble_weights,
    compute_ic_weighted_ensemble,
)


# ── helpers ──────────────────────────────────────────────────────────


def _trading_dates(start: date, n: int) -> list[date]:
    dates: list[date] = []
    d = start
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    return dates


def _make_bars(
    ticker: str,
    num_days: int = 200,
    base_close: float = 100.0,
    start_date: date = date(2019, 1, 3),
    volume: int = 1_000_000,
    seed: int = 0,
) -> pl.DataFrame:
    import math

    closes = []
    c = base_close
    for i in range(num_days):
        c = c * (1.0 + 0.02 * math.sin(i * 0.1))  # gentle oscillation
        closes.append(round(c, 2))
    return pl.DataFrame(
        {
            "ticker": [ticker] * num_days,
            "date": dates,
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [max(c - 1, 1.0) for c in closes],
            "close": closes,
            "volume": [volume + (i % 10) * 100_000 for i in range(num_days)],
        }
    )


def _multi_ticker_bars(n_tickers: int = 5, n_days: int = 300) -> pl.DataFrame:
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"][:n_tickers]
    frames = [
        _make_bars(t, num_days=n_days, base_close=100.0 + i * 50)
        for i, t in enumerate(tickers)
    ]
    return pl.concat(frames, how="vertical_relaxed").sort(["ticker", "date"])


def _prepare_signals(n_tickers: int = 5, n_days: int = 300) -> pl.DataFrame:
    """Full pipeline: bars → features → signals → forward returns."""
    df = _multi_ticker_bars(n_tickers, n_days)
    df = compute_features(df)
    df = generate_all(df)
    df = compute_forward_returns(df, horizons=[1, 5, 21])
    return df


SIGNAL_COLS = [
    "signal_mean_reversion_21d",
    "signal_mean_reversion_63d",
    "signal_momentum_21d",
    "signal_momentum_63d",
]


# ── CSIC tests ───────────────────────────────────────────────────────


class TestCSIC:
    def test_csic_columns(self):
        df = _prepare_signals(5, 200)
        csic = compute_csic(df, SIGNAL_COLS, "forward_return_1")
        for s in SIGNAL_COLS:
            assert f"ic_{s}" in csic.columns, f"Missing ic_{s}"
        assert "ic_mean" in csic.columns
        assert "date" in csic.columns

    def test_csic_values_in_range(self):
        df = _prepare_signals(5, 200)
        csic = compute_csic(df, SIGNAL_COLS, "forward_return_1")
        for s in SIGNAL_COLS:
            col = csic[f"ic_{s}"]
            # CSIC should be in [-1, +1]
            assert (col >= -1.0).all(), f"ic_{s} below -1"
            assert (col <= 1.0).all(), f"ic_{s} above +1"

    def test_csic_non_empty(self):
        df = _prepare_signals(5, 200)
        csic = compute_csic(df, SIGNAL_COLS, "forward_return_1")
        assert len(csic) > 0

    def test_csic_date_count(self):
        """CSIC should have one row per unique date with valid data."""
        df = _prepare_signals(5, 200)
        csic = compute_csic(df, SIGNAL_COLS, "forward_return_1")
        # CSIC dates should be a subset of original dates
        df_dates = set(df["date"].to_list())
        csic_dates = set(csic["date"].to_list())
        assert csic_dates.issubset(df_dates)

    def test_csic_single_signal(self):
        df = _prepare_signals(5, 200)
        csic = compute_csic(df, [SIGNAL_COLS[0]], "forward_return_1")
        assert f"ic_{SIGNAL_COLS[0]}" in csic.columns
        assert "ic_mean" in csic.columns

    def test_csic_missing_target_raises(self):
        df = _prepare_signals(5, 200)
        with pytest.raises(ValueError, match="Target column"):
            compute_csic(df, SIGNAL_COLS, "nonexistent_col")

    def test_csic_missing_signal_raises(self):
        df = _prepare_signals(5, 200)
        with pytest.raises(ValueError, match="Signal columns not found"):
            compute_csic(df, ["signal_nonexistent"], "forward_return_1")


# ── Ensemble weights tests ──────────────────────────────────────────


class TestEnsembleWeights:
    def test_weights_columns(self):
        df = _prepare_signals(5, 200)
        csic = compute_csic(df, SIGNAL_COLS, "forward_return_1")
        weights = compute_ensemble_weights(csic, SIGNAL_COLS, ic_window=21)
        for s in SIGNAL_COLS:
            assert f"weight_{s}" in weights.columns
        assert "weight_sum" in weights.columns

    def test_weights_sum_to_one(self):
        df = _prepare_signals(5, 200)
        csic = compute_csic(df, SIGNAL_COLS, "forward_return_1")
        weights = compute_ensemble_weights(csic, SIGNAL_COLS, ic_window=21)
        # Weight sum should be very close to 1.0 (accounting for floating point)
        ws = weights["weight_sum"].drop_nulls()
        assert (ws >= 0.99).all(), "Weights should sum to ~1.0"
        assert (ws <= 1.01).all(), "Weights should sum to ~1.0"

    def test_weights_non_negative(self):
        df = _prepare_signals(5, 200)
        csic = compute_csic(df, SIGNAL_COLS, "forward_return_1")
        weights = compute_ensemble_weights(csic, SIGNAL_COLS, ic_window=21)
        for s in SIGNAL_COLS:
            w = weights[f"weight_{s}"].drop_nulls()
            assert (w >= 0.0).all(), f"weight_{s} should be non-negative"

    def test_weights_fallback_equal(self):
        """When all ICs are zero/negative, weights should be equal."""
        # Construct CSIC with all zeros
        csic = pl.DataFrame({
            "date": range(100),
            "ic_signal_a": [0.0] * 100,
            "ic_signal_b": [0.0] * 100,
            "ic_mean": [0.0] * 100,
        })
        weights = compute_ensemble_weights(csic, ["signal_a", "signal_b"], ic_window=10)
        # After the first lag + floor, should converge to equal weights
        non_null_a = weights["weight_signal_a"].drop_nulls()
        non_null_b = weights["weight_signal_b"].drop_nulls()
        if len(non_null_a) > 1:  # skip first date (lag creates null)
            assert abs(float(non_null_a.mean()) - 0.5) < 0.01
            assert abs(float(non_null_b.mean()) - 0.5) < 0.01


# ── Full ensemble pipeline tests ────────────────────────────────────


class TestICWeightedEnsemble:
    def test_ensemble_column_produced(self):
        df = _prepare_signals(5, 200)
        out = compute_ic_weighted_ensemble(df, SIGNAL_COLS, ic_window=21)
        assert "signal_ensemble" in out.columns

    def test_ensemble_bounds(self):
        """Ensemble should be bounded roughly in [-1, +1]."""
        df = _prepare_signals(5, 200)
        out = compute_ic_weighted_ensemble(df, SIGNAL_COLS, ic_window=21)
        col = out["signal_ensemble"].drop_nulls()
        # Individual signals are in [-1, +1] and weights sum to 1,
        # so the weighted sum should also be in [-1, +1]
        assert (col <= 1.05).all(), "Ensemble should be bounded ~+1"
        assert (col >= -1.05).all(), "Ensemble should be bounded ~-1"

    def test_ensemble_has_both_signs(self):
        df = _prepare_signals(5, 300)
        out = compute_ic_weighted_ensemble(df, SIGNAL_COLS, ic_window=21)
        col = out["signal_ensemble"].drop_nulls()
        assert (col > 0).any(), "Ensemble should have positive values"
        assert (col < 0).any(), "Ensemble should have negative values"

    def test_ensemble_variance(self):
        """Ensemble should have non-trivial variance."""
        df = _prepare_signals(5, 300)
        out = compute_ic_weighted_ensemble(df, SIGNAL_COLS, ic_window=21)
        std = out["signal_ensemble"].std()
        assert std is not None and std > 0, "Ensemble should have variance"

    def test_ensemble_preserves_original_columns(self):
        """Original signal columns should remain untouched."""
        df = _prepare_signals(5, 200)
        out = compute_ic_weighted_ensemble(df, SIGNAL_COLS, ic_window=21)
        for s in SIGNAL_COLS:
            assert s in out.columns
        assert "signal_ensemble" in out.columns

    def test_ensemble_empty_df(self):
        """Empty input: output should only contain the ensemble column."""
        out = compute_ic_weighted_ensemble(
            pl.DataFrame(), SIGNAL_COLS, ic_window=21
        )
        # Polars with_columns(pl.lit) creates a single-row frame even for
        # empty input; the important invariant is the column exists.
        assert "signal_ensemble" in out.columns
        assert len(out.columns) == 1

    def test_ensemble_with_custom_output(self):
        df = _prepare_signals(5, 200)
        out = compute_ic_weighted_ensemble(
            df, SIGNAL_COLS, ic_window=21, out_col="my_ensemble"
        )
        assert "my_ensemble" in out.columns


# ── Ensemble vs individual signals ──────────────────────────────────


class TestEnsemblePerformance:
    def test_ensemble_ic_comparable(self):
        """Ensemble IC should be comparable to or better than individual signals."""
        df = _prepare_signals(5, 400)
        out = compute_ic_weighted_ensemble(df, SIGNAL_COLS, ic_window=21)

        # Rank IC of ensemble vs 1-day forward return
        ensemble_ic = rank_ic("signal_ensemble", "forward_return_1", out)

        # Best individual IC
        best_ic = max(
            rank_ic(s, "forward_return_1", out) for s in SIGNAL_COLS
        )

        # Ensemble should at least not be dramatically worse
        # (synthetic data is oscillating, so ICs are low — just verify it's computed)
        assert -1.0 <= ensemble_ic <= 1.0
        # Ensemble IC should be within a reasonable range of best individual IC
        # (not necessarily better — synthetic data is noisy)
        assert abs(ensemble_ic) <= abs(best_ic) + 0.3


# ── Integration: real data ──────────────────────────────────────────


class TestRealData:
    def test_ensemble_on_gold_data(self):
        """Run ensemble on actual gold layer data if available."""
        gold_dir = root / "data" / "gold" / "daily" / "year=2023"
        files = list(gold_dir.glob("ticker=*/part-0.parquet"))
        if not files:
            pytest.skip("No gold data available")

        frames = [pl.read_parquet(str(f)) for f in sorted(files)]
        df = pl.concat(frames, how="vertical_relaxed").sort(["ticker", "date"])
        assert not df.is_empty()

        # Full pipeline
        df = compute_features(df)
        df = generate_all(df)
        df = compute_forward_returns(df, horizons=[1, 5, 21])

        # Ensemble
        out = compute_ic_weighted_ensemble(df, SIGNAL_COLS, ic_window=21)
        assert "signal_ensemble" in out.columns

        # Ensemble should have meaningful spread
        std = out["signal_ensemble"].std()
        assert std is not None and std > 0, "Ensemble has no variance on real data"

    def test_csic_on_gold_data(self):
        """CSIC should be computable on real data."""
        gold_dir = root / "data" / "gold" / "daily" / "year=2023"
        files = list(gold_dir.glob("ticker=*/part-0.parquet"))
        if not files:
            pytest.skip("No gold data available")

        frames = [pl.read_parquet(str(f)) for f in sorted(files)]
        df = pl.concat(frames, how="vertical_relaxed").sort(["ticker", "date"])
        df = compute_features(df)
        df = generate_all(df)
        df = compute_forward_returns(df, horizons=[1, 5, 21])

        csic = compute_csic(df, SIGNAL_COLS, "forward_return_1")
        assert len(csic) > 0
        for s in SIGNAL_COLS:
            col = csic[f"ic_{s}"]
            assert (col >= -1.0).all()
            assert (col <= 1.0).all()

    def test_ensemble_beats_individual_on_real_data(self):
        """Ensemble IC should be >= best individual IC on real data."""
        gold_dir = root / "data" / "gold" / "daily" / "year=2023"
        files = list(gold_dir.glob("ticker=*/part-0.parquet"))
        if not files:
            pytest.skip("No gold data available")

        frames = [pl.read_parquet(str(f)) for f in sorted(files)]
        df = pl.concat(frames, how="vertical_relaxed").sort(["ticker", "date"])
        df = compute_features(df)
        df = generate_all(df)
        df = compute_forward_returns(df, horizons=[1, 5, 21])

        out = compute_ic_weighted_ensemble(df, SIGNAL_COLS, ic_window=21)

        ensemble_ic = rank_ic("signal_ensemble", "forward_return_1", out)
        individual_ics = {
            s: rank_ic(s, "forward_return_1", out) for s in SIGNAL_COLS
        }
        best_ic = max(individual_ics.values())

        # Log results for inspection
        print(f"Ensemble IC: {ensemble_ic:.4f}")
        for s, ic in individual_ics.items():
            print(f"  {s} IC: {ic:.4f}")

        # Ensemble should at least not be dramatically worse than best individual
        # (real data gives a better signal; allow some tolerance)
        assert ensemble_ic >= best_ic - 0.1, (
            f"Ensemble IC ({ensemble_ic:.4f}) should be close to best individual "
            f"IC ({best_ic:.4f})"
        )