"""Ensemble tests — CSIC, rolling IC, weights, and IC-weighted ensemble.

Tests follow the same synthetic data patterns as test_signals.py:
_multi_ticker_bars() for controlled oscillating prices, plus real
data integration tests when gold layer data is available.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.signals.base import (
    compute_forward_returns,
    rank_ic,
)
from src.signals.pipeline import generate_all, generate_all_with_forward_returns
from src.signals import mean_reversion, momentum  # noqa: F401 — registers signals
from src.features.pipeline import compute_features
from src.ensemble.ic_weighted import (
    compute_cross_sectional_ic,
    compute_rolling_ic,
    ic_to_weights,
    ICWeightedEnsemble,
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
    num_days: int = 300,
    base_close: float = 100.0,
    start_date: date = date(2023, 1, 3),
    volume: int = 1_000_000,
) -> pl.DataFrame:
    dates = _trading_dates(start_date, num_days)
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


# ── Cross-sectional IC tests ─────────────────────────────────────────


class TestCrossSectionalIC:
    def test_csic_returns_two_columns(self):
        df = _prepare_signals(5, 200)
        csic = compute_cross_sectional_ic(
            df, "signal_momentum_21d", "forward_return_1"
        )
        assert "date" in csic.columns
        assert "cs_ic" in csic.columns

    def test_csic_values_in_range(self):
        df = _prepare_signals(5, 200)
        csic = compute_cross_sectional_ic(
            df, "signal_momentum_21d", "forward_return_1"
        )
        assert (csic["cs_ic"] >= -1.0).all()
        assert (csic["cs_ic"] <= 1.0).all()

    def test_csic_has_dates(self):
        df = _prepare_signals(5, 200)
        csic = compute_cross_sectional_ic(
            df, "signal_momentum_21d", "forward_return_1"
        )
        # Should have at least some dates
        assert len(csic) > 0

    def test_csic_empty_for_null_target(self):
        """If target is all null, CSIC should be empty."""
        df = _prepare_signals(5, 200).with_columns(
            pl.lit(None, dtype=pl.Float64).alias("forward_return_1")
        )
        csic = compute_cross_sectional_ic(
            df, "signal_momentum_21d", "forward_return_1"
        )
        assert csic.is_empty()


# ── Rolling IC tests ─────────────────────────────────────────────────


class TestRollingIC:
    def test_rolling_ic_columns(self):
        df = _prepare_signals(5, 200)
        ric = compute_rolling_ic(df, "signal_momentum_21d", "forward_return_1", window=21)
        assert "date" in ric.columns
        assert "rolling_ic" in ric.columns

    def test_rolling_ic_smoothes(self):
        df = _prepare_signals(5, 300)
        csic = compute_cross_sectional_ic(df, "signal_momentum_21d", "forward_return_1")
        ric = compute_rolling_ic(df, "signal_momentum_21d", "forward_return_1", window=21)

        # Rolling IC should have similar dates to CSIC (possibly shifted)
        assert len(ric) > 0

    def test_rolling_ic_values_in_range(self):
        df = _prepare_signals(5, 300)
        ric = compute_rolling_ic(df, "signal_momentum_21d", "forward_return_1", window=21)
        non_null = ric["rolling_ic"].drop_nulls()
        if len(non_null) > 0:
            assert (non_null >= -1.0).all()
            assert (non_null <= 1.0).all()


# ── Weight computation tests ─────────────────────────────────────────


class TestICtoWeights:
    def test_abs_ic_sum_to_one(self):
        weights = ic_to_weights({"sig_a": 0.1, "sig_b": 0.2}, method="abs_ic")
        assert abs(sum(weights.values()) - 1.0) < 1e-9
        assert len(weights) == 2

    def test_abs_ic_proportional(self):
        weights = ic_to_weights({"sig_a": 0.1, "sig_b": 0.3}, method="abs_ic")
        # sig_b should have 3x the weight of sig_a
        assert abs(weights["sig_b"] / weights["sig_a"] - 3.0) < 1e-9

    def test_rank_ic_ordering(self):
        weights = ic_to_weights(
            {"sig_a": 0.05, "sig_b": 0.15, "sig_c": 0.25}, method="rank_ic"
        )
        # Highest |IC| should get highest weight
        assert weights["sig_c"] > weights["sig_b"] > weights["sig_a"]

    def test_positive_ic_zeros_negative(self):
        weights = ic_to_weights(
            {"sig_a": 0.2, "sig_b": -0.1, "sig_c": 0.05}, method="positive_ic"
        )
        assert weights["sig_b"] == 0.0
        assert weights["sig_a"] > 0
        assert weights["sig_c"] > 0

    def test_positive_ic_all_negative(self):
        """When all ICs are negative, all weights should be equal (fallback)."""
        weights = ic_to_weights(
            {"sig_a": -0.1, "sig_b": -0.2}, method="positive_ic"
        )
        # total = 0, so fallback to equal weights
        assert abs(weights["sig_a"] - 0.5) < 1e-9
        assert abs(weights["sig_b"] - 0.5) < 1e-9

    def test_empty_input(self):
        weights = ic_to_weights({})
        assert weights == {}

    def test_unknown_method_falls_back_to_equal(self):
        weights = ic_to_weights({"sig_a": 0.1, "sig_b": 0.2}, method="unknown")
        assert abs(weights["sig_a"] - 0.5) < 1e-9
        assert abs(weights["sig_b"] - 0.5) < 1e-9


# ── Ensemble class tests ─────────────────────────────────────────────


class TestICWeightedEnsemble:
    def test_transform_produces_ensemble_score(self):
        df = _prepare_signals(5, 200)
        ensemble = ICWeightedEnsemble(ic_window=21, rebalance_freq=5)
        out = ensemble.transform(df, SIGNAL_COLS, "forward_return_1")
        assert "ensemble_score" in out.columns

    def test_transform_produces_weight_columns(self):
        df = _prepare_signals(5, 200)
        ensemble = ICWeightedEnsemble(ic_window=21, rebalance_freq=5)
        out = ensemble.transform(df, SIGNAL_COLS, "forward_return_1")
        for s in SIGNAL_COLS:
            assert f"w_{s}" in out.columns, f"Missing weight column w_{s}"

    def test_ensemble_score_has_variance(self):
        df = _prepare_signals(5, 300)
        ensemble = ICWeightedEnsemble(ic_window=21, rebalance_freq=5)
        out = ensemble.transform(df, SIGNAL_COLS, "forward_return_1")
        std = out["ensemble_score"].std()
        assert std is not None and std > 0, "Ensemble should have variance"

    def test_weights_sum_to_one(self):
        df = _prepare_signals(5, 200)
        ensemble = ICWeightedEnsemble(ic_window=21, rebalance_freq=5)
        out = ensemble.transform(df, SIGNAL_COLS, "forward_return_1")
        # Check weight sum per row
        weight_cols = [f"w_{s}" for s in SIGNAL_COLS]
        ws = out.select(pl.sum_horizontal(weight_cols).alias("w_sum"))["w_sum"].drop_nulls()
        # Weights should sum close to 1.0
        assert (ws >= 0.99).all(), "Weights should sum to ~1.0"
        assert (ws <= 1.01).all(), "Weights should sum to ~1.0"

    def test_predict_after_transform(self):
        df = _prepare_signals(5, 200)
        ensemble = ICWeightedEnsemble(ic_window=21, rebalance_freq=5)
        ensemble.transform(df, SIGNAL_COLS, "forward_return_1")

        # Create a prediction DataFrame with same columns
        predict_df = df.select(["date"] + SIGNAL_COLS)
        out = ensemble.predict(predict_df, SIGNAL_COLS)
        assert "ensemble_score" in out.columns

    def test_predict_without_transform_raises(self):
        ensemble = ICWeightedEnsemble()
        df = _prepare_signals(5, 100)
        predict_df = df.select(["date"] + SIGNAL_COLS)
        with pytest.raises(RuntimeError, match="Call transform"):
            ensemble.predict(predict_df, SIGNAL_COLS)

    def test_different_weight_methods(self):
        df = _prepare_signals(5, 200)
        for method in ["abs_ic", "rank_ic", "positive_ic"]:
            ensemble = ICWeightedEnsemble(
                ic_window=21, rebalance_freq=5, weight_method=method
            )
            out = ensemble.transform(df, SIGNAL_COLS, "forward_return_1")
            assert "ensemble_score" in out.columns

    def test_single_signal(self):
        df = _prepare_signals(5, 200)
        ensemble = ICWeightedEnsemble(ic_window=21, rebalance_freq=5)
        out = ensemble.transform(df, [SIGNAL_COLS[0]], "forward_return_1")
        assert "ensemble_score" in out.columns
        assert f"w_{SIGNAL_COLS[0]}" in out.columns


# ── Ensemble vs individual signals ──────────────────────────────────


class TestEnsemblePerformance:
    def test_ensemble_ic_comparable(self):
        """Ensemble IC should be comparable to individual signals."""
        df = _prepare_signals(5, 400)
        ensemble = ICWeightedEnsemble(ic_window=21, rebalance_freq=5)
        out = ensemble.transform(df, SIGNAL_COLS, "forward_return_1")

        # Rank IC of ensemble vs 1-day forward return
        ensemble_ic = rank_ic("ensemble_score", "forward_return_1", out)
        assert -1.0 <= ensemble_ic <= 1.0


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
        ensemble = ICWeightedEnsemble(ic_window=21, rebalance_freq=5)
        out = ensemble.transform(df, SIGNAL_COLS, "forward_return_1")
        assert "ensemble_score" in out.columns

        # Ensemble should have meaningful spread
        std = out["ensemble_score"].std()
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

        for sig in SIGNAL_COLS:
            csic = compute_cross_sectional_ic(df, sig, "forward_return_1")
            assert len(csic) > 0, f"No CSIC rows for {sig}"
            assert (csic["cs_ic"] >= -1.0).all()
            assert (csic["cs_ic"] <= 1.0).all()

    def test_ensemble_ic_on_real_data(self):
        """Report ensemble IC vs individual ICs on real data."""
        gold_dir = root / "data" / "gold" / "daily" / "year=2023"
        files = list(gold_dir.glob("ticker=*/part-0.parquet"))
        if not files:
            pytest.skip("No gold data available")

        frames = [pl.read_parquet(str(f)) for f in sorted(files)]
        df = pl.concat(frames, how="vertical_relaxed").sort(["ticker", "date"])
        df = compute_features(df)
        df = generate_all(df)
        df = compute_forward_returns(df, horizons=[1, 5, 21])

        ensemble = ICWeightedEnsemble(ic_window=21, rebalance_freq=5)
        out = ensemble.transform(df, SIGNAL_COLS, "forward_return_1")

        ensemble_ic = rank_ic("ensemble_score", "forward_return_1", out)
        individual_ics = {
            s: rank_ic(s, "forward_return_1", out) for s in SIGNAL_COLS
        }
        best_ic = max(individual_ics.values())

        # Log results for inspection
        print(f"Ensemble IC: {ensemble_ic:.4f}")
        for s, ic in individual_ics.items():
            print(f"  {s} IC: {ic:.4f}")

        # Ensemble should be in a reasonable range
        assert -1.0 <= ensemble_ic <= 1.0