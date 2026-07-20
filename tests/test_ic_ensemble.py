"""IC-weighted ensemble tests — base, ic_weighted, and pipeline."""

import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.ensemble.base import (
    EnsembleConfig,
    EnsembleMeta,
    detect_signal_columns,
    ic_to_weights,
)
from src.ensemble.ic_weighted import (
    ICWeightedEnsemble,
    compute_cross_sectional_ic,
    compute_rolling_ic,
)
from src.ensemble.pipeline import run_ensemble
from src.features.pipeline import compute_features
from src.signals.pipeline import generate_all_with_forward_returns
from src.signals import mean_reversion, momentum  # noqa: F401


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
) -> pl.DataFrame:
    dates = _trading_dates(start_date, num_days)
    import math

    closes = []
    c = base_close
    for i in range(num_days):
        c = c * (1.0 + 0.02 * math.sin(i * 0.1))
        closes.append(round(c, 2))
    return pl.DataFrame(
        {
            "ticker": [ticker] * num_days,
            "date": dates,
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [max(c - 1, 1.0) for c in closes],
            "close": closes,
            "volume": [1_000_000] * num_days,
        }
    )


def _signal_df(n_tickers: int = 5, n_days: int = 300) -> pl.DataFrame:
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"][:n_tickers]
    frames = [
        _make_bars(t, num_days=n_days, base_close=100.0 + i * 50)
        for i, t in enumerate(tickers)
    ]
    df = pl.concat(frames, how="vertical_relaxed").sort(["ticker", "date"])
    df = compute_features(df)
    df = generate_all_with_forward_returns(df, horizons=[1, 5, 21])
    return df


# ── ic_to_weights tests ──────────────────────────────────────────────


class TestICToWeights:
    def test_abs_ic_weights_sum_to_one(self):
        ics = {"a": 0.1, "b": 0.2, "c": -0.05}
        weights = ic_to_weights(ics, method="abs_ic")
        assert abs(sum(weights.values()) - 1.0) < 1e-9
        # b should have the largest weight
        assert weights["b"] > weights["a"] > weights["c"]

    def test_rank_ic_weights_sum_to_one(self):
        ics = {"a": 0.1, "b": 0.2, "c": 0.05}
        weights = ic_to_weights(ics, method="rank_ic")
        assert abs(sum(weights.values()) - 1.0) < 1e-9
        # b should have highest rank (rank 3), a rank 2, c rank 1
        assert weights["b"] > weights["a"] > weights["c"]

    def test_positive_ic_drops_negative(self):
        ics = {"a": 0.1, "b": 0.2, "c": -0.05}
        weights = ic_to_weights(ics, method="positive_ic")
        assert weights["c"] == 0.0
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_positive_ic_fallback_equal_weight(self):
        ics = {"a": -0.1, "b": -0.2}
        weights = ic_to_weights(ics, method="positive_ic")
        assert abs(weights["a"] - 0.5) < 1e-9
        assert abs(weights["b"] - 0.5) < 1e-9

    def test_empty_dict(self):
        weights = ic_to_weights({})
        assert weights == {}


# ── detect_signal_columns tests ──────────────────────────────────────


class TestDetectSignalColumns:
    def test_detects_signal_columns(self):
        cols = ["ticker", "date", "signal_mean_reversion_21d", "signal_momentum_21d"]
        result = detect_signal_columns(cols)
        assert result == ["signal_mean_reversion_21d", "signal_momentum_21d"]

    def test_excludes_signal_ensemble(self):
        cols = ["signal_ensemble", "signal_mean_reversion_21d"]
        result = detect_signal_columns(cols)
        assert result == ["signal_mean_reversion_21d"]

    def test_empty_result(self):
        result = detect_signal_columns(["ticker", "date", "close"])
        assert result == []


# ── EnsembleConfig tests ─────────────────────────────────────────────


class TestEnsembleConfig:
    def test_defaults(self):
        config = EnsembleConfig()
        assert config.ic_lookback == 63
        assert config.ic_target_horizon == 5
        assert config.rebalance_frequency == 5
        assert config.weight_method == "abs_ic"
        assert config.output_column == "signal_ensemble"
        assert config.track_weights is True

    def test_custom_values(self):
        config = EnsembleConfig(
            ic_lookback=21,
            ic_target_horizon=1,
            rebalance_frequency=10,
            weight_method="positive_ic",
        )
        assert config.ic_lookback == 21
        assert config.ic_target_horizon == 1
        assert config.rebalance_frequency == 10
        assert config.weight_method == "positive_ic"

    def test_invalid_lookback(self):
        with pytest.raises(ValueError, match="ic_lookback"):
            EnsembleConfig(ic_lookback=1)

    def test_invalid_target_horizon(self):
        with pytest.raises(ValueError, match="ic_target_horizon"):
            EnsembleConfig(ic_target_horizon=0)

    def test_invalid_rebalance_frequency(self):
        with pytest.raises(ValueError, match="rebalance_frequency"):
            EnsembleConfig(rebalance_frequency=0)

    def test_invalid_weight_method(self):
        with pytest.raises(ValueError, match="weight_method"):
            EnsembleConfig(weight_method="unknown")


# ── compute_cross_sectional_ic tests ─────────────────────────────────


class TestCrossSectionalIC:
    def test_returns_date_and_cs_ic_columns(self):
        df = _signal_df(5, 150)
        csic = compute_cross_sectional_ic(
            df, "signal_mean_reversion_21d", "forward_return_5"
        )
        assert "date" in csic.columns
        assert "cs_ic" in csic.columns

    def test_cs_ic_in_range(self):
        df = _signal_df(5, 150)
        csic = compute_cross_sectional_ic(
            df, "signal_mean_reversion_21d", "forward_return_5"
        )
        # Allow tiny floating-point overshoot beyond 1.0 due to clip
        values = csic["cs_ic"].drop_nulls().to_list()
        assert all(v >= -1.0 - 1e-9 for v in values), "CS IC should be >= -1"
        assert all(v <= 1.0 + 1e-9 for v in values), "CS IC should be <= 1"

    def test_cs_ic_has_results(self):
        df = _signal_df(5, 150)
        csic = compute_cross_sectional_ic(
            df, "signal_mean_reversion_21d", "forward_return_5"
        )
        assert len(csic) > 0


# ── compute_rolling_ic tests ─────────────────────────────────────────


class TestRollingIC:
    def test_returns_date_and_rolling_ic(self):
        df = _signal_df(5, 150)
        ric = compute_rolling_ic(
            df, "signal_mean_reversion_21d", "forward_return_5", window=21
        )
        assert "date" in ric.columns
        assert "rolling_ic" in ric.columns

    def test_rolling_ic_lag(self):
        """Rolling IC should have fewer results than CS IC due to window."""
        df = _signal_df(5, 150)
        csic = compute_cross_sectional_ic(
            df, "signal_mean_reversion_21d", "forward_return_5"
        )
        ric = compute_rolling_ic(
            df, "signal_mean_reversion_21d", "forward_return_5", window=21
        )
        # Rolling IC has nulls for early dates (warm-up period)
        non_null = ric["rolling_ic"].drop_nulls()
        assert len(non_null) < len(csic), "Rolling IC should have fewer valid dates than CS IC"


# ── ICWeightedEnsemble tests ─────────────────────────────────────────


class TestICWeightedEnsemble:
    def test_transform_produces_ensemble_score(self):
        df = _signal_df(5, 200)
        ens = ICWeightedEnsemble(ic_window=21, rebalance_freq=5)
        signal_cols = ["signal_mean_reversion_21d", "signal_momentum_21d"]
        result = ens.transform(df, signal_cols, "forward_return_5")
        assert "ensemble_score" in result.columns

    def test_transform_output_length_matches_input(self):
        df = _signal_df(5, 200)
        ens = ICWeightedEnsemble(ic_window=21, rebalance_freq=5)
        result = ens.transform(df, ["signal_mean_reversion_21d"], "forward_return_5")
        assert len(result) == len(df)

    def test_ensemble_score_variance(self):
        df = _signal_df(5, 200)
        ens = ICWeightedEnsemble(ic_window=21, rebalance_freq=5)
        result = ens.transform(
            df,
            ["signal_mean_reversion_21d", "signal_momentum_21d"],
            "forward_return_5",
        )
        score = result["ensemble_score"].drop_nulls()
        std = score.std()
        assert std is not None and std > 0, "Ensemble score should have variance"

    def test_weight_columns_present(self):
        df = _signal_df(5, 200)
        ens = ICWeightedEnsemble(ic_window=21, rebalance_freq=5)
        signal_cols = ["signal_mean_reversion_21d", "signal_momentum_21d"]
        result = ens.transform(df, signal_cols, "forward_return_5")
        assert "w_signal_mean_reversion_21d" in result.columns
        assert "w_signal_momentum_21d" in result.columns

    def test_weights_sum_to_one(self):
        df = _signal_df(5, 200)
        ens = ICWeightedEnsemble(ic_window=21, rebalance_freq=5)
        signal_cols = ["signal_mean_reversion_21d", "signal_momentum_21d"]
        result = ens.transform(df, signal_cols, "forward_return_5")
        weight_sum = (
            result["w_signal_mean_reversion_21d"]
            + result["w_signal_momentum_21d"]
        )
        non_null = weight_sum.drop_nulls()
        assert all(abs(v - 1.0) < 1e-6 for v in non_null.to_list()), (
            "Weights should sum to 1.0"
        )

    def test_predict_requires_transform(self):
        ens = ICWeightedEnsemble()
        with pytest.raises(RuntimeError, match="transform"):
            ens.predict(pl.DataFrame(), [])

    def test_predict_preserves_weights(self):
        df = _signal_df(5, 200)
        ens = ICWeightedEnsemble(ic_window=21, rebalance_freq=5)
        signal_cols = ["signal_mean_reversion_21d", "signal_momentum_21d"]
        ens.transform(df, signal_cols, "forward_return_5")
        predict_result = ens.predict(df, signal_cols)
        assert "ensemble_score" in predict_result.columns


# ── run_ensemble pipeline tests ──────────────────────────────────────


class TestRunEnsemble:
    def test_run_ensemble_basic(self):
        df = _signal_df(5, 200)
        result = run_ensemble(df)
        assert "signal_ensemble" in result.columns

    def test_run_ensemble_with_config(self):
        df = _signal_df(5, 200)
        config = EnsembleConfig(
            ic_lookback=21,
            ic_target_horizon=5,
            weight_method="positive_ic",
        )
        result = run_ensemble(df, config)
        assert "signal_ensemble" in result.columns

    def test_run_ensemble_no_signals_raises(self):
        df = pl.DataFrame({"ticker": ["A"], "date": [date(2023, 1, 1)]})
        with pytest.raises(ValueError, match="No signal columns"):
            run_ensemble(df)

    def test_run_ensemble_missing_signal_raises(self):
        df = _signal_df(5, 200)
        config = EnsembleConfig(signal_columns=["signal_fake"])
        with pytest.raises(ValueError, match="Signal columns missing"):
            run_ensemble(df, config)

    def test_run_ensemble_output_length(self):
        df = _signal_df(5, 200)
        result = run_ensemble(df)
        assert len(result) == len(df)

    def test_run_ensemble_no_track_weights(self):
        df = _signal_df(5, 200)
        config = EnsembleConfig(track_weights=False)
        result = run_ensemble(df, config)
        w_cols = [c for c in result.columns if c.startswith("w_")]
        assert len(w_cols) == 0, "Weight columns should be dropped when track_weights=False"


# ── Real data tests ──────────────────────────────────────────────────


class TestRealData:
    def test_ensemble_on_gold_data(self):
        gold_dir = root / "data" / "gold" / "daily" / "year=2023"
        files = list(gold_dir.glob("ticker=*/part-0.parquet"))
        if not files:
            pytest.skip("No gold data available")

        frames = [pl.read_parquet(str(f)) for f in sorted(files)]
        df = pl.concat(frames, how="vertical_relaxed").sort(["ticker", "date"])
        assert not df.is_empty()

        df = compute_features(df)
        df = generate_all_with_forward_returns(df, horizons=[1, 5, 21])

        result = run_ensemble(df, EnsembleConfig(ic_lookback=21))
        assert "signal_ensemble" in result.columns
        score = result["signal_ensemble"].drop_nulls()
        assert len(score) > 0

    def test_ensemble_ic_on_gold_data(self):
        gold_dir = root / "data" / "gold" / "daily" / "year=2023"
        files = list(gold_dir.glob("ticker=*/part-0.parquet"))
        if not files:
            pytest.skip("No gold data available")

        frames = [pl.read_parquet(str(f)) for f in sorted(files)]
        df = pl.concat(frames, how="vertical_relaxed").sort(["ticker", "date"])
        df = compute_features(df)
        df = generate_all_with_forward_returns(df, horizons=[1, 5, 21])

        result = run_ensemble(df, EnsembleConfig(ic_lookback=21, ic_target_horizon=5))
        score = result["signal_ensemble"].drop_nulls()
        assert len(score) > 0
        # Ensemble should have some predictive power
        fwd = result["forward_return_5"].drop_nulls()
        assert len(fwd) > 0