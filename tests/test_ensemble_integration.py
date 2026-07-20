"""Ensemble integration tests — unified pipeline, both modes, validation.

Tests that the EnsemblePipeline orchestrator:
    - Runs IC-weighted mode end-to-end
    - Runs LightGBM mode end-to-end
    - Validates ensemble against individual signals
    - Produces the correct I/O contract (signal_ensemble column)
    - Auto-detects signal columns
    - Handles missing columns gracefully
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.features.pipeline import compute_features
from src.signals.pipeline import generate_all, generate_all_with_forward_returns
from src.signals import mean_reversion, momentum  # noqa: F401 — registers signals
from src.ensemble.pipeline import (
    EnsemblePipeline,
    EnsemblePipelineConfig,
    run_ensemble,
)
from src.ensemble.validation import (
    EnsembleMetrics,
    SignalMetrics,
    validate_ensemble,
    validate_signals,
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
    start_date: date = date(2020, 1, 3),
) -> pl.DataFrame:
    import math

    dates = _trading_dates(start_date, num_days)
    closes = []
    c = base_close
    for i in range(num_days):
        c = c * (1.0 + 0.02 * math.sin(i * 0.1))
        closes.append(round(c, 2))
    return pl.DataFrame({
        "ticker": [ticker] * num_days,
        "date": dates,
        "open": closes,
        "high": [c + 1 for c in closes],
        "low": [max(c - 1, 1.0) for c in closes],
        "close": closes,
        "volume": [1_000_000 + (i % 10) * 100_000 for i in range(num_days)],
    })


def _multi_ticker_bars(n_tickers: int = 5, n_days: int = 300) -> pl.DataFrame:
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"][:n_tickers]
    frames = [
        _make_bars(t, num_days=n_days, base_close=100.0 + i * 50)
        for i, t in enumerate(tickers)
    ]
    return pl.concat(frames, how="vertical_relaxed").sort(["ticker", "date"])


def _prepare_signals(n_tickers: int = 5, n_days: int = 300) -> pl.DataFrame:
    df = _multi_ticker_bars(n_tickers, n_days)
    df = compute_features(df)
    df = generate_all(df)
    df = generate_all_with_forward_returns(df, horizons=[1, 5, 21])
    return df


SIGNAL_COLS = [
    "signal_mean_reversion_21d",
    "signal_mean_reversion_63d",
    "signal_momentum_21d",
    "signal_momentum_63d",
]


# ── Config tests ─────────────────────────────────────────────────────

class TestPipelineConfig:
    def test_defaults_ic_weighted(self):
        cfg = EnsemblePipelineConfig()
        assert cfg.mode == "ic_weighted"
        assert cfg.ic_lookback == 63
        assert cfg.rebalance_frequency == 5

    def test_lightgbm_mode(self):
        cfg = EnsemblePipelineConfig(mode="lightgbm")
        assert cfg.mode == "lightgbm"
        assert cfg.lightgbm_target_horizon == 1

    def test_custom_config(self):
        cfg = EnsemblePipelineConfig(
            mode="ic_weighted",
            ic_lookback=21,
            weight_method="positive_ic",
            output_column="my_ensemble",
        )
        assert cfg.ic_lookback == 21
        assert cfg.weight_method == "positive_ic"
        assert cfg.output_column == "my_ensemble"

    def test_to_ensemble_config(self):
        cfg = EnsemblePipelineConfig(ic_lookback=42, rebalance_frequency=10)
        ec = cfg.to_ensemble_config()
        assert ec.ic_lookback == 42
        assert ec.rebalance_frequency == 10


# ── IC-weighted pipeline tests ───────────────────────────────────────

class TestICWeightedPipeline:
    def test_run_produces_ensemble_column(self):
        df = _prepare_signals(5, 200)
        pipeline = EnsemblePipeline()
        result = pipeline.run(df)
        assert "signal_ensemble" in result.columns

    def test_run_preserves_input_columns(self):
        df = _prepare_signals(5, 200)
        pipeline = EnsemblePipeline()
        result = pipeline.run(df)
        for col in df.columns:
            assert col in result.columns

    def test_run_preserves_row_count(self):
        df = _prepare_signals(5, 200)
        pipeline = EnsemblePipeline()
        result = pipeline.run(df)
        assert len(result) == len(df)

    def test_run_with_explicit_signals(self):
        df = _prepare_signals(5, 200)
        cfg = EnsemblePipelineConfig(signal_columns=SIGNAL_COLS[:2])
        pipeline = EnsemblePipeline(cfg)
        result = pipeline.run(df)
        assert "signal_ensemble" in result.columns

    def test_run_with_custom_output_column(self):
        df = _prepare_signals(5, 200)
        cfg = EnsemblePipelineConfig(output_column="my_ensemble")
        pipeline = EnsemblePipeline(cfg)
        result = pipeline.run(df)
        assert "my_ensemble" in result.columns
        assert "signal_ensemble" not in result.columns

    def test_run_with_different_weight_methods(self):
        df = _prepare_signals(5, 200)
        for method in ["abs_ic", "rank_ic", "positive_ic"]:
            cfg = EnsemblePipelineConfig(weight_method=method)
            pipeline = EnsemblePipeline(cfg)
            result = pipeline.run(df)
            assert "signal_ensemble" in result.columns

    def test_run_no_signal_columns_raises(self):
        df = _multi_ticker_bars(2, 50)
        pipeline = EnsemblePipeline()
        with pytest.raises(ValueError, match="No signal columns"):
            pipeline.run(df)

    def test_run_missing_signal_columns_raises(self):
        df = _prepare_signals(5, 200)
        cfg = EnsemblePipelineConfig(signal_columns=["signal_fake"])
        pipeline = EnsemblePipeline(cfg)
        with pytest.raises(ValueError, match="missing"):
            pipeline.run(df)

    def test_run_track_weights_off(self):
        df = _prepare_signals(5, 200)
        cfg = EnsemblePipelineConfig(track_weights=False)
        pipeline = EnsemblePipeline(cfg)
        result = pipeline.run(df)
        w_cols = [c for c in result.columns if c.startswith("w_")]
        assert len(w_cols) == 0

    def test_ensemble_score_bounded(self):
        df = _prepare_signals(5, 200)
        pipeline = EnsemblePipeline()
        result = pipeline.run(df)
        scores = result["signal_ensemble"].drop_nulls()
        assert (scores >= -1.0).all()
        assert (scores <= 1.0).all()


# ── LightGBM pipeline tests ─────────────────────────────────────────

class TestLightGBMPipeline:
    def test_run_produces_prediction(self):
        df = _prepare_signals(5, 200)
        cfg = EnsemblePipelineConfig(mode="lightgbm")
        pipeline = EnsemblePipeline(cfg)
        result = pipeline.run(df, train_end="2020-06-01")
        assert "ensemble_prediction" in result.columns

    def test_run_prediction_range(self):
        df = _prepare_signals(5, 200)
        cfg = EnsemblePipelineConfig(mode="lightgbm")
        pipeline = EnsemblePipeline(cfg)
        result = pipeline.run(df, train_end="2020-06-01")
        probs = result["ensemble_prediction"].drop_nulls()
        assert (probs >= 0.0).all()
        assert (probs <= 1.0).all()

    def test_run_preserves_input_columns(self):
        df = _prepare_signals(5, 200)
        cfg = EnsemblePipelineConfig(mode="lightgbm")
        pipeline = EnsemblePipeline(cfg)
        result = pipeline.run(df, train_end="2020-06-01")
        for col in df.columns:
            assert col in result.columns

    def test_run_custom_signals(self):
        df = _prepare_signals(5, 200)
        cfg = EnsemblePipelineConfig(
            mode="lightgbm",
            signal_columns=SIGNAL_COLS[:2],
        )
        pipeline = EnsemblePipeline(cfg)
        result = pipeline.run(df, train_end="2020-06-01")
        assert "ensemble_prediction" in result.columns

    def test_run_val_split(self):
        df = _prepare_signals(5, 300)
        cfg = EnsemblePipelineConfig(mode="lightgbm")
        pipeline = EnsemblePipeline(cfg)
        result = pipeline.run(
            df,
            train_end="2021-01-01",
            val_start="2020-06-01",
        )
        assert "ensemble_prediction" in result.columns

    def test_run_past_cutoff_raises(self):
        df = _prepare_signals(5, 200)
        cfg = EnsemblePipelineConfig(mode="lightgbm")
        pipeline = EnsemblePipeline(cfg)
        with pytest.raises(ValueError, match="hold-back"):
            pipeline.run(df, train_end="2025-01-01")


# ── Validation tests ─────────────────────────────────────────────────

class TestPipelineValidation:
    def test_validate_returns_metrics(self):
        df = _prepare_signals(5, 250)
        pipeline = EnsemblePipeline()
        result = pipeline.run(df)
        em = pipeline.validate(result, target_col="forward_return_1")
        assert isinstance(em, EnsembleMetrics)
        assert -1.0 <= em.ensemble_flat_ic <= 1.0

    def test_validate_signals(self):
        df = _prepare_signals(5, 250)
        pipeline = EnsemblePipeline()
        sig_metrics = pipeline.validate_signals(df, target_col="forward_return_1")
        assert isinstance(sig_metrics, list)
        assert len(sig_metrics) > 0
        assert isinstance(sig_metrics[0], SignalMetrics)

    def test_validate_dominates_flag(self):
        df = _prepare_signals(5, 250)
        pipeline = EnsemblePipeline()
        result = pipeline.run(df)
        em = pipeline.validate(result, target_col="forward_return_1")
        assert isinstance(em.dominates_all, bool)

    def test_report_returns_string(self):
        df = _prepare_signals(5, 250)
        pipeline = EnsemblePipeline()
        result = pipeline.run(df)
        report = pipeline.report(result, target_col="forward_return_1")
        assert isinstance(report, str)
        assert "ENSEMBLE" in report

    def test_lightgbm_validate(self):
        df = _prepare_signals(5, 250)
        cfg = EnsemblePipelineConfig(mode="lightgbm")
        pipeline = EnsemblePipeline(cfg)
        result = pipeline.run(df, train_end="2020-06-01")
        em = pipeline.validate(
            result,
            ensemble_col="ensemble_prediction",
            target_col="forward_return_1",
        )
        assert isinstance(em, EnsembleMetrics)


# ── Weight stats tests ───────────────────────────────────────────────

class TestWeightStats:
    def test_weight_stats_returns_list(self):
        df = _prepare_signals(5, 250)
        pipeline = EnsemblePipeline()
        pipeline.run(df)
        stats = pipeline.weight_stats()
        assert isinstance(stats, list)
        assert len(stats) > 0

    def test_weight_stats_values(self):
        df = _prepare_signals(5, 250)
        pipeline = EnsemblePipeline()
        pipeline.run(df)
        stats = pipeline.weight_stats()
        for ws in stats:
            assert ws.mean >= 0.0
            assert ws.max <= 1.0

    def test_weight_stats_requires_run(self):
        pipeline = EnsemblePipeline()
        with pytest.raises(RuntimeError, match="run"):
            pipeline.weight_stats()

    def test_weight_stats_ic_mode_only(self):
        cfg = EnsemblePipelineConfig(mode="lightgbm")
        pipeline = EnsemblePipeline(cfg)
        with pytest.raises(RuntimeError, match="ic_weighted"):
            pipeline.weight_stats()


# ── Legacy run_ensemble compatibility ───────────────────────────────

class TestLegacyRunEnsemble:
    def test_run_ensemble_produces_output(self):
        df = _prepare_signals(5, 200)
        result = run_ensemble(df)
        assert "signal_ensemble" in result.columns

    def test_run_ensemble_with_config(self):
        from src.ensemble.base import EnsembleConfig
        df = _prepare_signals(5, 200)
        config = EnsembleConfig(ic_lookback=21, output_column="legacy_out")
        result = run_ensemble(df, config=config)
        assert "legacy_out" in result.columns


# ── End-to-end: real gold data ───────────────────────────────────────

class TestRealDataPipeline:
    def _load_gold(self, years=None):
        gold_dir = root / "data" / "gold" / "daily"
        if years is None:
            years = sorted(d.name for d in gold_dir.iterdir() if d.is_dir())
        frames = []
        for y in years:
            for f in (gold_dir / y).rglob("part-0.parquet"):
                frames.append(pl.read_parquet(str(f)))
        return pl.concat(frames, how="vertical_relaxed").sort(["ticker", "date"])

    def test_ic_pipeline_on_gold(self):
        df = self._load_gold(["year=2020", "year=2021", "year=2022"])
        if df.is_empty():
            pytest.skip("No gold data")
        df = compute_features(df)
        df = generate_all(df)
        df = generate_all_with_forward_returns(df, horizons=[1, 5, 21])

        pipeline = EnsemblePipeline()
        result = pipeline.run(df)
        assert "signal_ensemble" in result.columns
        assert len(result) == len(df)

    def test_lightgbm_pipeline_on_gold(self):
        df = self._load_gold(["year=2020", "year=2021", "year=2022"])
        if df.is_empty():
            pytest.skip("No gold data")
        df = compute_features(df)
        df = generate_all(df)
        df = generate_all_with_forward_returns(df, horizons=[1, 5, 21])

        cfg = EnsemblePipelineConfig(mode="lightgbm")
        pipeline = EnsemblePipeline(cfg)
        result = pipeline.run(df, train_end="2022-07-01")
        assert "ensemble_prediction" in result.columns

    def test_validation_on_gold(self):
        df = self._load_gold(["year=2020", "year=2021", "year=2022"])
        if df.is_empty():
            pytest.skip("No gold data")
        df = compute_features(df)
        df = generate_all(df)
        df = generate_all_with_forward_returns(df, horizons=[1, 5, 21])

        pipeline = EnsemblePipeline()
        result = pipeline.run(df)
        em = pipeline.validate(result, target_col="forward_return_1")
        assert len(em.signals) >= 4
        assert -1.0 <= em.ensemble_flat_ic <= 1.0

    def test_full_pipeline_validation_on_gold(self):
        """Full pipeline: signals → ensemble → validate → report."""
        df = self._load_gold(["year=2020", "year=2021", "year=2022"])
        if df.is_empty():
            pytest.skip("No gold data")
        df = compute_features(df)
        df = generate_all(df)
        df = generate_all_with_forward_returns(df, horizons=[1, 5, 21])

        pipeline = EnsemblePipeline()
        result = pipeline.run(df)
        report = pipeline.report(result, target_col="forward_return_1")
        assert "ENSEMBLE" in report
        assert "INDIVIDUAL SIGNALS" in report