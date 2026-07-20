"""Ensemble tests — LightGBM meta-learner and IC-weighted validation."""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.features.pipeline import compute_features
from src.signals.pipeline import generate_all_with_forward_returns
from src.signals import mean_reversion, momentum  # noqa: F401
from src.ensemble.lightgbm import (
    LightGBMEnsemble,
    LightGBMEnsembleConfig,
    HOLD_BACK_CUTOFF,
)
from src.ensemble.ic_weighted import (
    ICWeightedEnsemble,
    compute_cross_sectional_ic,
    compute_rolling_ic,
)
from src.ensemble.validation import (
    SignalMetrics,
    EnsembleMetrics,
    WeightStats,
    compute_cs_ic_stats,
    signal_metrics,
    validate_signals,
    validate_ensemble,
    weight_report,
    format_metrics_table,
    format_ensemble_report,
)


def _trading_dates(start: date, n: int) -> list[date]:
    dates: list[date] = []
    d = start
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    return dates


def _make_bars(ticker: str, n: int = 150, base: float = 100.0,
               start: date = date(2019, 1, 3), seed: int = 0) -> pl.DataFrame:
    dates = _trading_dates(start, n)
    rng = np.random.RandomState(seed)
    closes = [base]
    for i in range(1, n):
        closes.append(round(closes[-1] * (1 + 0.0005 + 0.02 * rng.randn()), 2))
    return pl.DataFrame({
        "ticker": [ticker] * n,
        "date": dates,
        "open": closes,
        "high": [c + abs(x) for c, x in zip(closes, rng.randn(n))],
        "low": [max(c - abs(x), 1.0) for c, x in zip(closes, rng.randn(n))],
        "close": closes,
        "volume": [1_000_000 + int(x * 100_000) for x in rng.rand(n)],
    })


def _signal_df(n_ticks: int = 5, n_days: int = 150) -> pl.DataFrame:
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"][:n_ticks]
    frames = [_make_bars(t, n_days, 100.0 + i * 50, seed=i) for i, t in enumerate(tickers)]
    df = pl.concat(frames, how="vertical_relaxed").sort(["ticker", "date"])
    df = compute_features(df)
    df = generate_all_with_forward_returns(df, horizons=[1, 5, 21])
    return df


@pytest.fixture
def df() -> pl.DataFrame:
    return _signal_df()


# ── Config tests ─────────────────────────────────────────────────────

class TestConfig:
    def test_hold_back_cutoff(self):
        assert HOLD_BACK_CUTOFF == date(2023, 1, 1)

    def test_config_defaults(self):
        c = LightGBMEnsembleConfig()
        assert c.signal_columns == []
        assert c.target_horizon == 1


# ── Fitting tests ────────────────────────────────────────────────────

class TestFit:
    def test_fit(self, df):
        LightGBMEnsemble().fit(df, train_end="2020-06-01")

    def test_fit_chaining(self, df):
        ens = LightGBMEnsemble()
        assert ens.fit(df, train_end="2020-06-01") is ens

    def test_fit_date_obj(self, df):
        LightGBMEnsemble().fit(df, train_end=date(2020, 6, 1))

    def test_fit_past_cutoff_raises(self, df):
        with pytest.raises(ValueError, match="hold-back"):
            LightGBMEnsemble().fit(df, train_end="2025-01-01")

    def test_fit_no_data_raises(self, df):
        with pytest.raises(ValueError, match="No training data"):
            LightGBMEnsemble().fit(df, train_end="2018-01-01")

    def test_fit_bad_horizon_raises(self, df):
        with pytest.raises(ValueError, match="Missing required columns"):
            LightGBMEnsemble(config=LightGBMEnsembleConfig(target_horizon=999)).fit(df, train_end="2020-06-01")

    def test_fit_no_signals_raises(self):
        d = pl.DataFrame({"date": [date(2019, 1, 1)], "close": [100.0]})
        with pytest.raises(ValueError, match="No signal columns"):
            LightGBMEnsemble().fit(d, train_end="2020-01-01")

    def test_fit_missing_feature_raises(self, df):
        with pytest.raises(ValueError, match="Feature columns missing"):
            LightGBMEnsemble(config=LightGBMEnsembleConfig(signal_columns=["signal_fake"])).fit(df, train_end="2020-06-01")


# ── Auto-detection ───────────────────────────────────────────────────

class TestAutoDetect:
    def test_auto_detect(self, df):
        ens = LightGBMEnsemble().fit(df, train_end="2020-06-01")
        assert len(ens._feature_cols) >= 4
        assert all(c.startswith("signal_") for c in ens._feature_cols)

    def test_explicit_cols(self, df):
        cols = ["signal_momentum_21d", "signal_momentum_63d"]
        ens = LightGBMEnsemble(config=LightGBMEnsembleConfig(signal_columns=cols))
        ens.fit(df, train_end="2020-06-01")
        assert ens._feature_cols == cols


# ── Predictions ──────────────────────────────────────────────────────

class TestPredict:
    def test_predict_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            LightGBMEnsemble().predict(pl.DataFrame())

    def test_predict_column(self, df):
        ens = LightGBMEnsemble().fit(df, train_end="2020-06-01")
        out = ens.predict(df, return_column="p")
        assert "p" in out.columns

    def test_predict_shape(self, df):
        ens = LightGBMEnsemble().fit(df, train_end="2020-06-01")
        assert len(ens.predict(df)) == len(df)

    def test_predict_range(self, df):
        ens = LightGBMEnsemble().fit(df, train_end="2020-06-01")
        p = ens.predict(df, return_column="p")["p"].drop_nulls().to_list()
        assert all(0.0 <= v <= 1.0 for v in p)

    def test_predict_variance(self, df):
        ens = LightGBMEnsemble().fit(df, train_end="2020-06-01")
        p = ens.predict(df, return_column="p")["p"].drop_nulls().to_list()
        assert len(set(p)) > 1

    def test_direction_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            LightGBMEnsemble().predict_direction(pl.DataFrame())

    def test_direction_values(self, df):
        ens = LightGBMEnsemble().fit(df, train_end="2020-06-01")
        vals = set(ens.predict_direction(df, return_column="dir")["dir"].drop_nulls().to_list())
        assert vals.issubset({-1.0, 0.0, 1.0})

    def test_direction_spread(self, df):
        ens = LightGBMEnsemble().fit(df, train_end="2020-06-01")
        vals = ens.predict_direction(df)["ensemble_direction"].drop_nulls().to_list()
        assert len(set(vals)) > 1


# ── Feature importance ───────────────────────────────────────────────

class TestImportance:
    def test_importance_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            LightGBMEnsemble().feature_importance()

    def test_importance(self, df):
        ens = LightGBMEnsemble().fit(df, train_end="2020-06-01")
        imp = ens.feature_importance()
        assert len(imp) == len(ens._feature_cols)
        assert all(v >= 0 for v in imp.values())
        assert sum(imp.values()) > 0


# ── Metrics ──────────────────────────────────────────────────────────

class TestMetrics:
    def test_metrics(self, df):
        ens = LightGBMEnsemble().fit(df, train_end="2020-06-01")
        m = ens.training_metrics()
        assert "train_auc" in m and "train_logloss" in m
        assert 0.0 <= m["train_auc"] <= 1.0

    def test_val_split(self, df):
        ens = LightGBMEnsemble()
        ens.fit(df, train_end="2020-06-01", val_start="2020-06-02")
        assert "train_auc" in ens.training_metrics()


# ── Hold-back safety ─────────────────────────────────────────────────

class TestHoldBack:
    def test_at_cutoff(self, df):
        LightGBMEnsemble().fit(df, train_end="2023-01-01")

    def test_before_cutoff(self, df):
        LightGBMEnsemble().fit(df, train_end="2020-01-01")


# ── End-to-end ───────────────────────────────────────────────────────

class TestE2E:
    def test_pipeline(self, df):
        ens = LightGBMEnsemble().fit(df, train_end="2020-06-01")
        out = ens.predict(df, return_column="ep")
        assert "ep" in out.columns
        assert len(out["ep"].drop_nulls()) > 0
        assert len(ens.feature_importance()) >= 4
        assert ens.training_metrics()["train_auc"] > 0.0

    def test_horizons(self, df):
        for h in [1, 5]:
            ens = LightGBMEnsemble(config=LightGBMEnsembleConfig(target_horizon=h))
            ens.fit(df, train_end="2020-06-01")
            assert ens._fitted

    def test_custom_params(self, df):
        cfg = LightGBMEnsembleConfig(lgbm_params={"n_estimators": 50, "num_leaves": 8})
        ens = LightGBMEnsemble(config=cfg).fit(df, train_end="2020-06-01")
        assert ens.model_.n_estimators_ == 50

    def test_additional_features(self, df):
        df2 = df.with_columns(
            pl.col("close").pct_change().over("ticker").alias("my_feat")
        )
        ens = LightGBMEnsemble(config=LightGBMEnsembleConfig(additional_features=["my_feat"]))
        ens.fit(df2, train_end="2020-06-01")
        assert "my_feat" in ens._feature_cols


# ── Real data ────────────────────────────────────────────────────────

class TestRealData:
    def test_gold_data(self):
        gd = root / "data" / "gold" / "daily" / "year=2022"
        files = list(gd.glob("ticker=*/part-0.parquet"))
        if not files:
            pytest.skip("No gold data")
        df = pl.concat([pl.read_parquet(str(f)) for f in sorted(files)],
                       how="vertical_relaxed").sort(["ticker", "date"])
        df = compute_features(df)
        df = generate_all_with_forward_returns(df, horizons=[1, 5, 21])
        ens = LightGBMEnsemble().fit(df, train_end="2022-07-01")
        assert ens._fitted
        out = ens.predict(df)
        assert "ensemble_prediction" in out.columns
        assert 0.0 <= ens.training_metrics()["train_auc"] <= 1.0


# ═════════════════════════════════════════════════════════════════════
# IC-weighted ensemble and validation tests
# ═════════════════════════════════════════════════════════════════════

import math as _math


def _ic_bars(ticker: str, n: int = 250, base_close: float = 100.0, phase: float = 0.0):
    dates = _trading_dates(date(2023, 1, 3), n)
    closes = []
    c = base_close
    for i in range(n):
        c = c * (1.0 + 0.005 + 0.015 * _math.sin(i * 0.1 + phase))
        closes.append(round(c, 2))
    return pl.DataFrame({
        "ticker": [ticker] * n, "date": dates, "open": closes,
        "high": [c + 1 for c in closes], "low": [max(c - 1, 1.0) for c in closes],
        "close": closes,
        "volume": [1_000_000 + (i % 10) * 100_000 for i in range(n)],
    })


def _ic_multi_bars(n_tickers: int = 5, n_days: int = 250) -> pl.DataFrame:
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"][:n_tickers]
    frames = [_ic_bars(t, n_days, 80.0 + i * 40, i * 0.7)
              for i, t in enumerate(tickers)]
    return pl.concat(frames, how="vertical_relaxed").sort(["ticker", "date"])


def _ic_signal_df(n_ticks: int = 5, n_days: int = 250) -> pl.DataFrame:
    df = _ic_multi_bars(n_ticks, n_days)
    df = compute_features(df)
    from src.signals.pipeline import generate_all
    df = generate_all(df)
    return df.with_columns(
        ((pl.col("close").shift(-1).over("ticker")) / pl.col("close") - 1)
        .alias("forward_return_1")
    )


class TestCrossSectionalIC:
    def test_returns_correct_columns(self):
        df = _ic_multi_bars(5, 200)
        df = df.with_columns(pl.col("close").rank("average").over("date").alias("signal"))
        df = df.with_columns(
            ((pl.col("close").shift(-1).over("ticker")) / pl.col("close") - 1).alias("target"))
        cs = compute_cross_sectional_ic(df, "signal", "target")
        assert "date" in cs.columns and "cs_ic" in cs.columns

    def test_non_empty_multi_ticker(self):
        df = _ic_multi_bars(5, 200)
        df = df.with_columns(pl.col("close").rank("average").over("date").alias("signal"))
        df = df.with_columns(
            ((pl.col("close").shift(-1).over("ticker")) / pl.col("close") - 1).alias("target"))
        assert len(compute_cross_sectional_ic(df, "signal", "target")) > 0

    def test_empty_single_ticker(self):
        df = _ic_bars("SINGLE", 100)
        df = df.with_columns(pl.col("close").rank("average").over("date").alias("signal"))
        df = df.with_columns(
            ((pl.col("close").shift(-1).over("ticker")) / pl.col("close") - 1).alias("target"))
        assert compute_cross_sectional_ic(df, "signal", "target").is_empty()


class TestRollingIC:
    def test_returns_columns(self):
        df = _ic_multi_bars(5, 200)
        df = df.with_columns(pl.col("close").rank("average").over("date").alias("signal"))
        df = df.with_columns(
            ((pl.col("close").shift(-1).over("ticker")) / pl.col("close") - 1).alias("target"))
        rolling = compute_rolling_ic(df, "signal", "target", window=21)
        assert "date" in rolling.columns and "rolling_ic" in rolling.columns


class TestICWeightedEnsemble:
    def test_produces_ensemble_score(self):
        df = _ic_signal_df()
        ens = ICWeightedEnsemble(ic_window=21, rebalance_freq=5)
        out = ens.transform(df, signal_cols=["signal_mean_reversion_21d", "signal_momentum_21d"],
                           target_col="forward_return_1")
        assert "ensemble_score" in out.columns

    def test_weights_sum_to_one(self):
        df = _ic_signal_df()
        ens = ICWeightedEnsemble(ic_window=21, rebalance_freq=5)
        out = ens.transform(df, signal_cols=["signal_mean_reversion_21d", "signal_momentum_21d"],
                           target_col="forward_return_1")
        w_sum = out["w_signal_mean_reversion_21d"] + out["w_signal_momentum_21d"]
        assert (w_sum > 0.99).all() and (w_sum < 1.01).all()

    def test_predict_requires_transform(self):
        ens = ICWeightedEnsemble()
        with pytest.raises(RuntimeError, match="transform"):
            ens.predict(pl.DataFrame(), signal_cols=["signal_mean_reversion_21d"])

    def test_all_four_signals(self):
        df = _ic_signal_df()
        ens = ICWeightedEnsemble(ic_window=21, rebalance_freq=5)
        out = ens.transform(df, signal_cols=[
            "signal_mean_reversion_21d", "signal_mean_reversion_63d",
            "signal_momentum_21d", "signal_momentum_63d"], target_col="forward_return_1")
        assert "ensemble_score" in out.columns
        for sig in ["signal_mean_reversion_21d", "signal_mean_reversion_63d",
                     "signal_momentum_21d", "signal_momentum_63d"]:
            assert f"w_{sig}" in out.columns


class TestCSICStats:
    def test_returns_three_values(self):
        df = _ic_signal_df(5, 200)
        mean, std, pos = compute_cs_ic_stats(df, "signal_momentum_21d", "forward_return_1")
        assert -1.0 <= mean <= 1.0 and std >= 0.0 and 0.0 <= pos <= 1.0

    def test_zeros_for_missing_column(self):
        mean, std, pos = compute_cs_ic_stats(_ic_multi_bars(2, 50), "nonexistent", "target")
        assert mean == 0.0 and std == 0.0 and pos == 0.0


class TestValidateSignals:
    def test_returns_list_of_metrics(self):
        df = _ic_signal_df(5, 250)
        metrics = validate_signals(df, signal_cols=["signal_mean_reversion_21d", "signal_momentum_21d"])
        assert len(metrics) == 2
        assert abs(metrics[0].flat_ic) >= abs(metrics[1].flat_ic)

    def test_skips_missing_columns(self):
        df = _ic_signal_df(5, 100)
        metrics = validate_signals(df, signal_cols=["signal_momentum_21d", "signal_nonexistent"])
        assert len(metrics) == 1


class TestValidateEnsemble:
    def test_returns_ensemble_metrics(self):
        df = _ic_signal_df(5, 250)
        ens = ICWeightedEnsemble(ic_window=21, rebalance_freq=5)
        df = ens.transform(df, signal_cols=["signal_mean_reversion_21d", "signal_momentum_21d"],
                          target_col="forward_return_1")
        em = validate_ensemble(df, ensemble_col="ensemble_score",
                              signal_cols=["signal_mean_reversion_21d", "signal_momentum_21d"])
        assert isinstance(em, EnsembleMetrics)
        assert -1.0 <= em.ensemble_flat_ic <= 1.0
        assert len(em.signals) == 2

    def test_dominates_all_flag(self):
        df = _ic_signal_df(5, 250)
        ens = ICWeightedEnsemble(ic_window=21, rebalance_freq=5)
        df = ens.transform(df, signal_cols=["signal_mean_reversion_21d", "signal_momentum_21d"],
                          target_col="forward_return_1")
        em = validate_ensemble(df, ensemble_col="ensemble_score",
                              signal_cols=["signal_mean_reversion_21d", "signal_momentum_21d"])
        assert isinstance(em.dominates_all, bool)


class TestWeightReport:
    def test_requires_transform(self):
        with pytest.raises(RuntimeError, match="transform"):
            weight_report(ICWeightedEnsemble())

    def test_returns_weight_stats(self):
        df = _ic_signal_df(5, 250)
        ens = ICWeightedEnsemble(ic_window=21, rebalance_freq=5)
        ens.transform(df, signal_cols=["signal_mean_reversion_21d", "signal_momentum_21d"],
                     target_col="forward_return_1")
        stats = weight_report(ens)
        assert len(stats) == 2
        for ws in stats:
            assert isinstance(ws, WeightStats) and ws.mean >= 0.0 and ws.max <= 1.0


class TestFormatReports:
    def test_metrics_table(self):
        metrics = [SignalMetrics(signal="signal_a", flat_ic=0.05, win_rate=0.55,
                                 cs_ic_mean=0.04, cs_ic_std=0.03, cs_ic_positive_pct=0.60)]
        assert "signal_a" in format_metrics_table(metrics)

    def test_ensemble_report_pass(self):
        em = EnsembleMetrics(ensemble="e", ensemble_flat_ic=0.06, ensemble_win_rate=0.57,
                            ensemble_cs_ic_mean=0.05, ensemble_cs_ic_std=0.03,
                            ensemble_cs_ic_positive_pct=0.65, dominates_all=True)
        assert "[PASS]" in format_ensemble_report(em)

    def test_ensemble_report_warn(self):
        em = EnsembleMetrics(ensemble="e", ensemble_flat_ic=0.02, ensemble_win_rate=0.50,
                            ensemble_cs_ic_mean=0.01, ensemble_cs_ic_std=0.03,
                            ensemble_cs_ic_positive_pct=0.50, dominates_all=False)
        assert "[WARN]" in format_ensemble_report(em)


class TestRealDataIC:
    def _load_gold(self, years=None):
        gold_dir = root / "data" / "gold" / "daily"
        if years is None:
            years = sorted(d.name for d in gold_dir.iterdir() if d.is_dir())
        frames = []
        for y in years:
            for f in (gold_dir / y).rglob("part-0.parquet"):
                frames.append(pl.read_parquet(str(f)))
        return pl.concat(frames, how="vertical_relaxed").sort(["ticker", "date"])

    def test_ensemble_on_real_data(self):
        df = self._load_gold(["year=2020", "year=2021", "year=2022"])
        if df.is_empty():
            pytest.skip("No gold data")
        df = compute_features(df)
        from src.signals.pipeline import generate_all
        df = generate_all(df)
        df = df.with_columns(
            ((pl.col("close").shift(-1).over("ticker")) / pl.col("close") - 1).alias("forward_return_1"))
        ens = ICWeightedEnsemble(ic_window=63, rebalance_freq=5)
        df = ens.transform(df, signal_cols=[
            "signal_mean_reversion_21d", "signal_mean_reversion_63d",
            "signal_momentum_21d", "signal_momentum_63d"], target_col="forward_return_1")
        assert "ensemble_score" in df.columns
        score = df["ensemble_score"].drop_nulls()
        assert len(score) > 0
        assert (score <= 1.0).all() and (score >= -1.0).all()

    def test_validation_on_real_data(self):
        df = self._load_gold(["year=2020", "year=2021", "year=2022"])
        if df.is_empty():
            pytest.skip("No gold data")
        df = compute_features(df)
        from src.signals.pipeline import generate_all
        df = generate_all(df)
        df = df.with_columns(
            ((pl.col("close").shift(-1).over("ticker")) / pl.col("close") - 1).alias("forward_return_1"))
        signal_cols = ["signal_mean_reversion_21d", "signal_mean_reversion_63d",
                       "signal_momentum_21d", "signal_momentum_63d"]
        ens = ICWeightedEnsemble(ic_window=63, rebalance_freq=5)
        df = ens.transform(df, signal_cols=signal_cols, target_col="forward_return_1")
        em = validate_ensemble(df, ensemble_col="ensemble_score", signal_cols=signal_cols,
                              target_col="forward_return_1", horizons=[1, 5, 21])
        assert len(em.signals) == 4
        assert len(weight_report(ens)) == 4
        assert len(format_ensemble_report(em)) > 0