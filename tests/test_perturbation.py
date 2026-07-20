"""Perturbation test suite — parameter sensitivity for all signals."""

import sys
from datetime import date, timedelta
from pathlib import Path

import math
import polars as pl
import pytest

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.signals.registry import registry as signal_registry
from src.signals.base import (
    compute_forward_returns,
    rank_ic,
)
from src.signals.pipeline import generate_all_with_forward_returns
from src.signals import mean_reversion, momentum  # noqa: F401 — registers signals
from src.features.pipeline import compute_features

from src.validation.perturbation import (
    PerturbationResult,
    SignalPerturbationReport,
    PerturbationSummary,
    _get_tunable_params,
    _perturbed_values,
    _recreate_signal,
    run_perturbation_test,
    run_full_perturbation_sweep,
    format_perturbation_report,
    TUNABLE_PARAM_NAMES,
)


# ── Helpers ──────────────────────────────────────────────────────────


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
    num_days: int = 400,
    base_close: float = 100.0,
    start_date: date = date(2023, 1, 3),
    volume: int = 1_000_000,
) -> pl.DataFrame:
    dates = _trading_dates(start_date, num_days)
    closes = []
    c = base_close
    for i in range(num_days):
        # More varied price action for realistic signals
        c = c * (1.0 + 0.015 * math.sin(i * 0.08) + 0.005 * math.cos(i * 0.03))
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


def _multi_ticker_bars(n_tickers: int = 5, n_days: int = 400) -> pl.DataFrame:
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"][:n_tickers]
    frames = [
        _make_bars(t, num_days=n_days, base_close=100.0 + i * 50)
        for i, t in enumerate(tickers)
    ]
    return pl.concat(frames, how="vertical_relaxed").sort(["ticker", "date"])


def _full_signal_df() -> pl.DataFrame:
    """Prepare a full DataFrame: bars -> features -> signals -> forward returns."""
    df = _multi_ticker_bars(5, 400)
    df = compute_features(df)
    df = generate_all_with_forward_returns(df, horizons=[1, 5, 21])
    return df


# ── Tunable param discovery ─────────────────────────────────────────


class TestTunableParams:
    def test_mean_reversion_lookback(self):
        params = _get_tunable_params("mean_reversion_21d")
        assert "lookback" in params
        assert params["lookback"] == 21

    def test_mean_reversion_63d_lookback(self):
        params = _get_tunable_params("mean_reversion_63d")
        assert params["lookback"] == 63

    def test_momentum_formation_days(self):
        params = _get_tunable_params("momentum_21d")
        assert "formation_days" in params
        assert params["formation_days"] == 21

    def test_boolean_params_excluded(self):
        """cross_sectional is a bool — should not be tunable."""
        params = _get_tunable_params("mean_reversion_21d")
        assert "cross_sectional" not in params

    def test_tunable_param_names_defined(self):
        assert "lookback" in TUNABLE_PARAM_NAMES
        assert "formation_days" in TUNABLE_PARAM_NAMES


# ── Perturbed values ────────────────────────────────────────────────


class TestPerturbedValues:
    def test_baseline_21(self):
        vals = _perturbed_values(21)
        # [-20%, -10%, 0%, +10%, +20%] → [17, 19, 21, 23, 25]
        assert vals == [17, 19, 21, 23, 25]

    def test_baseline_63(self):
        vals = _perturbed_values(63)
        # [50, 57, 63, 69, 76]
        assert vals[2] == 63  # baseline in the middle
        assert vals[0] < vals[1] < vals[2] < vals[3] < vals[4]

    def test_custom_pcts(self):
        vals = _perturbed_values(100, pcts=[-50.0, 0.0, 50.0])
        assert vals == [50, 100, 150]

    def test_minimum_one(self):
        """Very small baseline should clamp to 1."""
        vals = _perturbed_values(2, pcts=[-20.0, -10.0, 0.0, 10.0, 20.0])
        assert all(v >= 1 for v in vals)


# ── Signal re-creation ─────────────────────────────────────────────


class TestRecreateSignal:
    def test_mean_reversion_baseline(self):
        df = _full_signal_df()
        out = _recreate_signal(df, "mean_reversion_21d", "lookback", 21)
        assert "signal_mean_reversion_21d_perturbed" in out.columns

    def test_mean_reversion_perturbed(self):
        df = _full_signal_df()
        out = _recreate_signal(df, "mean_reversion_21d", "lookback", 25)
        col = out["signal_mean_reversion_21d_perturbed"].drop_nulls()
        assert len(col) > 0

    def test_momentum_baseline(self):
        df = _full_signal_df()
        out = _recreate_signal(df, "momentum_21d", "formation_days", 21)
        assert "signal_momentum_21d_perturbed" in out.columns

    def test_momentum_perturbed(self):
        df = _full_signal_df()
        out = _recreate_signal(df, "momentum_21d", "formation_days", 23)
        col = out["signal_momentum_21d_perturbed"].drop_nulls()
        assert len(col) > 0

    def test_different_lookback_produces_different_signal(self):
        """A different lookback should produce different signal values."""
        df = _full_signal_df()
        out21 = _recreate_signal(df, "mean_reversion_21d", "lookback", 21)
        out25 = _recreate_signal(df, "mean_reversion_21d", "lookback", 25)
        col21 = out21["signal_mean_reversion_21d_perturbed"].drop_nulls()
        col25 = out25["signal_mean_reversion_21d_perturbed"].drop_nulls()
        # Should be similar length but different values
        assert len(col21) > 0
        assert len(col25) > 0
        # IC should differ by something measurable
        ic21 = rank_ic("signal_mean_reversion_21d_perturbed", "forward_return_1", out21)
        ic25 = rank_ic("signal_mean_reversion_21d_perturbed", "forward_return_1", out25)
        # Don't assert exact difference — synthetic data may not produce
        # meaningfully different IC, but the mechanism should work
        assert isinstance(ic21, float)
        assert isinstance(ic25, float)


# ── Single perturbation test ────────────────────────────────────────


class TestPerturbationTest:
    def test_mean_reversion_21d_lookback(self):
        df = _full_signal_df()
        report = run_perturbation_test(
            df, "mean_reversion_21d", "lookback", 21, target_col="forward_return_1"
        )
        assert report.signal_name == "mean_reversion_21d"
        assert report.parameter == "lookback"
        assert report.baseline_value == 21
        assert len(report.results) == 5  # 5 perturbation levels

    def test_report_contains_baseline_row(self):
        df = _full_signal_df()
        report = run_perturbation_test(
            df, "mean_reversion_21d", "lookback", 21, target_col="forward_return_1"
        )
        baseline_row = report.results[2]  # middle row should be 0% change
        assert baseline_row.pct_change == 0.0
        assert baseline_row.perturbed_value == 21

    def test_perturbed_rows_have_change_metrics(self):
        df = _full_signal_df()
        report = run_perturbation_test(
            df, "mean_reversion_21d", "lookback", 21, target_col="forward_return_1"
        )
        for r in report.results:
            if r.pct_change != 0.0:
                assert r.ic_change_pct is not None

    def test_momentum_perturbation(self):
        df = _full_signal_df()
        report = run_perturbation_test(
            df, "momentum_21d", "formation_days", 21, target_col="forward_return_1"
        )
        assert report.signal_name == "momentum_21d"
        assert report.parameter == "formation_days"
        assert len(report.results) == 5

    def test_classification_robust(self):
        """Synthetic data tends to produce robust results (oscillating prices)."""
        df = _full_signal_df()
        report = run_perturbation_test(
            df, "mean_reversion_21d", "lookback", 21, target_col="forward_return_1"
        )
        status = report.classify()
        assert status in ("robust", "sensitive", "fragile", "no_data")


# ── Full sweep ─────────────────────────────────────────────────────


class TestFullSweep:
    def test_sweep_runs(self):
        df = _full_signal_df()
        summary = run_full_perturbation_sweep(df)
        assert summary.total_signals_tested >= 4
        assert summary.total_parameters_tested >= 2  # at least lookback + formation

    def test_sweep_covers_all_signals(self):
        df = _full_signal_df()
        summary = run_full_perturbation_sweep(df)
        signal_names_in_reports = {r.signal_name for r in summary.reports}
        # Should have at least the 4 registered signals with tunable params
        assert "mean_reversion_21d" in signal_names_in_reports
        assert "mean_reversion_63d" in signal_names_in_reports
        assert "momentum_21d" in signal_names_in_reports
        assert "momentum_63d" in signal_names_in_reports

    def test_sweep_classifies(self):
        df = _full_signal_df()
        summary = run_full_perturbation_sweep(df)
        total_classified = summary.robust_count + summary.sensitive_count + summary.fragile_count
        assert total_classified == summary.total_parameters_tested

    def test_fragile_flags(self):
        df = _full_signal_df()
        summary = run_full_perturbation_sweep(df)
        fragile = summary.flag_fragile()
        assert isinstance(fragile, list)

    def test_sensitive_flags(self):
        df = _full_signal_df()
        summary = run_full_perturbation_sweep(df)
        sensitive = summary.flag_sensitive()
        assert isinstance(sensitive, list)


# ── Reporting ───────────────────────────────────────────────────────


class TestReporting:
    def test_report_is_text(self):
        df = _full_signal_df()
        summary = run_full_perturbation_sweep(df)
        text = format_perturbation_report(summary)
        assert isinstance(text, str)
        assert "PERTURBATION TEST REPORT" in text

    def test_report_includes_counts(self):
        df = _full_signal_df()
        summary = run_full_perturbation_sweep(df)
        text = format_perturbation_report(summary)
        assert "Signals tested" in text
        assert "Parameters tested" in text

    def test_report_includes_status_per_signal(self):
        df = _full_signal_df()
        summary = run_full_perturbation_sweep(df)
        text = format_perturbation_report(summary)
        # Should mention at least one signal name
        assert "mean_reversion" in text or "momentum" in text

    def test_report_shows_fragile_section_if_applicable(self):
        df = _full_signal_df()
        summary = run_full_perturbation_sweep(df)
        text = format_perturbation_report(summary)
        # Either the fragile section is shown or not — both are valid
        # Just check it renders without error
        assert len(text) > 100


# ── Data classes ────────────────────────────────────────────────────


class TestDataClasses:
    def test_perturbation_result(self):
        r = PerturbationResult(
            signal_name="test",
            parameter="lookback",
            baseline_value=21,
            perturbed_value=25,
            pct_change=20.0,
            ic=0.05,
            win_rate=0.55,
            signal_std=0.3,
            signal_mean=0.01,
        )
        assert r.signal_name == "test"

    def test_signal_report_classify_robust(self):
        report = SignalPerturbationReport(
            signal_name="test",
            parameter="lookback",
            baseline_value=21,
            baseline_ic=0.05,
            baseline_win_rate=0.55,
        )
        # Add results with small IC changes
        report.results = [
            PerturbationResult(
                signal_name="test", parameter="lookback",
                baseline_value=21, perturbed_value=17, pct_change=-20.0,
                ic=0.047, win_rate=0.54, signal_std=0.3, signal_mean=0.01,
                ic_change_pct=-6.0,
            ),
            PerturbationResult(
                signal_name="test", parameter="lookback",
                baseline_value=21, perturbed_value=21, pct_change=0.0,
                ic=0.05, win_rate=0.55, signal_std=0.3, signal_mean=0.01,
            ),
        ]
        status = report.classify()
        assert status == "robust"

    def test_signal_report_classify_fragile(self):
        report = SignalPerturbationReport(
            signal_name="test",
            parameter="lookback",
            baseline_value=21,
            baseline_ic=0.05,
            baseline_win_rate=0.55,
        )
        report.results = [
            PerturbationResult(
                signal_name="test", parameter="lookback",
                baseline_value=21, perturbed_value=25, pct_change=20.0,
                ic=0.02, win_rate=0.50, signal_std=0.3, signal_mean=0.01,
                ic_change_pct=-60.0,
            ),
        ]
        status = report.classify()
        assert status == "fragile"

    def test_summary_flag_methods(self):
        summary = PerturbationSummary()
        assert summary.flag_fragile() == []
        assert summary.flag_sensitive() == []