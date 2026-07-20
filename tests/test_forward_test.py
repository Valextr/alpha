"""Forward test framework tests.

Tests the validation/forward_test module with synthetic data,
verifying that IC, win rate, and comparison logic work correctly.
"""

import sys
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import pytest

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.validation.forward_test import (
    run_forward_test,
    generate_report,
    SignalForwardResult,
    ForwardTestResult,
    HOLD_BACK_START,
    TRAIN_END,
    VALIDATION_END,
)
from src.features.pipeline import compute_features
from src.signals.pipeline import generate_all_with_forward_returns


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
            "volume": [volume + (i % 10) * 100_000 for i in range(num_days)],
        }
    )


def _multi_ticker_bars(n_tickers: int = 5, n_days: int = 300, start: date = date(2023, 1, 3)) -> pl.DataFrame:
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"][:n_tickers]
    frames = [
        _make_bars(t, num_days=n_days, base_close=100.0 + i * 50, start_date=start)
        for i, t in enumerate(tickers)
    ]
    return pl.concat(frames, how="vertical_relaxed").sort(["ticker", "date"])


# ── Tests ────────────────────────────────────────────────────────────


class TestConstants:
    def test_hold_back_start(self):
        assert HOLD_BACK_START == date(2023, 1, 1)

    def test_train_end(self):
        assert TRAIN_END == date(2020, 1, 1)

    def test_validation_end(self):
        assert VALIDATION_END == date(2023, 1, 1)

    def test_segments_dont_overlap(self):
        # Train ends where validation begins; validation ends where hold-back begins
        assert TRAIN_END.year == 2020
        assert TRAIN_END < VALIDATION_END
        assert HOLD_BACK_START == VALIDATION_END


class TestRunForwardTest:
    def test_returns_results_for_all_signals(self):
        """Should return a result for every registered signal."""
        df = _multi_ticker_bars(5, 300)
        results = run_forward_test(df, horizons=[1, 5])
        assert len(results) >= 4  # at least 4 signals registered

    def test_result_has_valid_ic(self):
        """IC should be bounded in [-1, +1]."""
        df = _multi_ticker_bars(5, 300)
        results = run_forward_test(df, horizons=[1, 5])
        for r in results:
            for h, ic in r.ic_by_horizon.items():
                assert -1.0 <= ic <= 1.0, f"IC out of range for {r.signal_name} at {h}d"

    def test_result_has_valid_win_rate(self):
        """Win rate should be bounded in [0, 1]."""
        df = _multi_ticker_bars(5, 300)
        results = run_forward_test(df, horizons=[1])
        for r in results:
            for h, wr in r.win_rate_by_horizon.items():
                assert 0.0 <= wr <= 1.0, f"Win rate out of range for {r.signal_name}"

    def test_result_has_nonzero_count(self):
        """Signal results should have observations."""
        df = _multi_ticker_bars(5, 300)
        results = run_forward_test(df, horizons=[1])
        for r in results:
            assert r.count > 0, f"{r.signal_name} has zero observations"

    def test_result_has_per_ticker_ic(self):
        """Per-ticker IC should be populated."""
        df = _multi_ticker_bars(3, 300)
        results = run_forward_test(df, horizons=[1])
        for r in results:
            assert len(r.per_ticker_ic) > 0, f"No per-ticker IC for {r.signal_name}"

    def test_filtered_by_category(self):
        """Should only run signals in the specified category."""
        df = _multi_ticker_bars(5, 300)
        results = run_forward_test(df, horizons=[1], categories=["mean_reversion"])
        for r in results:
            assert "mean_reversion" in r.signal_name

    def test_ic_by_horizon_matches_requested(self):
        """IC keys should match the horizons we requested."""
        df = _multi_ticker_bars(5, 300)
        horizons = [1, 5, 21]
        results = run_forward_test(df, horizons=horizons)
        for r in results:
            for h in horizons:
                assert h in r.ic_by_horizon, f"Missing horizon {h} for {r.signal_name}"


class TestGenerateReport:
    def test_report_contains_all_signals(self):
        results = [SignalForwardResult(signal_name="signal_test")]
        report = generate_report(results, ["AAPL"], "2023-2026")
        assert "signal_test" in report.signals_evaluated

    def test_report_summary_is_non_empty(self):
        results = [SignalForwardResult(signal_name="signal_test", count=100, mean=0.0, std=0.5, ic_by_horizon={1: 0.05})]
        report = generate_report(results, ["AAPL"], "2023-2026")
        assert len(report.summary) > 0

    def test_report_date_range(self):
        date_range = "2023-01-02 to 2026-07-18"
        results = [SignalForwardResult(signal_name="signal_test")]
        report = generate_report(results, ["AAPL", "MSFT"], date_range)
        assert "2023" in report.actual_date_range


class TestEndToEndWithRealData:
    def test_hold_back_data_has_rows(self):
        """Verify the hold-back segment (2023+) should have data."""
        gold_dir = root / "data" / "gold" / "daily" / "year=2023"
        files = list(gold_dir.glob("ticker=*/part-0.parquet"))
        if not files:
            pytest.skip("No 2023 gold data available")

        frames = [pl.read_parquet(str(f)) for f in sorted(files)]
        df = pl.concat(frames, how="vertical_relaxed").sort(["ticker", "date"])
        df = df.filter(pl.col("date") >= date(2023, 1, 1))
        assert not df.is_empty(), "Hold-back data should have rows"

    def test_forward_test_on_holdback_data(self):
        """Run a full forward test on hold-back data."""
        # Load all hold-back data
        gold_dir = root / "data" / "gold" / "daily"
        holdback_years = ["year=2023", "year=2024", "year=2025", "year=2026"]
        files = []
        for yr in holdback_years:
            yr_dir = gold_dir / yr
            if yr_dir.exists():
                files.extend(list(yr_dir.glob("ticker=*/part-0.parquet")))

        if not files:
            pytest.skip("No hold-back gold data")

        frames = [pl.read_parquet(str(f)) for f in sorted(files)]
        df = pl.concat(frames, how="vertical_relaxed").sort(["ticker", "date"])
        df = df.filter(pl.col("date") >= date(2023, 1, 1))

        results = run_forward_test(df, horizons=[1, 5, 21])
        assert len(results) >= 4, f"Expected >= 4 signals, got {len(results)}"

        # IC should be bounded
        for r in results:
            for h, ic in r.ic_by_horizon.items():
                assert -1.0 <= ic <= 1.0, f"IC out of range: {r.signal_name} {h}d = {ic}"
            assert r.count > 0, f"{r.signal_name} has no observations"


class TestComparisonConstants:
    def test_train_validation_holdback_coverage(self):
        """Verify the segments roughly match the 60/20/20 split from the plan."""
        # From 2014 to 2026 is ~12 years
        train_years = TRAIN_END.year - 2014  # ~6 years
        val_years = VALIDATION_END.year - TRAIN_END.year  # ~3 years
        hb_years = 2026 - HOLD_BACK_START.year  # ~3 years
        assert train_years >= 5, "Training should be at least 5 years"
        assert val_years >= 2, "Validation should be at least 2 years"
        assert hb_years >= 2, "Hold-back should be at least 2 years"