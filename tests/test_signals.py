"""Signal factory tests — registry, generation, validation, and IC analysis."""

import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.signals.registry import registry as signal_registry
from src.signals.base import (
    compute_forward_returns,
    rank_ic,
    ic_decay,
    win_rate,
    signal_summary,
)
from src.signals.pipeline import generate_all, generate_all_with_forward_returns
from src.signals import mean_reversion, momentum  # noqa: F401 — registers signals
from src.features.pipeline import compute_features


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
    # Add some randomness via a simple pattern: alternating returns
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


# ── Registry tests ──────────────────────────────────────────────────


class TestSignalRegistry:
    def setup_method(self):
        # Save state so we don't pollute other tests
        self._sigs = {k: v for k, v in signal_registry._signals.items()}

    def teardown_method(self):
        signal_registry._signals.clear()
        signal_registry._signals.update(self._sigs)

    def test_signals_registered(self):
        # After importing signal modules, signals should be in registry
        assert "mean_reversion_21d" in signal_registry
        assert "mean_reversion_63d" in signal_registry
        assert "momentum_21d" in signal_registry
        assert "momentum_63d" in signal_registry

    def test_list_signals(self):
        signals = signal_registry.list_signals()
        assert len(signals) >= 4

    def test_signals_by_category(self):
        cats = signal_registry.signals_by_category()
        assert "mean_reversion" in cats
        assert "momentum" in cats

    def test_generate_functions_exist(self):
        for name in signal_registry._signals:
            fn = signal_registry.get_generate_fn(name)
            assert fn is not None, f"generate_fn for {name} is None"

    def test_dependencies_valid(self):
        missing = signal_registry.validate_dependencies()
        assert not missing, f"Missing signal dependencies: {missing}"


# ── Signal generation tests ─────────────────────────────────────────


class TestMeanReversion:
    def test_signal_column_produced(self):
        df = _multi_ticker_bars(5, 200)
        df = compute_features(df)
        out = mean_reversion.generate_mean_reversion_21d(df)
        assert "signal_mean_reversion_21d" in out.columns

    def test_signal_bounds(self):
        df = _multi_ticker_bars(5, 200)
        df = compute_features(df)
        out = mean_reversion.generate_mean_reversion_21d(df)
        col = out["signal_mean_reversion_21d"]
        non_null = col.drop_nulls()
        assert (non_null <= 1.0).all(), "Signal should be bounded above by 1"
        assert (non_null >= -1.0).all(), "Signal should be bounded below by -1"

    def test_signal_has_negative_values(self):
        df = _multi_ticker_bars(5, 200)
        df = compute_features(df)
        out = mean_reversion.generate_mean_reversion_21d(df)
        col = out["signal_mean_reversion_21d"].drop_nulls()
        assert (col > 0).any(), "Signal should have positive values"
        assert (col < 0).any(), "Signal should have negative values"

    def test_longer_lookback(self):
        df = _multi_ticker_bars(5, 200)
        df = compute_features(df)
        out = mean_reversion.generate_mean_reversion_63d(df)
        assert "signal_mean_reversion_63d" in out.columns


class TestMomentum:
    def test_signal_column_produced(self):
        df = _multi_ticker_bars(5, 200)
        df = compute_features(df)
        out = momentum.generate_momentum_21d(df)
        assert "signal_momentum_21d" in out.columns

    def test_signal_bounds(self):
        df = _multi_ticker_bars(5, 200)
        df = compute_features(df)
        out = momentum.generate_momentum_21d(df)
        col = out["signal_momentum_21d"].drop_nulls()
        assert (col <= 1.0).all()
        assert (col >= -1.0).all()

    def test_signal_has_both_signs(self):
        df = _multi_ticker_bars(5, 200)
        df = compute_features(df)
        out = momentum.generate_momentum_21d(df)
        col = out["signal_momentum_21d"].drop_nulls()
        assert (col > 0).any()
        assert (col < 0).any()

    def test_longer_formation(self):
        df = _multi_ticker_bars(5, 200)
        df = compute_features(df)
        out = momentum.generate_momentum_63d(df)
        assert "signal_momentum_63d" in out.columns


# ── Pipeline tests ──────────────────────────────────────────────────


class TestPipeline:
    def test_generate_all(self):
        df = _multi_ticker_bars(5, 200)
        df = compute_features(df)
        out = generate_all(df)
        # Should have all 4 signal columns
        assert "signal_mean_reversion_21d" in out.columns
        assert "signal_mean_reversion_63d" in out.columns
        assert "signal_momentum_21d" in out.columns
        assert "signal_momentum_63d" in out.columns

    def test_generate_all_with_forward_returns(self):
        df = _multi_ticker_bars(5, 200)
        df = compute_features(df)
        out = generate_all_with_forward_returns(df, horizons=[1, 5])
        assert "signal_mean_reversion_21d" in out.columns
        assert "forward_return_1" in out.columns
        assert "forward_return_5" in out.columns

    def test_generate_by_category(self):
        df = _multi_ticker_bars(5, 200)
        df = compute_features(df)
        out = generate_all(df, categories=["mean_reversion"])
        assert "signal_mean_reversion_21d" in out.columns
        assert "signal_momentum_21d" not in out.columns

    def test_empty_df(self):
        out = generate_all(pl.DataFrame())
        assert out.is_empty()


# ── Validation tests ────────────────────────────────────────────────


class TestForwardReturns:
    def test_forward_returns_columns(self):
        df = _multi_ticker_bars(5, 100)
        out = compute_forward_returns(df, horizons=[1, 5, 21])
        assert "forward_return_1" in out.columns
        assert "forward_return_5" in out.columns
        assert "forward_return_21" in out.columns

    def test_forward_returns_per_ticker(self):
        """Forward returns should be computed within each ticker group."""
        df = _multi_ticker_bars(2, 50)
        out = compute_forward_returns(df, horizons=[1])
        # Check that the forward return for the first row of AAPL
        # matches (next_close / current_close) - 1
        aapl = out.filter(pl.col("ticker") == "AAPL").sort("date")
        if len(aapl) >= 2:
            first = aapl.row(0)
            second = aapl.row(1)
            # row() returns a tuple; columns are in insertion order
            close_idx = out.columns.index("close")
            fwd_idx = out.columns.index("forward_return_1")
            expected = (second[close_idx] / first[close_idx]) - 1
            actual = first[fwd_idx]
            if actual is not None:
                assert abs(actual - expected) < 1e-6


class TestICValidation:
    def test_rank_ic_returns_number(self):
        df = _multi_ticker_bars(5, 200)
        df = compute_features(df)
        df = generate_all_with_forward_returns(df)
        ic = rank_ic("signal_momentum_21d", "forward_return_1", df)
        assert -1.0 <= ic <= 1.0

    def test_ic_decay(self):
        df = _multi_ticker_bars(5, 200)
        df = compute_features(df)
        df = generate_all_with_forward_returns(df, horizons=[1, 5, 21])
        decay = ic_decay("signal_momentum_21d", df, horizons=[1, 5, 21])
        assert 1 in decay
        assert 5 in decay
        assert 21 in decay

    def test_win_rate_returns_number(self):
        df = _multi_ticker_bars(5, 200)
        df = compute_features(df)
        df = generate_all_with_forward_returns(df)
        wr = win_rate("signal_momentum_21d", "forward_return_1", df)
        assert 0.0 <= wr <= 1.0

    def test_signal_summary(self):
        df = _multi_ticker_bars(5, 200)
        df = compute_features(df)
        df = generate_all_with_forward_returns(df, horizons=[1, 5, 21])
        summary = signal_summary("signal_momentum_21d", df, horizons=[1, 5, 21])
        assert "count" in summary
        assert "mean" in summary
        assert "ic_1d" in summary
        assert "win_rate_1d" in summary


# ── Integration: real data ──────────────────────────────────────────


class TestRealData:
    def test_signals_on_gold_data(self):
        """Run signals on actual gold layer data if available."""
        gold_dir = root / "data" / "gold" / "daily" / "year=2023"
        files = list(gold_dir.glob("ticker=*/part-0.parquet"))
        if not files:
            pytest.skip("No gold data available")

        frames = [pl.read_parquet(str(f)) for f in sorted(files)]
        df = pl.concat(frames, how="vertical_relaxed").sort(["ticker", "date"])
        assert not df.is_empty()

        # Compute features first
        df = compute_features(df)

        # Generate signals
        out = generate_all_with_forward_returns(df, horizons=[1, 5])

        # All signal columns should be present
        assert "signal_mean_reversion_21d" in out.columns
        assert "signal_momentum_21d" in out.columns
        assert "forward_return_1" in out.columns

        # Signals should have meaningful spread (not all zero)
        for col in ["signal_mean_reversion_21d", "signal_momentum_21d"]:
            std = out[col].std()
            assert std is not None and std > 0, f"{col} has no variance"

    def test_ic_on_real_data(self):
        """IC should be computable on real data."""
        gold_dir = root / "data" / "gold" / "daily" / "year=2023"
        files = list(gold_dir.glob("ticker=*/part-0.parquet"))
        if not files:
            pytest.skip("No gold data available")

        frames = [pl.read_parquet(str(f)) for f in sorted(files)]
        df = pl.concat(frames, how="vertical_relaxed").sort(["ticker", "date"])
        df = compute_features(df)
        df = generate_all_with_forward_returns(df, horizons=[1, 5, 21])

        for sig in ["signal_mean_reversion_21d", "signal_momentum_21d"]:
            ic = rank_ic(sig, "forward_return_1", df)
            assert -1.0 <= ic <= 1.0, f"{sig} IC out of range: {ic}"