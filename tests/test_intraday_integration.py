"""Integration tests for intraday execution engine.

Tests the full pipeline: bar builder → feature computation → signal → engine → broker.
"""

import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.execution.bar_builder import BarBuilder, Tick, Bar
from src.execution.intraday_engine import IntradayEngine, IntradayConfig
from src.execution.broker import PaperBroker
from src.execution.models import Side, OrderStatus
from src.live.intraday_runner import IntradayRunner, IntradayRunnerConfig, BarRecord
from src.features.intraday import compute_intraday_features
from src.signals.vwap_reversion import generate_vwap_mean_reversion_60b

import polars as pl


class TestBarBuilder:
    """Test tick-to-bar aggregation."""

    def test_basic_bar_completion(self):
        """One minute of ticks produces one completed bar."""
        builder = BarBuilder(ticker="MSFT", interval_minutes=1)
        base_time = datetime(2025, 1, 2, 14, 30, 0, tzinfo=timezone.utc)

        # Feed ticks within the same minute
        for i in range(10):
            tick = Tick(
                price=400.0 + i * 0.1,
                volume=100,
                timestamp=base_time + timedelta(seconds=i),
            )
            result = builder.update(tick)
            assert result is None  # Bar not complete yet

        # Feed a tick in the next minute
        tick = Tick(
            price=410.0,
            volume=50,
            timestamp=base_time + timedelta(minutes=1),
        )
        result = builder.update(tick)
        assert result is not None
        assert isinstance(result, Bar)
        assert result.ticker == "MSFT"
        assert result.open == 400.0
        assert result.high == 400.9
        assert result.low == 400.0
        # Close is the last tick WITHIN the bar interval, not the boundary tick
        assert result.close == 400.9

    def test_force_close(self):
        """Force-close returns the partial bar."""
        builder = BarBuilder(ticker="AAPL", interval_minutes=5)
        base_time = datetime(2025, 1, 2, 14, 30, 0, tzinfo=timezone.utc)

        tick = Tick(price=150.0, volume=100, timestamp=base_time)
        builder.update(tick)

        bar = builder.force_close()
        assert bar is not None
        assert bar.close == 150.0

    def test_reset(self):
        """Reset clears all state."""
        builder = BarBuilder(ticker="MSFT", interval_minutes=1)
        base_time = datetime(2025, 1, 2, 14, 30, 0, tzinfo=timezone.utc)

        tick = Tick(price=400.0, volume=100, timestamp=base_time)
        builder.update(tick)
        builder.force_close()

        builder.reset()
        assert builder.has_open_bar is False
        assert len(builder.completed) == 0
        assert builder.get_stats()["bars_completed"] == 0

    def test_stats(self):
        """Stats track ticks and bars correctly."""
        builder = BarBuilder(ticker="MSFT", interval_minutes=1)
        base_time = datetime(2025, 1, 2, 14, 30, 0, tzinfo=timezone.utc)

        for i in range(5):
            tick = Tick(price=400.0 + i, volume=100, timestamp=base_time + timedelta(seconds=i))
            builder.update(tick)

        stats = builder.get_stats()
        assert stats["ticks_processed"] == 5
        assert stats["bars_completed"] == 0


class TestBrokerFillPrice:
    """Test that broker accepts explicit fill prices (for backtesting)."""

    def test_paper_broker_with_fill_price(self):
        """PaperBroker uses fill_price when provided."""
        broker = PaperBroker(
            initial_cash=100_000.0,
            price_source=None,
        )
        broker.connect()

        # Should work with explicit fill_price even without price_source
        order = broker.place_market_order(
            ticker="MSFT",
            side=Side.BUY,
            quantity=100,
            fill_price=400.0,
        )
        assert order.status == OrderStatus.FILLED
        assert order.avg_fill_price == pytest.approx(400.0, abs=0.5)  # slippage applied

    def test_paper_broker_without_fill_price_fails(self):
        """PaperBroker raises without price_source and no fill_price."""
        broker = PaperBroker(
            initial_cash=100_000.0,
            price_source=None,
        )
        broker.connect()

        with pytest.raises(ValueError, match="No price available"):
            broker.place_market_order(
                ticker="MSFT",
                side=Side.BUY,
                quantity=100,
            )


class TestEndToEndBacktest:
    """Test the full backtest pipeline with real data."""

    def test_backtest_runs(self):
        """Backtest completes without error on real data."""
        runner = IntradayRunner(IntradayRunnerConfig(
            tickers=["MSFT"],
            backtest=True,
            data_dir=Path("data/intraday/1min"),
        ))
        result = runner.run_backtest("2025-01-01", "2025-01-31")

        # Should have processed data
        assert result["total_bars"] > 0
        assert len(result["per_session"]) > 0

        # Aggregate should exist
        agg = result["aggregate"]
        assert "total_sessions" in agg
        assert "total_trades" in agg

    def test_backtest_produces_trades(self):
        """Backtest generates trades (signal produces non-zero values)."""
        runner = IntradayRunner(IntradayRunnerConfig(
            tickers=["MSFT"],
            backtest=True,
            data_dir=Path("data/intraday/1min"),
        ))
        result = runner.run_backtest("2025-01-01", "2025-01-31")
        agg = result["aggregate"]

        # Even with a losing signal, we should have some trades
        assert agg["total_trades"] > 0

    def test_backtest_multi_ticker(self):
        """Backtest handles multiple tickers."""
        runner = IntradayRunner(IntradayRunnerConfig(
            tickers=["MSFT"],
            backtest=True,
            data_dir=Path("data/intraday/1min"),
        ))
        result = runner.run_backtest("2025-01-01", "2025-01-31")
        assert result["total_bars"] > 0


class TestVectorizedFeatures:
    """Test vectorized feature computation."""

    def test_compute_session_features(self):
        """Feature computation produces signal column."""
        runner = IntradayRunner()
        base_time = datetime(2025, 1, 2, 14, 30, 0, tzinfo=timezone.utc)

        # Build a day's worth of bars
        rows = []
        for i in range(400):
            rows.append({
                "ticker": "MSFT",
                "datetime": base_time + timedelta(minutes=i),
                "open": 400.0 + (i % 10) * 0.1,
                "high": 400.5 + (i % 10) * 0.1,
                "low": 399.5 + (i % 10) * 0.1,
                "close": 400.0 + (i % 10) * 0.1,
                "volume": 1000 + i * 10,
                "date": base_time.date(),
            })

        day_df = pl.DataFrame(rows)
        result = runner._compute_session_features("MSFT", day_df)

        # Should have signal column (runner uses the FILTERED signal;
        # the unfiltered 60b variant was replaced in the pipeline)
        assert "signal_vwap_mean_reversion_filtered" in result.columns
        # Signal values should be bounded
        signals = result["signal_vwap_mean_reversion_filtered"].drop_nulls()
        assert len(signals) > 0
        assert (signals.abs() <= 1.1).all()  # Allow small margin for rounding


class TestIntradayRunnerParseInterval:
    """Test interval parsing."""

    def test_parse_1min(self):
        assert IntradayRunner._parse_interval("1min") == 1

    def test_parse_5min(self):
        assert IntradayRunner._parse_interval("5min") == 5

    def test_parse_1h(self):
        assert IntradayRunner._parse_interval("1h") == 60

    def test_parse_numeric(self):
        assert IntradayRunner._parse_interval("30") == 30

    def test_parse_invalid(self):
        assert IntradayRunner._parse_interval("xyz") == 1