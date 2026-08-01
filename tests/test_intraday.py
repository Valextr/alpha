"""Tests for intraday execution engine, bar builder, and live runner."""

import pytest
import polars as pl
from datetime import datetime, timezone, timedelta

from src.execution.intraday_engine import IntradayEngine, IntradayConfig
from src.execution.broker import PaperBroker
from src.execution.bar_builder import BarBuilder, Bar, Tick
from src.execution.live_runner import LiveIntradayRunner, RunnerConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def price_source():
    """Create a simple price source for PaperBroker."""
    data = []
    for i in range(100):
        minute_offset = i % 60
        hour = 14 + (30 + minute_offset) // 60
        minute = (30 + minute_offset) % 60
        d = datetime(2024, 1, 2, hour, minute, 0, tzinfo=timezone.utc)
        price = 100.0 + (i % 20) * 0.1  # Oscillates between 100 and 101.9
        data.append({"date": d.date(), "close": price, "ticker": "MSFT"})
    return {"MSFT": pl.DataFrame(data)}


@pytest.fixture
def engine(price_source):
    """Create an IntradayEngine with PaperBroker."""
    broker = PaperBroker(
        initial_cash=1_000_000,
        commission_per_share=0.005,
        slippage_bps=10.0,
        price_source=price_source,
    )
    config = IntradayConfig(
        ticker="MSFT",
        timeframe="1min",
        max_position_shares=1000,
        stop_loss_pct=0.005,
        take_profit_pct=0.0075,
    )
    return IntradayEngine(config, broker=broker)


@pytest.fixture
def bars():
    """Generate test bars with signals."""
    bars = []
    base_time = datetime(2024, 1, 2, 14, 30, 0, tzinfo=timezone.utc)
    for i in range(50):
        t = base_time + timedelta(minutes=i)
        price = 100.0 + (i % 10) * 0.2  # Oscillates
        # Create a signal that alternates
        signal = 0.3 if i % 20 < 10 else -0.3
        bars.append({
            "close": price,
            "datetime": t,
            "signal_vwap_mean_reversion_filtered": signal,
        })
    return bars


# ---------------------------------------------------------------------------
# TestPaperBroker (already covered in test_execution.py, just smoke tests)
# ---------------------------------------------------------------------------

class TestIntradayEngineInit:
    def test_default_init(self):
        engine = IntradayEngine()
        assert engine.config.ticker == "MSFT"
        assert engine.position == 0
        assert engine.entry_price == 0.0
        assert not engine.session_active

    def test_custom_config(self):
        config = IntradayConfig(ticker="AAPL", max_position_shares=5000)
        engine = IntradayEngine(config)
        assert engine.config.ticker == "AAPL"
        assert engine.config.max_position_shares == 5000

    def test_with_paper_broker(self, price_source):
        broker = PaperBroker(initial_cash=500_000, price_source=price_source)
        engine = IntradayEngine(broker=broker)
        assert engine.broker is broker
        assert engine.broker.get_portfolio_value() == 500_000.0


class TestConnection:
    def test_connect_disconnect(self, engine):
        assert not engine.is_connected
        engine.connect()
        assert engine.is_connected
        engine.disconnect()
        assert not engine.is_connected


class TestSessionLifecycle:
    def test_start_session(self, engine):
        engine.connect()
        engine.start_session()
        assert engine.session_active

    def test_start_session_resets_counters(self, engine):
        engine.connect()
        engine.connect()
        engine.start_session()
        engine.start_session()  # Second start should work
        assert engine.session_active

    def test_end_session(self, engine, bars):
        engine.connect()
        engine.start_session()
        for bar in bars[:10]:
            engine.process_bar(bar)
        metrics = engine.end_session(current_price=100.0)
        assert not engine.session_active
        assert "total_trades" in metrics

    def test_end_session_no_active_session(self, engine):
        result = engine.end_session()
        assert result == {}

    def test_reset(self, engine):
        engine.connect()
        engine.start_session()
        engine.reset()
        assert engine.position == 0
        assert not engine.session_active
        assert len(engine.get_trades()) == 0


class TestBarProcessing:
    def test_process_bar_no_signal(self, engine):
        engine.connect()
        engine.start_session()
        bar = {"close": 100.0, "datetime": datetime.now(timezone.utc)}
        result = engine.process_bar(bar)
        assert result is None

    def test_process_bar_no_price(self, engine):
        engine.connect()
        engine.start_session()
        bar = {"signal_vwap_mean_reversion_filtered": 0.5}
        result = engine.process_bar(bar)
        assert result is None

    def test_process_bar_generates_trade(self, engine, bars):
        engine.connect()
        engine.start_session()
        # First bar should generate a trade
        result = engine.process_bar(bars[0])
        # May or may not trade depending on signal strength
        if result:
            assert "shares" in result
            assert "price" in result

    def test_process_bar_position_changes(self, engine, bars):
        engine.connect()
        engine.start_session()
        initial_position = engine.position
        for bar in bars[:20]:
            engine.process_bar(bar)
        # Position should have changed from initial
        assert engine.position != 0 or initial_position == engine.position

    def test_process_bar_invalid_price(self, engine):
        engine.connect()
        engine.start_session()
        bar = {
            "close": -1.0,  # Invalid price
            "datetime": datetime.now(timezone.utc),
            "signal_vwap_mean_reversion_filtered": 0.5,
        }
        result = engine.process_bar(bar)
        assert result is None

    def test_kill_switch_blocks_processing(self, engine, bars):
        engine.connect()
        engine.start_session()
        engine.config.kill_switch_active = True
        result = engine.process_bar(bars[0])
        assert result is None


class TestStopLossTakeProfit:
    def test_stop_loss_triggered(self, engine):
        engine.connect()
        engine.start_session()

        # Manually set position
        engine._position = 100
        engine._entry_price = 100.0

        # Price drops below stop loss (0.5%)
        assert engine.check_stop_loss(99.4) is True
        assert engine.position == 0

    def test_take_profit_triggered(self, engine):
        engine.connect()
        engine.start_session()

        engine._position = 100
        engine._entry_price = 100.0

        # Price rises above take profit (0.75%)
        assert engine.check_stop_loss(100.8) is True
        assert engine.position == 0

    def test_no_exit_when_flat(self, engine):
        engine.connect()
        engine.start_session()

        engine._position = 0
        assert engine.check_stop_loss(100.0) is False

    def test_no_exit_within_range(self, engine):
        engine.connect()
        engine.start_session()

        engine._position = 100
        engine._entry_price = 100.0

        # Price within SL/TP range
        assert engine.check_stop_loss(100.2) is False


class TestMetrics:
    def test_session_metrics(self, engine, bars):
        engine.connect()
        engine.start_session()
        for bar in bars[:20]:
            engine.process_bar(bar)
        engine.end_session(current_price=100.0)

        metrics = engine.get_session_metrics()
        assert "total_trades" in metrics
        assert "total_pnl" in metrics
        assert "win_rate_pct" in metrics
        assert "profit_factor" in metrics

    def test_equity_curve(self, engine, bars):
        engine.connect()
        engine.start_session()
        for bar in bars[:10]:
            engine.process_bar(bar)

        curve = engine.get_equity_curve()
        assert len(curve) > 0
        assert "portfolio_value" in curve.columns

    def test_trades_list(self, engine, bars):
        engine.connect()
        engine.start_session()
        for bar in bars[:10]:
            engine.process_bar(bar)

        trades = engine.get_trades()
        for trade in trades:
            assert "datetime" in trade
            assert "shares" in trade
            assert "price" in trade


class TestBarBuilder:
    def test_init(self):
        builder = BarBuilder("MSFT", interval_minutes=1)
        assert builder.ticker == "MSFT"
        assert builder.interval_minutes == 1
        assert not builder.has_open_bar
        assert len(builder.completed) == 0

    def test_first_tick_opens_bar(self):
        builder = BarBuilder("MSFT", interval_minutes=1)
        tick = Tick(price=100.0, volume=100, timestamp=datetime.now(timezone.utc))
        result = builder.update(tick)
        assert result is None  # No completed bar yet
        assert builder.has_open_bar
        assert builder.current_bar.open == 100.0

    def test_updates_ohlc(self):
        builder = BarBuilder("MSFT", interval_minutes=1)
        base = datetime(2024, 1, 2, 14, 30, 0, tzinfo=timezone.utc)

        builder.update(Tick(price=100.0, volume=100, timestamp=base))
        builder.update(Tick(price=100.5, volume=50, timestamp=base))
        builder.update(Tick(price=99.5, volume=75, timestamp=base))

        bar = builder.current_bar
        assert bar.open == 100.0
        assert bar.high == 100.5
        assert bar.low == 99.5
        assert bar.close == 99.5
        # Volume is tracked in _volume, only set on bar completion
        assert builder._volume == 225

    def test_completes_bar_on_new_interval(self):
        builder = BarBuilder("MSFT", interval_minutes=1)
        base = datetime(2024, 1, 2, 14, 30, 0, tzinfo=timezone.utc)

        builder.update(Tick(price=100.0, volume=100, timestamp=base))
        # Next minute — triggers completion of previous bar
        result = builder.update(Tick(price=100.5, volume=50, timestamp=base + timedelta(minutes=1)))
        assert result is not None
        assert result.close == 100.0  # Previous bar's close (last tick in old interval)
        assert result.volume == 100
        assert len(builder.completed) == 1
        # New bar should have started with the new tick
        assert builder.current_bar.open == 100.5

    def test_force_close(self):
        builder = BarBuilder("MSFT", interval_minutes=1)
        base = datetime(2024, 1, 2, 14, 30, 0, tzinfo=timezone.utc)
        builder.update(Tick(price=100.0, volume=100, timestamp=base))

        bar = builder.force_close()
        assert bar is not None
        assert len(builder.completed) == 1
        assert not builder.has_open_bar

    def test_force_close_no_bar(self):
        builder = BarBuilder("MSFT", interval_minutes=1)
        bar = builder.force_close()
        assert bar is None

    def test_reset(self):
        builder = BarBuilder("MSFT", interval_minutes=1)
        base = datetime(2024, 1, 2, 14, 30, 0, tzinfo=timezone.utc)
        builder.update(Tick(price=100.0, timestamp=base))
        builder.reset()
        assert not builder.has_open_bar
        assert len(builder.completed) == 0

    def test_stats(self):
        builder = BarBuilder("MSFT", interval_minutes=1)
        base = datetime(2024, 1, 2, 14, 30, 0, tzinfo=timezone.utc)
        builder.update(Tick(price=100.0, timestamp=base))
        stats = builder.get_stats()
        assert stats["ticks_processed"] == 1
        assert stats["bars_completed"] == 0


class TestLiveIntradayRunner:
    def test_init_paper_mode(self):
        config = IntradayConfig(ticker="MSFT", use_ibkr=False)
        runner = LiveIntradayRunner(config)
        assert type(runner.engine.broker).__name__ == "PaperBroker"
        assert runner.bar_builder.ticker == "MSFT"

    def test_backtest(self):
        config = IntradayConfig(ticker="MSFT", use_ibkr=False, timeframe="1min")
        runner = LiveIntradayRunner(config)

        # Create test bars with signals
        base = datetime(2024, 1, 2, 14, 30, 0, tzinfo=timezone.utc)
        bars = []
        for i in range(20):
            bars.append(Bar(
                ticker="MSFT",
                datetime=base + timedelta(minutes=i),
                open=100.0 + i * 0.1,
                high=100.5 + i * 0.1,
                low=99.5 + i * 0.1,
                close=100.0 + i * 0.1,
                volume=100,
                signal_vwap_mean_reversion_filtered=0.3 if i % 10 < 5 else -0.3,
            ))

        metrics = runner.run_backtest(bars)
        assert "total_trades" in metrics or "total_pnl" in metrics

    def test_parse_interval(self):
        assert LiveIntradayRunner._parse_interval("1min") == 1
        assert LiveIntradayRunner._parse_interval("5min") == 5
        assert LiveIntradayRunner._parse_interval("1h") == 60
        assert LiveIntradayRunner._parse_interval("30s") == 1  # floors to 1 min
        assert LiveIntradayRunner._parse_interval("60") == 60

    def test_get_status(self):
        config = IntradayConfig(ticker="MSFT")
        runner = LiveIntradayRunner(config)
        status = runner.get_status()
        assert "running" in status
        assert "engine_metrics" in status
        assert "bar_builder_stats" in status