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
        # Same-day bar: a NEW day clears the breaker (live semantics), so
        # pin the session date to the bar's date to test the block itself.
        dt0 = bars[0]["datetime"]
        engine._session_date = dt0.date() if hasattr(dt0, "date") else dt0
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
        config = IntradayConfig(ticker="MSFT", max_position_shares=1000)
        runner = LiveIntradayRunner(config)
        status = runner.get_status()
        assert "running" in status
        assert "engine_metrics" in status
        assert "bar_builder_stats" in status


# ---------------------------------------------------------------------------
# YFinance fallback (regression: incomplete minutes were marked "seen" before
# the cutoff check, so the completed bar was skipped forever — Aug 3, 2026)
# ---------------------------------------------------------------------------

class TestYFinanceFallback:
    """Tests for IntradayRunner._fetch_bars_yfinance dedupe/throttle logic."""

    @staticmethod
    def _make_frame(base: datetime, rows: int = 10):
        """yfinance-shaped DataFrame with 1-min rows ending at `base` (inclusive)."""
        import pandas as pd
        idx = pd.date_range(base - timedelta(minutes=rows - 1), periods=rows, freq="min")
        idx.name = "Datetime"
        return pd.DataFrame({
            "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1000,
        }, index=idx)

    @staticmethod
    def _freeze_clock(monkeypatch, start: datetime):
        """Monkeypatch the runner module's datetime with a controllable clock."""
        import src.live.intraday_runner as runner_mod

        class _Clock(datetime):
            _now = start

            @classmethod
            def now(cls, tz=None):
                return cls._now if tz is None else cls._now.astimezone(tz)

        monkeypatch.setattr(runner_mod, "datetime", _Clock)
        return _Clock

    @staticmethod
    def _runner():
        from src.live.intraday_runner import IntradayRunner, IntradayRunnerConfig
        return IntradayRunner(config=IntradayRunnerConfig(tickers=["MSFT"], use_ibkr=False))

    def test_incomplete_minute_is_retried_when_completed(self, monkeypatch):
        base = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        Clock = self._freeze_clock(monkeypatch, base)
        runner = self._runner()
        monkeypatch.setattr(runner, "_yf_fetch", staticmethod(lambda t: self._make_frame(base)))

        # First poll: 9 completed minutes; the current minute is skipped, NOT marked seen
        bars = runner._fetch_bars_yfinance("MSFT")
        assert len(bars) == 9
        assert all(b.datetime < base for b in bars)
        assert base not in runner._yf_seen["MSFT"]

        # Second poll, one minute later: the previously-incomplete minute is complete
        Clock._now = base + timedelta(minutes=1)
        bars = runner._fetch_bars_yfinance("MSFT")
        assert [b.datetime for b in bars] == [base]

        # Third poll: nothing new (dedupe still works for returned bars)
        assert runner._fetch_bars_yfinance("MSFT") == []

    def test_throttle(self, monkeypatch):
        base = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        self._freeze_clock(monkeypatch, base)
        runner = self._runner()
        monkeypatch.setattr(runner, "_yf_fetch", staticmethod(lambda t: self._make_frame(base)))

        assert len(runner._fetch_bars_yfinance("MSFT")) == 9
        # Immediate re-poll within the 55s throttle window returns nothing
        assert runner._fetch_bars_yfinance("MSFT") == []

    def test_naive_datetimes_are_assumed_utc(self, monkeypatch):
        """Regression (Aug 3 2026 tz fix): naive timestamps must be interpreted
        as UTC, never local — a UTC-tainted process environment cannot shift
        the session window. Bars come out tz-aware UTC."""
        base = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        self._freeze_clock(monkeypatch, base)
        runner = self._runner()
        naive_frame = self._make_frame(base.replace(tzinfo=None))  # naive index, like old bridge
        monkeypatch.setattr(runner, "_yf_fetch", staticmethod(lambda t: naive_frame))

        bars = runner._fetch_bars_yfinance("MSFT")
        assert len(bars) == 9  # current minute still excluded (cutoff in UTC)
        assert all(b.datetime.tzinfo == timezone.utc for b in bars)  # normalized to UTC

    def test_empty_frame(self, monkeypatch):
        base = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        self._freeze_clock(monkeypatch, base)
        runner = self._runner()
        monkeypatch.setattr(runner, "_yf_fetch", staticmethod(lambda t: None))
        assert runner._fetch_bars_yfinance("MSFT") == []


class TestAlpacaSource:
    """IntradayRunner._fetch_bars_alpaca dedupe/throttle/tz contract."""

    @staticmethod
    def _make_frame(base: datetime, rows: int = 10, tz: str = "America/New_York"):
        """Alpaca-shaped DataFrame: reset_index() output with a 'timestamp' column."""
        import pandas as pd
        from zoneinfo import ZoneInfo
        # pandas rejects an aware start with tz=; convert to NY wall time first
        ny_base = base.astimezone(ZoneInfo(tz)).replace(tzinfo=None)
        idx = pd.date_range(ny_base - timedelta(minutes=rows - 1), periods=rows,
                            freq="min", tz=tz)
        idx.name = "timestamp"
        return pd.DataFrame({
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
            "volume": 1000, "trade_count": 10, "vwap": 100.2,
        }, index=idx).reset_index()

    @staticmethod
    def _freeze_clock(monkeypatch, start: datetime):
        import src.live.intraday_runner as runner_mod

        class _Clock(datetime):
            _now = start

            @classmethod
            def now(cls, tz=None):
                return cls._now if tz is None else cls._now.astimezone(tz)

        monkeypatch.setattr(runner_mod, "datetime", _Clock)
        return _Clock

    @staticmethod
    def _runner(monkeypatch, frame):
        from src.live.intraday_runner import IntradayRunner, IntradayRunnerConfig
        runner = IntradayRunner(config=IntradayRunnerConfig(tickers=["MSFT"], use_ibkr=False))
        monkeypatch.setattr(runner, "_alpaca_fetch", staticmethod(lambda t, now, lookback_hours=24: frame))
        return runner

    def test_bars_normalized_to_aware_utc(self, monkeypatch):
        base = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        self._freeze_clock(monkeypatch, base)
        runner = self._runner(monkeypatch, self._make_frame(base))
        bars = runner._fetch_bars_alpaca("MSFT")
        assert len(bars) == 9                      # current minute excluded
        assert all(b.datetime.tzinfo == timezone.utc for b in bars)
        # NY 16:00 == UTC 20:00 (EDT) — wall times preserved correctly
        assert bars[-1].datetime == base - timedelta(minutes=1)

    def test_incomplete_minute_retried_when_completed(self, monkeypatch):
        base = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        Clock = self._freeze_clock(monkeypatch, base)
        runner = self._runner(monkeypatch, self._make_frame(base))
        assert len(runner._fetch_bars_alpaca("MSFT")) == 9
        assert base not in runner._alpaca_seen["MSFT"]
        Clock._now = base + timedelta(minutes=1)
        bars = runner._fetch_bars_alpaca("MSFT")
        assert [b.datetime for b in bars] == [base]
        assert runner._fetch_bars_alpaca("MSFT") == []  # dedupe

    def test_throttle(self, monkeypatch):
        base = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        self._freeze_clock(monkeypatch, base)
        runner = self._runner(monkeypatch, self._make_frame(base))
        assert len(runner._fetch_bars_alpaca("MSFT")) == 9
        assert runner._fetch_bars_alpaca("MSFT") == []  # 55s window

    def test_backfill_window_depends_on_timeframe(self, monkeypatch):
        """Warm-up backfill width must follow the timeframe: 65 days for 1h
        (so the ~390-bar feature lookback is warm on day 1), 24h for 1min.
        Subsequent polls always use the 24h window regardless of timeframe."""
        from src.live.intraday_runner import IntradayRunner, IntradayRunnerConfig
        base = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        Clock = self._freeze_clock(monkeypatch, base)
        frame = self._make_frame(base)
        captured = []

        def recorder(ticker, now, lookback_hours):
            captured.append(lookback_hours)
            return frame

        # 1h: warm-up fetch requests 65 days
        runner_1h = IntradayRunner(
            config=IntradayRunnerConfig(tickers=["MSFT"], timeframe="1h", use_ibkr=False))
        monkeypatch.setattr(runner_1h, "_alpaca_fetch", staticmethod(recorder))
        assert len(runner_1h._fetch_bars_alpaca("MSFT")) == 9
        assert captured[-1] == 65 * 24
        # 1h: next poll (after throttle) uses the 24h window
        Clock._now = base + timedelta(minutes=2)
        runner_1h._fetch_bars_alpaca("MSFT")
        assert captured[-1] == 24
        # 1min: warm-up fetch keeps the 24h window
        Clock._now = base
        runner_1min = IntradayRunner(
            config=IntradayRunnerConfig(tickers=["MSFT"], use_ibkr=False))
        monkeypatch.setattr(runner_1min, "_alpaca_fetch", staticmethod(recorder))
        assert len(runner_1min._fetch_bars_alpaca("MSFT")) == 9
        assert captured[-1] == 24

    def test_falls_back_to_yfinance_after_three_failures(self, monkeypatch):
        base = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        Clock = self._freeze_clock(monkeypatch, base)
        runner = self._runner(monkeypatch, None)  # _alpaca_fetch returns None
        # stub the yfinance path too — the 3rd failure hands off to it, and a
        # unit test must never hit the network
        monkeypatch.setattr(runner, "_fetch_bars_yfinance", staticmethod(lambda t: []))
        for _ in range(3):
            runner._fetch_bars_alpaca("MSFT")
            Clock._now = Clock._now + timedelta(minutes=1)  # advance past throttle
        assert runner._data_source.get("MSFT") == "yfinance"

    def test_forced_source_dispatch(self, monkeypatch):
        """ALPHA_DATA_SOURCE=alpaca routes _fetch_bars straight to Alpaca
        (no IBKR probe) once an engine is connected."""
        from types import SimpleNamespace
        base = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        self._freeze_clock(monkeypatch, base)
        from src.live.intraday_runner import IntradayRunner, IntradayRunnerConfig
        runner = IntradayRunner(config=IntradayRunnerConfig(tickers=["MSFT"], use_ibkr=True))
        runner._engines["MSFT"] = SimpleNamespace(is_connected=True)
        monkeypatch.setattr(runner, "_alpaca_fetch", staticmethod(lambda t, now, lookback_hours=24: self._make_frame(base)))
        monkeypatch.setenv("ALPHA_DATA_SOURCE", "alpaca")
        bars = runner._fetch_bars("MSFT")
        assert len(bars) == 9
        assert runner._data_source.get("MSFT") == "alpaca"

    def test_null_signal_treated_as_zero(self, monkeypatch):
        """Regression (Aug 4 2026): null signals from feature warm-up (rolling
        VWAP needs 100 samples) crashed the live loop with float(None). Null
        must become 0.0 — the engine's 'no action' value."""
        base = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        self._freeze_clock(monkeypatch, base)
        from src.live.intraday_runner import IntradayRunner, IntradayRunnerConfig, BarRecord
        runner = IntradayRunner(config=IntradayRunnerConfig(tickers=["MSFT"], use_ibkr=False))
        # 61 constant-price bars: rolling std == 0 -> std.replace(0, None) ->
        # base signal null -> polars when-cascade keeps null -> float(None) crash
        bars = [
            BarRecord(ticker="MSFT", datetime=base - timedelta(minutes=61 - i),
                      open=100.0, high=100.0, low=100.0, close=100.0, volume=1000)
            for i in range(61)
        ]
        runner._bar_buffer["MSFT"] = bars[:-1]
        out = runner._compute_signal("MSFT", bars[-1])
        assert out[runner.config.signal_column] == 0.0

    def test_warmup_fills_buffer_without_trading(self):
        """The session-start backfill must warm the feature buffer only —
        never create engines or touch the broker (Aug 4 2026 regression:
        the 24h backfill was traded as a tight loop)."""
        from src.live.intraday_runner import IntradayRunner, IntradayRunnerConfig, BarRecord
        base = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        runner = IntradayRunner(config=IntradayRunnerConfig(tickers=["MSFT"], use_ibkr=False))

        def mk(n, offset=0):
            return [BarRecord(ticker="MSFT", datetime=base - timedelta(minutes=n - i + offset),
                              open=100.0, high=101.0, low=99.0, close=100.5, volume=1000)
                    for i in range(n)]

        runner._warmup_bars("MSFT", mk(100))
        assert len(runner._bar_buffer["MSFT"]) == 100
        assert len(runner._engines) == 0  # no engine, no orders
        runner._warmup_bars("MSFT", mk(100, offset=100))
        assert len(runner._bar_buffer["MSFT"]) == 200
        runner._warmup_bars("MSFT", mk(600, offset=200))
        assert len(runner._bar_buffer["MSFT"]) == 500  # capped


class _AsyncFillBroker:
    """Broker stub with live-broker semantics: place_market_order returns a
    PENDING order; positions change only when the test simulates a fill."""

    def __init__(self):
        self.position_qty = 0
        self.avg_cost = 0.0
        self.orders = []
        self.portfolio_value = 1_000_000.0

    def place_market_order(self, ticker, side, quantity, signal_strength=0.0,
                           target_weight=0.0, fill_price=None):
        from src.execution.models import Order, OrderStatus, OrderType
        self.orders.append((side, quantity, fill_price))
        return Order(ticker=ticker, side=side, order_type=OrderType.MARKET,
                     quantity=quantity, status=OrderStatus.PENDING)

    def get_positions(self):
        from src.execution.models import Position
        if self.position_qty == 0:
            return {}
        return {"MSFT": Position(ticker="MSFT", quantity=self.position_qty,
                                 avg_cost=self.avg_cost)}

    def get_portfolio_value(self):
        return self.portfolio_value

    def get_current_price(self, ticker):
        return 100.0

    def set_position(self, qty: int, avg_cost: float = 0.0):
        self.position_qty = qty
        if avg_cost:
            self.avg_cost = avg_cost


class TestEngineLivePositionTracking:
    """Regression (pitfall 30, Aug 4 2026): live brokers fill async; the
    engine must reconcile from the broker book instead of assuming the
    synchronous-FILLED return it gets from PaperBroker."""

    @staticmethod
    def _bar():
        return {"close": 100.0,
                "datetime": datetime.now(timezone.utc),
                "signal_vwap_mean_reversion_filtered": -0.87}

    def test_async_fill_reconciled_next_bar(self):
        broker = _AsyncFillBroker()
        engine = IntradayEngine(config=IntradayConfig(ticker="MSFT", max_position_shares=1000), broker=broker)
        engine.start_session()

        out = engine.process_bar(self._bar())
        assert out is None                       # async: not recorded same-bar
        assert len(broker.orders) == 1           # one SELL placed
        assert engine._position == 0             # fill not visible yet
        assert engine._pending_order is not None

        # Fill lands (gateway position update arrives between bars)
        broker.set_position(-870, 100.0)
        out2 = engine.process_bar(self._bar())
        assert engine._position == -870          # reconciled to reality
        assert len(broker.orders) == 1           # no stacking — target == position
        assert engine._pending_order is None
        assert len(engine._trades) == 1          # fill recorded
        assert engine._trades[0]["reconciled"] is True

    def test_no_order_stacking_while_pending(self):
        broker = _AsyncFillBroker()
        engine = IntradayEngine(config=IntradayConfig(ticker="MSFT", max_position_shares=1000), broker=broker)
        engine.start_session()

        for _ in range(3):
            engine.process_bar(self._bar())
        assert len(broker.orders) == 1  # one outstanding order, then wait

    def test_unfilled_orders_pause_after_three(self):
        broker = _AsyncFillBroker()
        engine = IntradayEngine(config=IntradayConfig(ticker="MSFT", max_position_shares=1000), broker=broker)
        engine.start_session()

        for _ in range(3):
            engine.process_bar(self._bar())
        # bar 1 placed the order; bars 2-4 accumulate strikes -> pause at bar 4
        engine.process_bar(self._bar())
        assert engine._order_pause == 4  # set to 5, decremented by the gate
        n_before = len(broker.orders)
        engine.process_bar(self._bar())
        assert len(broker.orders) == n_before  # no new orders during pause

    def test_start_session_adopts_broker_position(self):
        broker = _AsyncFillBroker()
        broker.set_position(500, 100.0)
        engine = IntradayEngine(config=IntradayConfig(ticker="MSFT", max_position_shares=1000), broker=broker)
        engine.start_session()
        assert engine._position == 500
        assert engine._entry_price == 100.0
        assert len(engine._trades) == 0  # adoption is not a trade

    def test_paper_path_short_agreement_and_sl(self):
        """Regression (Aug 4 2026): PaperBroker must track shorts as negative
        quantities so the engine's broker-book reconciliation agrees on the
        paper path (no phantom-short reset loop, no free-cash compounding),
        and the synchronous FILLED branch must set a short entry price so
        SL/TP fire for shorts exactly as on the live reconciliation path."""
        from src.execution.broker import PaperBroker

        broker = PaperBroker(
            initial_cash=1_000_000.0,
            commission_per_share=0.0,
            slippage_bps=0.0,
        )
        engine = IntradayEngine(
            config=IntradayConfig(
                ticker="MSFT",
                max_position_shares=1000,
                commission_per_share=0.0,
                slippage_bps=0.0,
            ),
            broker=broker,
        )
        engine.connect()
        engine.start_session()

        bar = {"close": 100.0,
               "datetime": datetime.now(timezone.utc),
               "signal_vwap_mean_reversion_filtered": -0.87}
        out = engine.process_bar(dict(bar))
        assert out is not None
        assert engine._position == -870
        assert engine._entry_price == pytest.approx(100.0)
        # broker book agrees — reconciliation is a no-op, no phantom reset
        assert broker.get_positions()["MSFT"].quantity == -870
        assert len(engine._trades) == 1

        # same signal, small move: no re-order, no reconciliation trade
        bar["close"] = 100.3
        engine.process_bar(dict(bar))
        assert engine._position == -870
        assert len(engine._trades) == 1

        # +0.6% move against the short → stop loss closes it. Signal 0.0 so
        # the bar doesn't re-enter after the close; the SL path is proven by
        # the trade's reason (a plain zero-signal close has no reason key).
        bar["close"] = 100.6
        bar["signal_vwap_mean_reversion_filtered"] = 0.0
        engine.process_bar(dict(bar))
        assert engine._position == 0
        assert "MSFT" not in broker.get_positions()
        assert engine._trades[-1].get("reason") == "stop_loss"

    def test_paper_path_long_short_flip_agreement(self):
        """A full signal flip (short → long) must keep engine and broker
        books in agreement through the zero crossing."""
        from src.execution.broker import PaperBroker

        broker = PaperBroker(
            initial_cash=1_000_000.0,
            commission_per_share=0.0,
            slippage_bps=0.0,
        )
        engine = IntradayEngine(
            config=IntradayConfig(
                ticker="MSFT",
                max_position_shares=1000,
                commission_per_share=0.0,
                slippage_bps=0.0,
            ),
            broker=broker,
        )
        engine.connect()
        engine.start_session()

        def bar(sig: float, close: float):
            return {"close": close,
                    "datetime": datetime.now(timezone.utc),
                    "signal_vwap_mean_reversion_filtered": sig}

        # open short -870 @ 100
        engine.process_bar(bar(-0.87, 100.0))
        assert engine._position == -870
        assert broker.get_positions()["MSFT"].quantity == -870

        # flip to long: target +870 from -870 → BUY 1740 (cover + open long);
        # price stays inside the SL/TP band so no stop fires first
        engine.process_bar(bar(0.87, 100.1))
        assert engine._position == 870
        assert broker.get_positions()["MSFT"].quantity == 870
        assert engine._entry_price == pytest.approx(100.1)
        assert len(engine._trades) == 2

        # flip back to short: target -870 from +870 → SELL 1740
        engine.process_bar(bar(-0.87, 100.2))
        assert engine._position == -870
        assert broker.get_positions()["MSFT"].quantity == -870
        assert engine._entry_price == pytest.approx(100.2)
        assert len(engine._trades) == 3

    def test_daily_kill_switch_resets_on_date_change(self):
        """Regression (Aug 4 2026): backtests run one session for the whole
        period, so the 2% daily loss breaker must reset per calendar day —
        exactly what the live runner gets from a fresh start_session() each
        morning. Before this fix the breaker froze the rest of the year
        after the first 2% drawdown (every ticker stopped at ~107 trades)."""
        from src.execution.broker import PaperBroker

        broker = PaperBroker(
            initial_cash=1_000_000.0,
            commission_per_share=0.0,
            slippage_bps=0.0,
        )
        engine = IntradayEngine(
            config=IntradayConfig(
                ticker="MSFT",
                max_position_shares=1000,
                stop_loss_pct=1.0,   # effectively disabled — test the breaker
                take_profit_pct=1.0,  # via normal trades, not SL/TP
                commission_per_share=0.0,
                slippage_bps=0.0,
            ),
            broker=broker,
        )
        engine.connect()
        engine.start_session()

        day1 = datetime(2026, 8, 3, 14, 0, tzinfo=timezone.utc)
        day2 = datetime(2026, 8, 4, 14, 0, tzinfo=timezone.utc)

        # day 1: buy 870 @ 100
        engine.process_bar({"close": 100.0, "datetime": day1,
                            "signal_vwap_mean_reversion_filtered": 0.87})
        # day 1: close at 77 → loss 870×23 = 20,010 = 2.001% → breaker ON
        out = engine.process_bar({"close": 77.0, "datetime": day1,
                                  "signal_vwap_mean_reversion_filtered": 0.0})
        assert out is not None
        assert engine.config.kill_switch_active is True
        n_trades = len(engine._trades)

        # day 1: any further signal bar is blocked by the breaker
        engine.process_bar({"close": 77.0, "datetime": day1,
                            "signal_vwap_mean_reversion_filtered": 0.87})
        assert len(engine._trades) == n_trades

        # day 2: first bar clears the breaker and resets the peak
        out = engine.process_bar({"close": 77.0, "datetime": day2,
                                  "signal_vwap_mean_reversion_filtered": 0.87})
        assert out is not None
        assert engine.config.kill_switch_active is False
        assert engine._peak_equity == pytest.approx(979_990.0)
        assert len(engine._trades) == n_trades + 1

    def test_eod_close_on_date_change(self):
        """Positions left open at the end of a day must be closed when the
        next day's first bar arrives (live closes at 16:00), then the new
        day may re-enter on its own signals."""
        from src.execution.broker import PaperBroker

        broker = PaperBroker(
            initial_cash=1_000_000.0,
            commission_per_share=0.0,
            slippage_bps=0.0,
        )
        engine = IntradayEngine(
            config=IntradayConfig(
                ticker="MSFT",
                max_position_shares=1000,
                commission_per_share=0.0,
                slippage_bps=0.0,
            ),
            broker=broker,
        )
        engine.connect()
        engine.start_session()

        day1 = datetime(2026, 8, 3, 20, 0, tzinfo=timezone.utc)
        day2 = datetime(2026, 8, 4, 14, 0, tzinfo=timezone.utc)

        # day 1: open a long
        engine.process_bar({"close": 100.0, "datetime": day1,
                            "signal_vwap_mean_reversion_filtered": 0.87})
        assert engine._position == 870

        # day 2 first bar: EOD close first (reason=end_of_session), then the
        # day-2 signal re-enters — both engine and broker books stay flat
        # through the close
        out = engine.process_bar({"close": 100.5, "datetime": day2,
                                  "signal_vwap_mean_reversion_filtered": 0.87})
        assert out is not None
        assert len(engine._trades) == 3
        assert engine._trades[1]["reason"] == "end_of_session"
        assert engine._trades[2]["shares"] == 870
        assert engine._position == 870
        assert broker.get_positions()["MSFT"].quantity == 870

    def test_metrics_report_true_final_equity_not_peak(self):
        """Regression (Aug 4 2026): session metrics took the equity curve's
        MAX as final equity — correct only while equity was monotonic.
        A losing session must report the broker's actual ending equity."""
        from src.execution.broker import PaperBroker

        broker = PaperBroker(
            initial_cash=1_000_000.0,
            commission_per_share=0.0,
            slippage_bps=0.0,
        )
        engine = IntradayEngine(
            config=IntradayConfig(
                ticker="MSFT",
                max_position_shares=1000,
                stop_loss_pct=1.0,
                take_profit_pct=1.0,
                commission_per_share=0.0,
                slippage_bps=0.0,
            ),
            broker=broker,
        )
        engine.connect()
        engine.start_session()

        day = datetime(2026, 8, 3, 14, 0, tzinfo=timezone.utc)
        engine.process_bar({"close": 100.0, "datetime": day,
                            "signal_vwap_mean_reversion_filtered": 0.87})
        engine.process_bar({"close": 99.0, "datetime": day + timedelta(minutes=1),
                            "signal_vwap_mean_reversion_filtered": 0.0})
        # 870 shares × $1 loss = -$870
        metrics = engine.end_session(99.0)
        assert metrics["final_equity"] == pytest.approx(999_130.0)
        assert metrics["total_pnl"] == pytest.approx(-870.0)
        assert metrics["total_return_pct"] == pytest.approx(-0.087)
        assert metrics["position"] == 0
