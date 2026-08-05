"""Intraday runner — bar-by-bar streaming trading loop.

Connects to IBKR for real-time bar streaming, computes intraday features,
generates signals, and executes trades through the IntradayEngine.

Pipeline:
    IBKR streaming bars → compute_intraday_features → signal → IntradayEngine → Broker → Market

Supports both:
- Live mode: Real-time streaming from IBKR Gateway/TWS
- Backtest mode: Historical bars from parquet files

Usage (live):
    from src.live.intraday_runner import IntradayRunner

    runner = IntradayRunner()
    runner.run_session()  # Runs until market close

Usage (backtest):
    runner = IntradayRunner(backtest=True, data_dir="data/intraday/1min")
    results = runner.run_backtest("2025-01-01", "2025-03-31")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, time as ttime, timezone
from pathlib import Path
from typing import Optional

import polars as pl
from datetime import date as _date

from src.execution.intraday_engine import IntradayConfig, IntradayEngine
from src.execution.broker import PaperBroker
from src.execution.models import Side
from src.features.intraday import compute_intraday_features
from src.signals.vwap_reversion_filtered import generate_vwap_mean_reversion_filtered

log = logging.getLogger(__name__)

# US market times (UTC)
MARKET_OPEN_UTC = ttime(13, 30)   # 09:30 ET year-round (13:30 UTC EDT / 14:30 UTC EST)
MARKET_CLOSE_UTC = ttime(20, 0)   # 16:00 ET year-round (20:00 UTC EDT / 21:00 UTC EST)
MARKET_OPEN_ET = ttime(9, 30)
MARKET_CLOSE_ET = ttime(16, 0)


@dataclass
class IntradayRunnerConfig:
    """Configuration for the intraday runner.

    Args:
        tickers: List of tickers to trade.
        timeframe: Bar timeframe (e.g., "1min", "5min").
        initial_cash: Starting cash balance.
        max_position_shares: Maximum shares per position.
        stop_loss_pct: Stop loss percentage.
        take_profit_pct: Take profit percentage.
        use_ibkr: Whether to connect to IBKR for live trading.
        backtest: Whether to run in backtest mode.
        data_dir: Path to intraday data for backtesting.
        signal_column: Name of the signal column to use.
        report_dir: Directory for daily reports.
    """

    tickers: list[str] = field(default_factory=lambda: ["MSFT"])
    timeframe: str = "1min"
    initial_cash: float = 1_000_000.0
    max_position_shares: int = 10000
    stop_loss_pct: float = 0.005
    take_profit_pct: float = 0.0075
    commission_per_share: float = 0.005
    slippage_bps: float = 10.0

    # IBKR
    use_ibkr: bool = False
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 4002

    # Backtest
    backtest: bool = False
    data_dir: Optional[Path] = None

    # Signal
    signal_column: str = "signal_vwap_mean_reversion_filtered"
    # Minimum |signal| to act on — weaker signals are zeroed (engine closes/
    # flattens), matching the threshold sweep's pre-filter semantics exactly.
    # 0.0 = trade every nonzero signal (current live behavior).
    signal_threshold: float = 0.0

    # Session
    market_open_utc: ttime = MARKET_OPEN_UTC
    market_close_utc: ttime = MARKET_CLOSE_UTC

    # Reports
    report_dir: Path = field(default_factory=lambda: Path("reports/intraday"))

    def __post_init__(self):
        self.report_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class BarRecord:
    """A single bar of intraday data."""

    ticker: str
    datetime: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class IntradayRunner:
    """Bar-by-bar intraday trading runner.

    Manages the full intraday trading loop:
    1. Wait for market open (or load historical data)
    2. Stream bars and compute features incrementally
    3. Generate signals and execute trades
    4. Monitor risk and close positions at market close
    5. Generate end-of-day reports

    Args:
        config: Runner configuration.
    """

    def __init__(self, config: Optional[IntradayRunnerConfig] = None):
        self.config = config or IntradayRunnerConfig()
        self._engines: dict[str, IntradayEngine] = {}
        self._bar_buffer: dict[str, list[BarRecord]] = {t: [] for t in self.config.tickers}
        # Multi-minute aggregation state (timeframe != "1min")
        self._hour_bars: dict[str, list[BarRecord]] = {t: [] for t in self.config.tickers}
        self._active_hour_key: dict[str, Optional[datetime]] = {t: None for t in self.config.tickers}
        self._session_date: Optional[date] = None
        self._running: bool = False

    # ------------------------------------------------------------------
    # Bar aggregation (multi-minute timeframes)
    # ------------------------------------------------------------------

    def _bar_interval_minutes(self) -> int:
        """Minutes per aggregated bar from config.timeframe ('1min','5m','15m','30m','1h')."""
        tf = self.config.timeframe.strip().lower()
        if tf.endswith("min"):
            return max(1, int(tf[:-3]))
        if tf.endswith("m"):
            return max(1, int(tf[:-1]))
        if tf.endswith("h"):
            return int(tf[:-1]) * 60
        raise ValueError(f"Unsupported timeframe: {self.config.timeframe}")

    @staticmethod
    def _window_key(dt: datetime) -> datetime:
        """UTC window key for :30-anchored aggregation — identical alignment
        to the backtest's group_by_dynamic(..., every='1h', offset='30m').

        Windows start at HH:30 UTC; minutes 13:31..14:29 share one key.
        """
        shifted = dt - timedelta(minutes=30)
        return datetime(shifted.year, shifted.month, shifted.day,
                        shifted.hour, 0, tzinfo=timezone.utc) if shifted.tzinfo else \
            shifted.replace(minute=0, second=0, microsecond=0)

    def _flush_hour(self, ticker: str) -> Optional[BarRecord]:
        """Aggregate the current minute-accumulator into one bar at the
        window start (matching resample_1h's include_boundaries=False)."""
        acc = self._hour_bars.get(ticker)
        key = self._active_hour_key.get(ticker)
        if not acc or key is None:
            return None
        rec = BarRecord(
            ticker=ticker,
            datetime=key + timedelta(minutes=30),
            open=acc[0].open,
            high=max(b.high for b in acc),
            low=min(b.low for b in acc),
            close=acc[-1].close,
            volume=sum(b.volume for b in acc),
        )
        self._hour_bars[ticker] = []
        self._active_hour_key[ticker] = None
        return rec

    def _ingest_minute(self, ticker: str, bar: BarRecord,
                       warm_only: bool = False) -> Optional[dict]:
        """Route one delivered 1-min bar.

        '1min' mode (or any interval ≤ the delivery granularity): forwards
        straight to process_bar — byte-identical behavior to pre-change runs.
        Multi-minute modes ('1h', ...): accumulates; only COMPLETED windows
        are flushed and reach the signal/engine path. warm_only feeds the
        aggregated bars into the feature buffer without any trading (same
        contract as _warmup_bars for the backfill).
        """
        if self._bar_interval_minutes() <= 1:
            if warm_only:
                buf = self._bar_buffer.setdefault(ticker, [])
                buf.append(bar)
                return None
            return self.process_bar(ticker, bar)

        key = self._window_key(bar.datetime)
        active = self._active_hour_key.get(ticker)
        flushed: Optional[BarRecord] = None
        if self._hour_bars[ticker] and key != active:
            rec = self._flush_hour(ticker)    # previous window is complete
            if rec is not None:
                flushed = rec
                buf = self._bar_buffer.setdefault(ticker, [])
                buf.append(rec)               # keep feature buffer fresh
                if len(buf) > 500:
                    self._bar_buffer[ticker] = buf[-500:]
        # Open the new window (if any) and store the current minute —
        # state MUST be fully updated before anything is processed.
        if key != active:
            self._active_hour_key[ticker] = key
        self._hour_bars.setdefault(ticker, []).append(bar)
        if flushed is not None and not warm_only:
            return self.process_bar(ticker, flushed)
        return None

    def _get_engine(self, ticker: str) -> IntradayEngine:
        """Get or create an engine for a ticker."""
        if ticker not in self._engines:
            if self.config.use_ibkr:
                from src.execution.ib_broker import IBBroker, IBConfig
                ib_config = IBConfig(
                    tws_host=self.config.ibkr_host,
                    tws_port=self.config.ibkr_port,
                )
                broker = IBBroker(ib_config)
            else:
                broker = PaperBroker(
                    initial_cash=self.config.initial_cash / len(self.config.tickers),
                    commission_per_share=self.config.commission_per_share,
                    slippage_bps=self.config.slippage_bps,
                    price_source={},
                )
            engine_config = IntradayConfig(
                ticker=ticker,
                timeframe=self.config.timeframe,
                signal_column=self.config.signal_column,
                max_position_shares=self.config.max_position_shares,
                stop_loss_pct=self.config.stop_loss_pct,
                take_profit_pct=self.config.take_profit_pct,
                commission_per_share=self.config.commission_per_share,
                slippage_bps=self.config.slippage_bps,
                initial_cash=self.config.initial_cash / len(self.config.tickers),
                market_open_utc=self.config.market_open_utc,
                market_close_utc=self.config.market_close_utc,
            )
            self._engines[ticker] = IntradayEngine(engine_config, broker=broker)
        return self._engines[ticker]

    def _compute_signal(self, ticker: str, bar: BarRecord) -> dict:
        """Compute signal for a single bar.

        Maintains a rolling buffer of bars per ticker and computes
        features + signals incrementally.

        Args:
            ticker: Ticker symbol.
            bar: Current bar.

        Returns:
            Dict with bar data + signal column.
        """
        # Add bar to buffer
        self._bar_buffer[ticker].append(bar)

        # Keep last 500 bars for feature computation
        if len(self._bar_buffer[ticker]) > 500:
            self._bar_buffer[ticker] = self._bar_buffer[ticker][-500:]

        # Build DataFrame from buffer
        bars = self._bar_buffer[ticker]
        if len(bars) < 60:  # Need minimum bars for features
            return {
                "close": bar.close,
                "datetime": bar.datetime,
                self.config.signal_column: 0.0,
            }

        df = pl.DataFrame([{
            "ticker": b.ticker,
            "datetime": b.datetime,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
        } for b in bars])

        # Compute features
        df = compute_intraday_features(df)

        # Generate signal
        df = generate_vwap_mean_reversion_filtered(df)

        # Get the latest signal
        latest = df.tail(1)
        # Null signals occur during feature warm-up (rolling VWAP needs 100
        # samples) and session edges; treat them as "no signal" (0.0) — never
        # crash the loop on float(None). The engine treats 0 as no action.
        raw_signal = float(latest["signal_vwap_mean_reversion_filtered"].item() or 0.0)
        # Threshold gate: |signal| < threshold → 0.0 (engine closes/flattens),
        # identical to the sweep's pre-filter semantics. Default 0.0 = act on
        # every nonzero signal.
        if abs(raw_signal) < self.config.signal_threshold:
            raw_signal = 0.0

        return {
            "close": bar.close,
            "datetime": bar.datetime,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "volume": bar.volume,
            self.config.signal_column: float(raw_signal),
        }

    def _warmup_bars(self, ticker: str, bars: list[BarRecord]) -> None:
        """Feed historical bars into the feature buffer WITHOUT trading.

        The first fetch of a session returns the 24h window; those bars exist
        only to warm the feature lookbacks (rolling VWAP, ADX, volume). The
        engine must never act on them.
        """
        buf = self._bar_buffer.setdefault(ticker, [])
        buf.extend(bars)
        if len(buf) > 500:
            self._bar_buffer[ticker] = buf[-500:]

    def process_bar(self, ticker: str, bar: BarRecord) -> Optional[dict]:
        """Process a single bar through the full pipeline.

        Args:
            ticker: Ticker symbol.
            bar: Bar data.

        Returns:
            Trade info if a trade was executed, None otherwise.
        """
        engine = self._get_engine(ticker)

        # Compute signal
        bar_with_signal = self._compute_signal(ticker, bar)

        # Process through engine
        return engine.process_bar(bar_with_signal)

    # ------------------------------------------------------------------
    # Backtest mode
    # ------------------------------------------------------------------

    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        tickers: Optional[list[str]] = None,
    ) -> dict:
        """Run a backtest on historical intraday data.

        Loads parquet files from the data directory, computes features
        vectorized per-session, and processes bars sequentially.

        Args:
            start_date: Start date (ISO format).
            end_date: End date (ISO format).
            tickers: Override tickers to backtest.

        Returns:
            Dict with per-ticker and aggregate metrics.
        """
        test_tickers = tickers or self.config.tickers
        data_dir = self.config.data_dir or Path("data/intraday/1min")

        log.info(
            f"Starting backtest: {start_date} to {end_date}, "
            f"tickers={test_tickers}"
        )

        all_metrics = {}
        total_bars = 0

        for ticker in test_tickers:
            engine = self._get_engine(ticker)
            engine.broker.connect()

            # Load data
            ticker_dir = data_dir / f"ticker={ticker}"
            if not ticker_dir.exists():
                log.warning(f"No data for {ticker} at {ticker_dir}")
                continue

            # Collect all parquet files in date range
            parquet_files = sorted(ticker_dir.rglob("*.parquet"))
            log.info(f"Found {len(parquet_files)} files for {ticker}")

            # Concatenate all files, then filter by date
            dfs = []
            for pf in parquet_files:
                df = pl.read_parquet(str(pf))
                if "date" in df.columns:
                    start_d = _date.fromisoformat(start_date)
                    end_d = _date.fromisoformat(end_date)
                    df = df.filter(
                        (pl.col("date") >= start_d) &
                        (pl.col("date") <= end_d)
                    )
                if not df.is_empty():
                    dfs.append(df)

            if not dfs:
                log.warning(f"No data loaded for {ticker} in range")
                continue

            # Group by date for session management
            all_data = pl.concat(dfs, how="vertical_relaxed")
            dates = all_data["date"].unique().sort().to_list()
            log.info(f"Processing {len(dates)} sessions for {ticker}")

            session_count = 0
            for d in dates:
                day_df = all_data.filter(pl.col("date") == d)
                if day_df.is_empty():
                    continue

                # Compute features vectorized for the entire session
                feature_df = self._compute_session_features(ticker, day_df)

                # Start session
                engine.start_session()
                session_count += 1

                # Process bars with pre-computed signals
                for row in feature_df.iter_rows(named=True):
                    signal_col = self.config.signal_column
                    signal_val = row.get(signal_col, 0.0)
                    if signal_val is None:
                        signal_val = 0.0

                    bar_dict = {
                        "close": float(row["close"]),
                        "datetime": row["datetime"],
                        "open": float(row.get("open", row["close"])),
                        "high": float(row.get("high", row["close"])),
                        "low": float(row.get("low", row["close"])),
                        "volume": int(row.get("volume", 0)),
                        signal_col: float(signal_val),
                    }
                    engine.process_bar(bar_dict)
                    total_bars += 1

                # End session
                metrics = engine.end_session()
                all_metrics[f"{ticker}_{d.isoformat()}"] = metrics

            # Reset buffer for next ticker
            self._bar_buffer[ticker] = []

        # Aggregate results
        aggregate = self._aggregate_metrics(all_metrics)

        log.info(
            f"Backtest complete: {total_bars} bars processed, "
            f"{len(all_metrics)} sessions"
        )

        return {
            "per_session": all_metrics,
            "aggregate": aggregate,
            "total_bars": total_bars,
        }

    def _compute_session_features(self, ticker: str, day_df: pl.DataFrame) -> pl.DataFrame:
        """Compute features and signals for an entire session vectorized.

        Args:
            ticker: Ticker symbol.
            day_df: DataFrame with one session's bars.

        Returns:
            DataFrame with signal column appended.
        """
        # Ensure required columns exist
        required = ["datetime", "open", "high", "low", "close", "volume"]
        for col in required:
            if col not in day_df.columns:
                day_df = day_df.with_columns(pl.lit(0.0).alias(col))

        # Add ticker column if missing
        if "ticker" not in day_df.columns:
            day_df = day_df.with_columns(pl.lit(ticker).alias("ticker"))

        # Sort by time
        day_df = day_df.sort("datetime")

        # Compute features vectorized
        feature_df = compute_intraday_features(day_df)

        # Generate signal vectorized
        feature_df = generate_vwap_mean_reversion_filtered(feature_df)

        return feature_df

    def _aggregate_metrics(self, session_metrics: dict) -> dict:
        """Aggregate metrics across all sessions.

        Args:
            session_metrics: Dict mapping session key -> metrics dict.

        Returns:
            Aggregated metrics.
        """
        if not session_metrics:
            return {}

        total_pnl = sum(m.get("total_pnl", 0) for m in session_metrics.values())
        total_trades = sum(m.get("total_trades", 0) for m in session_metrics.values())
        winning = sum(m.get("winning_trades", 0) for m in session_metrics.values())
        losing = sum(m.get("losing_trades", 0) for m in session_metrics.values())

        avg_return = (
            sum(m.get("total_return_pct", 0) for m in session_metrics.values())
            / len(session_metrics)
        )

        avg_sharpe = (
            sum(m.get("sharpe_ratio", 0) for m in session_metrics.values())
            / len(session_metrics)
        )

        max_dd = max(
            (abs(m.get("max_drawdown_pct", 0)) for m in session_metrics.values()),
            default=0,
        )

        return {
            "total_sessions": len(session_metrics),
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "winning_trades": winning,
            "losing_trades": losing,
            "win_rate_pct": (winning / (winning + losing) * 100) if (winning + losing) > 0 else 0,
            "avg_session_return_pct": avg_return,
            "avg_sharpe_ratio": avg_sharpe,
            "max_drawdown_pct": max_dd,
        }

    # ------------------------------------------------------------------
    # Live mode
    # ------------------------------------------------------------------

    def run_session(self) -> dict:
        """Run a live trading session.

        Connects to IBKR, waits for market open, streams bars,
        and runs until market close.

        Returns:
            Dict with session metrics.
        """
        if self.config.backtest:
            raise RuntimeError("Use run_backtest() for backtest mode")

        self._session_date = date.today()
        self._running = True

        log.info(f"Starting live session: {self._session_date}")

        try:
            # Connect engines
            for ticker in self.config.tickers:
                engine = self._get_engine(ticker)
                engine.connect()
                engine.start_session()

            # Wait for market open
            log.info(f"Waiting for market open at {self.config.market_open_utc} UTC...")
            self._wait_for_market_open()

            # Main loop
            log.info("Market open — starting bar stream")
            # Warm the feature buffer with the initial backfill — the first
            # fetch returns the 24h window (yesterday's session + pre-market)
            # and those bars must NEVER trigger orders (Aug 4 2026: the engine
            # traded the backfill as a tight loop before this warm-up existed).
            for ticker in self.config.tickers:
                warm = self._fetch_bars(ticker)
                if self._bar_interval_minutes() > 1:
                    # Multi-minute mode: feed the backfill through the
                    # aggregator — completed windows fill the feature buffer,
                    # nothing is traded. Any partial trailing window from the
                    # fetch (stale historical minutes) must never be flushed
                    # by live delivery, so drop it here.
                    for b in warm:
                        self._ingest_minute(ticker, b, warm_only=True)
                    self._hour_bars[ticker] = []
                    self._active_hour_key[ticker] = None
                else:
                    self._warmup_bars(ticker, warm)
                log.info(f"{ticker}: buffer warmed with {len(warm)} historical bars (no trading)")

            while self._running and not self._is_market_closed():
                # Fetch and process bars for each ticker
                for ticker in self.config.tickers:
                    bars = self._fetch_bars(ticker)
                    for bar in bars:
                        self._ingest_minute(ticker, bar)

                # Brief sleep to avoid busy-waiting
                time.sleep(1)

            # End session
            log.info("Market close — ending session")
            metrics = self._end_all_sessions()

        except KeyboardInterrupt:
            log.info("Interrupted by user")
            metrics = self._end_all_sessions()

        except Exception as e:
            log.error(f"Session error: {e}", exc_info=True)
            metrics = {"error": str(e)}

        finally:
            self._disconnect_all()
            self._running = False

        return metrics

    def _wait_for_market_open(self) -> None:
        """Block until 09:30 ET (America/New_York, DST-aware)."""
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("America/New_York"))
        target = now.replace(
            hour=9,
            minute=30,
            second=0,
            microsecond=0,
        )

        if now < target:
            wait_secs = (target - now).total_seconds()
            log.info(f"Waiting {wait_secs:.0f} seconds for market open")
            time.sleep(wait_secs)

    def _is_market_closed(self) -> bool:
        """Check if market has closed (16:00 ET, DST-aware)."""
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("America/New_York"))
        close_time = now.replace(
            hour=16,
            minute=0,
            second=0,
            microsecond=0,
        )
        return now >= close_time

    def _fetch_bars(self, ticker: str) -> list[BarRecord]:
        """Fetch latest bars from IBKR, falling back to yfinance 1-min polling.

        IBKR tick streaming requires a market data subscription; paper accounts
        without one return error 10089 and the tick buffer stays empty forever
        (the engine previously wedged silently). After a few empty poll cycles
        this method switches the ticker to yfinance for the rest of the session.
        Set ALPHA_DATA_SOURCE=ibkr|yfinance to force a source ("auto" default).
        """
        if not self.config.use_ibkr:
            return []

        engine = self._engines.get(ticker)
        if not engine or not engine.is_connected:
            return []

        # Per-ticker data source state
        if not hasattr(self, "_data_source"):
            self._data_source = {}
        if not hasattr(self, "_empty_polls"):
            self._empty_polls = {}

        import os
        forced = os.environ.get("ALPHA_DATA_SOURCE", "auto").lower()
        if forced in ("alpaca", "yfinance"):
            self._data_source[ticker] = forced
        if self._data_source.get(ticker) == "alpaca":
            return self._fetch_bars_alpaca(ticker)
        if self._data_source.get(ticker) == "yfinance":
            return self._fetch_bars_yfinance(ticker)

        # --- IBKR tick path ---
        try:
            from src.execution.ib_broker import IBBroker
            if not isinstance(engine.broker, IBBroker):
                return []

            client = engine.broker._client
            if not client.connected:
                return []

            contract = client.stock(ticker)
            client._ib.qualifyContracts(contract)

            # Collect ticks for this poll cycle
            tick_buffer: list[dict] = []

            def tick_callback(tick):
                tick_buffer.append({
                    "price": tick.price,
                    "volume": tick.size,
                    "timestamp": tick.time,
                })

            # Subscribe to streaming ticks
            client._ib.reqTickByTickData(
                contract,
                "TickByTickLast",
                0,  # numberOfTicks=0 for continuous stream
                False,  # ignoreSize=False
            )

            # Small window to collect ticks
            time.sleep(0.5)

            # Cancel subscription after collecting
            client._ib.cancelTickByTickData(contract, "Last")

            if not tick_buffer:
                strikes = self._empty_polls.get(ticker, 0) + 1
                self._empty_polls[ticker] = strikes
                if strikes >= 3:
                    log.warning(
                        f"{ticker}: no IBKR ticks after {strikes} polls — market data "
                        "subscription unavailable (error 10089 on paper accounts). "
                        "Switching to alpaca 1-min polling for this session."
                    )
                    self._data_source[ticker] = "alpaca"
                    return self._fetch_bars_alpaca(ticker)
                return []

            self._empty_polls[ticker] = 0

            # Feed ticks to bar builder
            builder = self._get_bar_builder(ticker)
            from src.execution.bar_builder import Tick
            for t in tick_buffer:
                builder.update(Tick(
                    price=t["price"],
                    volume=t["volume"],
                    timestamp=t["timestamp"],
                ))

            # Return completed bars
            completed = list(builder.completed)
            builder.completed.clear()
            return completed

        except Exception as e:
            log.warning(f"Failed to fetch bars for {ticker}: {e}")
            # Count exceptions as empty polls so a broken IBKR path (API
            # mismatch, missing subscription, etc.) still trips the fallback.
            strikes = self._empty_polls.get(ticker, 0) + 1
            self._empty_polls[ticker] = strikes
            if strikes >= 3:
                log.warning(
                    f"{ticker}: IBKR bar fetch failing repeatedly ({e}) — "
                    "switching to alpaca 1-min polling for this session."
                )
                self._data_source[ticker] = "alpaca"
                return self._fetch_bars_alpaca(ticker)
            return []

    def _fetch_bars_yfinance(self, ticker: str) -> list[BarRecord]:
        """Poll yfinance for completed 1-minute bars (fallback data source).

        yfinance intraday data trails by ~1-2 minutes; only completed bars are
        emitted so the strategy never acts on a partially-formed minute. Bars
        are deduplicated by minute timestamp across polls.

        The fetch runs in a single-worker thread pool with a 30s timeout:
        yfinance's history() has no built-in timeout and can hang indefinitely
        on a stalled connection, which would wedge the whole session loop.
        """
        if not hasattr(self, "_yf_last_fetch"):
            self._yf_last_fetch = {}
        if not hasattr(self, "_yf_seen"):
            self._yf_seen = {}
        if not hasattr(self, "_yf_pool"):
            from concurrent.futures import ThreadPoolExecutor
            self._yf_pool = ThreadPoolExecutor(max_workers=1)

        now = datetime.now(timezone.utc)
        last = self._yf_last_fetch.get(ticker)
        if last is not None and (now - last).total_seconds() < 55:
            return []
        self._yf_last_fetch[ticker] = now

        try:
            from concurrent.futures import TimeoutError as FutureTimeout
            future = self._yf_pool.submit(self._yf_fetch, ticker)
            hist = future.result(timeout=30)
        except FutureTimeout:
            log.warning(f"yfinance fetch timed out for {ticker} — skipping cycle")
            return []
        except Exception as e:
            log.warning(f"yfinance fetch failed for {ticker}: {e}")
            return []

        if hist is None or hist.empty:
            log.info(f"{ticker}: yfinance fetch returned no data (empty frame)")
            return []

        hist = hist.reset_index()
        seen = self._yf_seen.setdefault(ticker, set())
        cutoff = now.replace(second=0, microsecond=0)
        new_bars: list[BarRecord] = []
        for row in hist.itertuples():
            dt = row.Datetime
            # Normalize to aware UTC: the feature pipeline anchors sessions in
            # America/New_York and the parquet convention stores UTC. Naive
            # timestamps are assumed UTC (never local) so a process running in
            # a UTC-tainted environment cannot shift the session window.
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            minute_key = dt.replace(second=0, microsecond=0)
            if minute_key in seen:
                continue
            if minute_key >= cutoff:
                continue  # incomplete (current) bar — do NOT mark seen, retry next poll
            seen.add(minute_key)
            new_bars.append(BarRecord(
                ticker=ticker,
                datetime=dt,
                open=float(row.Open),
                high=float(row.High),
                low=float(row.Low),
                close=float(row.Close),
                volume=int(row.Volume),
            ))
        if new_bars:
            log.info(
                f"{ticker}: yfinance delivered {len(new_bars)} new 1-min bars "
                f"(latest {new_bars[-1].datetime})"
            )
        return new_bars

    @staticmethod
    def _yf_fetch(ticker: str):
        """Fetch 1-minute bars from yfinance (runs in a worker thread)."""
        import yfinance as yf
        return yf.Ticker(ticker).history(period="1d", interval="1m", prepost=False)

    def _fetch_bars_alpaca(self, ticker: str) -> list[BarRecord]:
        """Poll Alpaca for completed 1-minute bars (primary fallback source).

        Same contract as the yfinance path: aware-UTC BarRecords, dedupe by
        UTC minute, current incomplete minute excluded and retried next poll.
        Uses the same REST bars endpoint as the backtest lake (src/data/alpaca.py),
        so live bars match the 2025 backtest feed by construction.
        """
        if not hasattr(self, "_data_source"):
            self._data_source = {}
        if not hasattr(self, "_alpaca_last_fetch"):
            self._alpaca_last_fetch = {}
        if not hasattr(self, "_alpaca_seen"):
            self._alpaca_seen = {}
        if not hasattr(self, "_alpaca_strikes"):
            self._alpaca_strikes = {}
        if not hasattr(self, "_alpaca_warmup_done"):
            self._alpaca_warmup_done = {}
        if not hasattr(self, "_alpaca_pool"):
            from concurrent.futures import ThreadPoolExecutor
            self._alpaca_pool = ThreadPoolExecutor(max_workers=1)

        now = datetime.now(timezone.utc)
        last = self._alpaca_last_fetch.get(ticker)
        if last is not None and (now - last).total_seconds() < 55:
            return []
        self._alpaca_last_fetch[ticker] = now

        # The first fetch of the session is the warm-up backfill. Multi-minute
        # timeframes need ~65 calendar days of 1-min bars so the ~390-bar
        # feature lookback is warm from day 1; the 1min path keeps its 24h
        # window. Later polls always use the rolling 24h (alpaca-py 0.43.5
        # paginates internally — page size 10k, next_page_token loop — so one
        # call returns the full window).
        warmup_done = self._alpaca_warmup_done.get(ticker, False)
        lookback_hours = 65 * 24 if (not warmup_done and self._bar_interval_minutes() > 1) else 24

        try:
            from concurrent.futures import TimeoutError as FutureTimeout
            future = self._alpaca_pool.submit(self._alpaca_fetch, ticker, now, lookback_hours)
            hist = future.result(timeout=30)
        except FutureTimeout:
            log.warning(f"alpaca fetch timed out for {ticker} — skipping cycle")
            hist = None
        except Exception as e:
            log.warning(f"alpaca fetch failed for {ticker}: {e}")
            hist = None

        if hist is None or hist.empty:
            strikes = self._alpaca_strikes.get(ticker, 0) + 1
            self._alpaca_strikes[ticker] = strikes
            if strikes >= 3:
                log.warning(
                    f"{ticker}: alpaca returning no data after {strikes} polls — "
                    "switching to yfinance 1-min polling for this session."
                )
                self._data_source[ticker] = "yfinance"
                return self._fetch_bars_yfinance(ticker)
            return []

        self._alpaca_strikes[ticker] = 0
        self._alpaca_warmup_done[ticker] = True
        seen = self._alpaca_seen.setdefault(ticker, set())
        cutoff = now.replace(second=0, microsecond=0)
        new_bars: list[BarRecord] = []
        for row in hist.itertuples():
            dt = row.timestamp
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)     # naive assumed UTC
            else:
                dt = dt.astimezone(timezone.utc)
            minute_key = dt.replace(second=0, microsecond=0)
            if minute_key in seen:
                continue
            if minute_key >= cutoff:
                continue  # incomplete (current) minute — do NOT mark seen
            seen.add(minute_key)
            new_bars.append(BarRecord(
                ticker=ticker,
                datetime=dt,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=int(row.volume),
            ))
        if new_bars:
            log.info(
                f"{ticker}: alpaca delivered {len(new_bars)} new 1-min bars "
                f"(latest {new_bars[-1].datetime})"
            )
        return new_bars

    @staticmethod
    def _alpaca_fetch(ticker: str, now: datetime, lookback_hours: int = 24):
        """Fetch 1-minute bars from Alpaca (runs in a worker thread).

        lookback_hours: width of the fetch window. The session-start warm-up
        backfill passes 65*24 for multi-minute timeframes; live polls pass 24.
        """
        import os
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import DataFeed

        client = StockHistoricalDataClient(
            api_key=os.environ["ALPACA_API_KEY"],
            secret_key=os.environ["ALPACA_API_SECRET"],
        )
        req = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=TimeFrame.Minute,
            start=now - timedelta(hours=lookback_hours),
            end=now,
            # feed pinned to "iex": the free-tier account defaults to the SIP
            # feed for RECENT data and rejects it ("subscription does not
            # permit querying recent SIP data", observed 2026-08-04), while the
            # 2025 backtest lake was built from IEX (verified byte-identical for
            # 2025-07-15: 564 bars, volume sum 14,815,743, delta 0.0). IEX keeps
            # live bars consistent with the lake by construction.
            feed=DataFeed.IEX,
        )
        return client.get_stock_bars(req).df.reset_index()

    def _get_bar_builder(self, ticker: str):
        """Get or create a BarBuilder for a ticker."""
        from src.execution.bar_builder import BarBuilder
        if not hasattr(self, "_bar_builders"):
            self._bar_builders = {}

        if ticker not in self._bar_builders:
            interval_minutes = self._parse_interval(self.config.timeframe)
            self._bar_builders[ticker] = BarBuilder(
                ticker=ticker,
                interval_minutes=interval_minutes,
            )
        return self._bar_builders[ticker]

    @staticmethod
    def _parse_interval(timeframe: str) -> int:
        """Parse timeframe string to minutes."""
        if timeframe.endswith("min"):
            return int(timeframe[:-3])
        elif timeframe.endswith("m"):
            return int(timeframe[:-1])
        elif timeframe.endswith("h"):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith("s"):
            secs = int(timeframe[:-1])
            return max(1, secs // 60)
        else:
            # Default: treat as minutes
            try:
                return int(timeframe)
            except ValueError:
                return 1

    def _end_all_sessions(self) -> dict:
        """End all ticker sessions and collect metrics."""
        all_metrics = {}
        for ticker, engine in self._engines.items():
            metrics = engine.end_session()
            all_metrics[ticker] = metrics
        return all_metrics

    def _disconnect_all(self) -> None:
        """Disconnect all engines."""
        for engine in self._engines.values():
            engine.disconnect()

    def stop(self) -> None:
        """Stop the running session."""
        self._running = False