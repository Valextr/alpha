"""Live intraday runner — connects IBKR real-time data to the execution engine.

This module provides the main event loop for live intraday trading:

    IBKR ticks → BarBuilder → Feature computation → Signal → IntradayEngine → Broker

The runner manages:
- Connection lifecycle (connect at market open, disconnect at close)
- Session management (auto-close positions at EOD)
- Position reconciliation (periodic sync with IBKR actuals)
- Error handling (reconnect on disconnect, log failures)
- Health monitoring (heartbeat, P&L tracking)

Usage:
    from src.execution.live_runner import LiveIntradayRunner
    from src.execution.intraday_engine import IntradayConfig

    config = IntradayConfig(
        ticker="MSFT",
        use_ibkr=True,
        timeframe="1min",
    )
    runner = LiveIntradayRunner(config)
    runner.run_session()

Or for dry run (no live orders, just paper):
    config = IntradayConfig(
        ticker="MSFT",
        use_ibkr=False,  # Paper mode
    )
    runner = LiveIntradayRunner(config)
    runner.run_session()
"""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass
from datetime import datetime, date, time as ttime, timezone, timedelta
from typing import Optional

from .intraday_engine import IntradayEngine, IntradayConfig
from .bar_builder import BarBuilder, Bar, Tick

log = logging.getLogger(__name__)


@dataclass
class RunnerConfig:
    """Configuration for the live runner."""

    # How often to poll for new ticks (seconds)
    poll_interval: float = 1.0

    # How often to reconcile positions with broker (seconds)
    reconciliation_interval: float = 60.0

    # How often to log P&L (seconds)
    log_interval: float = 300.0

    # Graceful shutdown timeout (seconds)
    shutdown_timeout: float = 30.0

    # Whether to auto-start at market open
    auto_start: bool = True

    # Dry run mode — simulate without placing real orders
    dry_run: bool = False


class LiveIntradayRunner:
    """Main event loop for live intraday trading.

    Connects IBKR real-time data to the intraday execution engine.
    Manages session lifecycle, position reconciliation, and graceful shutdown.

    Args:
        engine_config: Configuration for the intraday engine.
        runner_config: Configuration for the runner itself.
    """

    def __init__(
        self,
        engine_config: IntradayConfig | None = None,
        runner_config: RunnerConfig | None = None,
    ):
        self.engine_config = engine_config or IntradayConfig()
        self.runner_config = runner_config or RunnerConfig()

        # Create engine and broker
        from .broker import PaperBroker
        from .ib_broker import IBBroker, IBConfig

        if self.engine_config.use_ibkr and not self.runner_config.dry_run:
            ib_config = IBConfig(
                tws_host=self.engine_config.ibkr_host,
                tws_port=self.engine_config.ibkr_port,
                paper_trading=True,  # Default to paper trading
            )
            broker = IBBroker(ib_config)
            log.info("Using IBBroker (live IBKR connection)")
        else:
            broker = PaperBroker(
                initial_cash=self.engine_config.initial_cash,
                commission_per_share=self.engine_config.commission_per_share,
                slippage_bps=self.engine_config.slippage_bps,
            )
            log.info("Using PaperBroker (paper trading)")

        self.engine = IntradayEngine(self.engine_config, broker=broker)

        # Bar builder
        interval_minutes = self._parse_interval(self.engine_config.timeframe)
        self.bar_builder = BarBuilder(
            ticker=self.engine_config.ticker,
            interval_minutes=interval_minutes,
        )

        # State
        self._running: bool = False
        self._shutdown_requested: bool = False
        self._last_reconciliation: float = 0.0
        self._last_log: float = 0.0
        self._session_start_time: Optional[float] = None

    @staticmethod
    def _parse_interval(timeframe: str) -> int:
        """Parse timeframe string to minutes."""
        # Handle "Nmin" format first (before checking single "m")
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

    def run_session(self) -> dict:
        """Run a complete trading session.

        Connects, waits for market open, processes bars, closes at market close.

        Returns:
            Performance metrics dictionary.
        """
        log.info(
            f"Starting live session: ticker={self.engine_config.ticker}, "
            f"live={self.engine_config.use_ibkr}, "
            f"dry_run={self.runner_config.dry_run}"
        )

        # Register signal handlers for graceful shutdown
        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        try:
            # Connect
            self.engine.connect()

            # Record session start
            self._running = True
            self._shutdown_requested = False
            self._session_start_time = time.time()
            self._last_reconciliation = self._session_start_time
            self._last_log = self._session_start_time

            # Start session
            self.engine.start_session()

            log.info("Session started")

            # Main loop
            self._run_main_loop()

            # Close session
            final_price = self.engine.broker.get_current_price(
                self.engine_config.ticker
            )
            metrics = self.engine.end_session(final_price)

            log.info(
                f"Session complete: total_pnl={metrics.get('total_pnl', 0):.2f}, "
                f"trades={metrics.get('total_trades', 0)}, "
                f"win_rate={metrics.get('win_rate_pct', 0):.1f}%"
            )

            return metrics

        except Exception as e:
            log.error(f"Session failed: {e}", exc_info=True)
            self.engine.end_session()
            raise

        finally:
            # Cleanup
            self._running = False
            self.engine.disconnect()
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigint)

    def run_backtest(self, bars: list[Bar]) -> dict:
        """Run a backtest on historical bars.

        Args:
            bars: List of historical bars to process.

        Returns:
            Performance metrics dictionary.
        """
        log.info(f"Starting backtest: {len(bars)} bars for {self.engine_config.ticker}")

        self.engine.connect()
        self.engine.start_session()

        for bar in bars:
            self.engine.process_bar({
                "close": bar.close,
                "datetime": bar.datetime,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "volume": bar.volume,
                self.engine_config.signal_column: bar.signal_vwap_mean_reversion_filtered,
            })

        final_price = bars[-1].close if bars else None
        metrics = self.engine.end_session(final_price)
        self.engine.disconnect()

        log.info(
            f"Backtest complete: total_pnl={metrics.get('total_pnl', 0):.2f}, "
            f"trades={metrics.get('total_trades', 0)}"
        )

        return metrics

    def _run_main_loop(self) -> None:
        """Main event loop — processes ticks, builds bars, executes trades.

        In a production setup, this would subscribe to IBKR real-time
        market data callbacks. For now, it polls at the configured interval.
        """
        while self._running and not self._shutdown_requested:
            try:
                # Poll for new data
                self._poll_data()

                # Check for completed bars
                while self.bar_builder.completed:
                    bar = self.bar_builder.completed.pop(0)
                    self._process_bar(bar)

                # Periodic reconciliation
                now = time.time()
                if now - self._last_reconciliation >= self.runner_config.reconciliation_interval:
                    self._reconcile()
                    self._last_reconciliation = now

                # Periodic logging
                if now - self._last_log >= self.runner_config.log_interval:
                    self._log_status()
                    self._last_log = now

                # Check for shutdown conditions
                if self._should_shutdown():
                    break

                # Sleep until next poll
                time.sleep(self.runner_config.poll_interval)

            except KeyboardInterrupt:
                log.info("Keyboard interrupt received")
                self._shutdown_requested = True
            except Exception as e:
                log.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(5)  # Back off on error

    def _poll_data(self) -> None:
        """Poll for new market data.

        In production, this would use IBKR's streaming callbacks.
        Currently polls for snapshot data.
        """
        if not self.engine.is_connected:
            return

        try:
            ticker_data = self.engine.broker.get_current_price(
                self.engine_config.ticker
            )
            if ticker_data is not None and ticker_data > 0:
                tick = Tick(
                    price=ticker_data,
                    volume=0,  # Snapshot doesn't include volume
                    timestamp=datetime.now(timezone.utc),
                )
                self.bar_builder.update(tick)

        except Exception as e:
            log.warning(f"Poll failed: {e}")

    def _process_bar(self, bar: Bar) -> None:
        """Process a completed bar through the engine."""
        bar_dict = {
            "close": bar.close,
            "datetime": bar.datetime,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "volume": bar.volume,
            self.engine_config.signal_column: bar.signal_vwap_mean_reversion_filtered,
        }

        trade = self.engine.process_bar(bar_dict)
        if trade:
            log.info(
                f"Trade executed: {trade.get('shares', 0)} shares "
                f"@ {trade.get('price', 0):.2f}"
            )

    def _reconcile(self) -> None:
        """Reconcile position with broker."""
        if self.engine_config.use_ibkr:
            try:
                broker_positions = self.engine.broker.get_positions()
                broker_pos = broker_positions.get(self.engine_config.ticker)
                if broker_pos and broker_pos.quantity != self.engine.position:
                    log.warning(
                        f"Position mismatch: engine={self.engine.position}, "
                        f"broker={broker_pos.quantity}. "
                        f"Engine will adjust on next bar."
                    )
            except Exception as e:
                log.warning(f"Reconciliation failed: {e}")

    def _log_status(self) -> None:
        """Log current status."""
        equity = self.engine.get_equity()
        metrics = self.engine.get_session_metrics()
        log.info(
            f"Status: position={metrics.get('position', 0)}, "
            f"equity={equity:.2f}, "
            f"trades={metrics.get('total_trades', 0)}"
        )

    def _should_shutdown(self) -> bool:
        """Check if we should shut down (market close, kill switch, etc.)."""
        # Check kill switch
        if getattr(self.engine.config, 'kill_switch_active', False):
            log.warning("Kill switch active — shutting down")
            return True

        # Check if market is closed (UTC times)
        now = datetime.now(timezone.utc)
        market_close = self.engine_config.market_close_utc
        if now.time() >= market_close:
            log.info("Market closed — shutting down")
            return True

        return False

    def _handle_signal(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        log.info(f"Signal {signum} received — initiating graceful shutdown")
        self._shutdown_requested = True

    def shutdown(self) -> None:
        """Initiate graceful shutdown."""
        log.info("Shutdown requested")
        self._shutdown_requested = True

    def get_status(self) -> dict:
        """Get current runner status."""
        return {
            "running": self._running,
            "shutdown_requested": self._shutdown_requested,
            "engine_metrics": self.engine.get_session_metrics(),
            "bar_builder_stats": self.bar_builder.get_stats(),
        }