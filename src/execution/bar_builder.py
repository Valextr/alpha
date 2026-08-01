"""Real-time bar builder for intraday trading.

Aggregates tick-level data from IBKR into fixed-interval bars
(1-minute, 5-minute, etc.) for the intraday engine to process.

Architecture:
    IBKR ticks → BarBuilder → complete bars → IntradayEngine

Usage:
    builder = BarBuilder(ticker="MSFT", interval_minutes=1)
    builder.start()

    # Feed ticks as they arrive
    for tick in ibkr_stream:
        builder.update(tick)

    # Get completed bars
    for bar in builder.completed:
        engine.process_bar(bar)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class Bar:
    """A single OHLCV bar with optional signal."""
    ticker: str
    datetime: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    # Optional signal (filled by downstream feature computation)
    signal_vwap_mean_reversion_filtered: float = 0.0


@dataclass
class Tick:
    """A single price tick."""
    price: float
    volume: int = 1
    timestamp: Optional[datetime] = None


class BarBuilder:
    """Aggregates ticks into fixed-interval bars.

    Args:
        ticker: Ticker symbol.
        interval_minutes: Bar interval in minutes (1, 5, 15, 30, 60).
    """

    def __init__(
        self,
        ticker: str,
        interval_minutes: int = 1,
    ):
        self.ticker = ticker
        self.interval_minutes = interval_minutes
        self.interval = timedelta(minutes=interval_minutes)

        # Current bar being built
        self._current_bar: Optional[Bar] = None
        self._bar_start: Optional[datetime] = None
        self._volume: int = 0

        # Completed bars queue
        self.completed: list[Bar] = []

        # Stats
        self._ticks_processed: int = 0
        self._bars_completed: int = 0

    @property
    def has_open_bar(self) -> bool:
        """Whether a bar is currently being accumulated."""
        return self._current_bar is not None

    @property
    def current_bar(self) -> Optional[Bar]:
        """The bar currently being built (partial)."""
        return self._current_bar

    def update(self, tick: Tick) -> Optional[Bar]:
        """Feed a tick and return a completed bar if the interval elapsed.

        Args:
            tick: Price tick.

        Returns:
            Completed Bar if the interval just finished, None otherwise.
        """
        now = tick.timestamp or datetime.now(timezone.utc)

        # Align to interval boundary
        if self._bar_start is None:
            # Start a new bar aligned to the interval
            self._bar_start = self._align_to_interval(now)
            self._current_bar = Bar(
                ticker=self.ticker,
                datetime=self._bar_start,
                open=tick.price,
                high=tick.price,
                low=tick.price,
                close=tick.price,
            )
            self._volume = tick.volume
            self._ticks_processed += 1
            return None

        # Check if we've moved to the next interval
        next_bar_start = self._bar_start + self.interval
        if now >= next_bar_start:
            # Finalize current bar — use the bar's last close, NOT the new tick
            if self._current_bar:
                self._current_bar.volume = self._volume
                completed = self._current_bar
                self.completed.append(completed)
                self._bars_completed += 1

                # Start new bar with the new tick
                self._bar_start = self._align_to_interval(now)
                self._current_bar = Bar(
                    ticker=self.ticker,
                    datetime=self._bar_start,
                    open=tick.price,
                    high=tick.price,
                    low=tick.price,
                    close=tick.price,
                )
                self._volume = tick.volume
                self._ticks_processed += 1
                return completed

        # Update current bar
        if self._current_bar:
            self._current_bar.high = max(self._current_bar.high, tick.price)
            self._current_bar.low = min(self._current_bar.low, tick.price)
            self._current_bar.close = tick.price
            self._volume += tick.volume
            self._ticks_processed += 1

        return None

    def force_close(self) -> Optional[Bar]:
        """Force-close the current bar even if the interval hasn't elapsed.

        Useful at end of session or when switching tickers.

        Returns:
            The completed bar, or None if no bar was open.
        """
        if self._current_bar is None:
            return None

        completed = self._current_bar
        self.completed.append(completed)
        self._bars_completed += 1

        self._current_bar = None
        self._bar_start = None
        self._volume = 0

        return completed

    def reset(self) -> None:
        """Reset the builder (new session)."""
        self._current_bar = None
        self._bar_start = None
        self._volume = 0
        self.completed.clear()
        self._ticks_processed = 0
        self._bars_completed = 0

    def _align_to_interval(self, dt: datetime) -> datetime:
        """Align a datetime to the nearest interval boundary.

        For 1-minute bars: floor to the minute.
        For 5-minute bars: floor to the nearest 5-minute mark.
        """
        minute = dt.minute
        aligned_minute = (minute // self.interval_minutes) * self.interval_minutes
        return dt.replace(minute=aligned_minute, second=0, microsecond=0)

    def get_stats(self) -> dict:
        """Get builder statistics."""
        return {
            "ticks_processed": self._ticks_processed,
            "bars_completed": self._bars_completed,
            "has_open_bar": self.has_open_bar,
        }