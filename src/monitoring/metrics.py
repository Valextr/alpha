"""Metrics computation for paper trading monitoring.

Provides real-time and historical metrics: P&L, drawdown, Sharpe ratio,
win rate, and more. Built on Polars for performance, consistent with
the rest of the Alpha codebase.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

import polars as pl

from .data_models import DailySummary, Fill, Position


class MetricsTracker:
    """Track and compute trading metrics from fills and positions.

    Usage:
        tracker = MetricsTracker(initial_balance=100_000.0)
        tracker.record_fill(fill)
        tracker.update_positions(positions)
        metrics = tracker.get_metrics()
        summary = tracker.daily_summary()
    """

    def __init__(
        self,
        initial_balance: float = 100_000.0,
        annualization: int = 252,
    ):
        self.initial_balance = initial_balance
        self.annualization = annualization
        self._fills: list[Fill] = []
        self._positions: dict[str, Position] = {}
        self._balance = initial_balance
        self._peak_balance = initial_balance
        self._daily_returns: list[float] = []
        self._daily_start_balance = initial_balance
        self._num_wins = 0
        self._num_losses = 0

    @property
    def current_balance(self) -> float:
        """Total portfolio value including unrealized P&L."""
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        return self._balance + unrealized

    @property
    def realized_pnl(self) -> float:
        """Total realized P&L from closed positions."""
        return sum(p.realized_pnl for p in self._positions.values())

    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized P&L from open positions."""
        return sum(p.unrealized_pnl for p in self._positions.values())

    @property
    def num_trades(self) -> int:
        return len(self._fills)

    @property
    def current_drawdown(self) -> float:
        """Current drawdown as a fraction of peak-to-trough."""
        if self._peak_balance <= 0:
            return 0.0
        return (self._peak_balance - self.current_balance) / self._peak_balance

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown observed so far."""
        if self._peak_balance <= 0:
            return 0.0
        trough = min(self._daily_returns or [self.current_balance])
        return (self._peak_balance - trough) / self._peak_balance

    def record_fill(self, fill: Fill) -> None:
        """Record an order fill and update balance."""
        self._fills.append(fill)
        cost = fill.quantity * fill.price + fill.commission
        if fill.side.value == "BUY":
            self._balance -= cost
        else:
            self._balance += cost

        # Update position tracking
        if fill.ticker not in self._positions:
            self._positions[fill.ticker] = Position(
                ticker=fill.ticker,
                quantity=0,
                avg_cost=0.0,
                current_price=fill.price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
            )

        pos = self._positions[fill.ticker]
        if fill.side.value == "BUY":
            # Average up — avg_cost tracks price basis only (commission already
            # deducted from _balance and should not taint unrealized P&L)
            price_basis = pos.avg_cost * abs(pos.quantity) + fill.quantity * fill.price
            pos.quantity += fill.quantity
            pos.avg_cost = price_basis / abs(pos.quantity) if pos.quantity else fill.price
        else:
            # Sell: compute realized P&L for shares closed
            pos.quantity -= fill.quantity
            # Realized P&L for the shares sold
            shares_closed = min(fill.quantity, pos.quantity + fill.quantity)
            if pos.avg_cost > 0 and shares_closed > 0:
                realized = shares_closed * (fill.price - pos.avg_cost) - fill.commission
                pos.realized_pnl += realized

        pos.current_price = fill.price
        self._update_unrealized(fill.ticker)

    def update_positions(self, positions: list[Position]) -> None:
        """Update current prices for all positions (e.g., from market data)."""
        for pos in positions:
            if pos.ticker in self._positions:
                self._positions[pos.ticker].current_price = pos.current_price
            else:
                self._positions[pos.ticker] = pos
            self._update_unrealized(pos.ticker)

        # Update peak balance
        bal = self.current_balance
        if bal > self._peak_balance:
            self._peak_balance = bal

    def _update_unrealized(self, ticker: str) -> None:
        """Recalculate unrealized P&L for a position."""
        if ticker not in self._positions:
            return
        pos = self._positions[ticker]
        if pos.quantity != 0:
            pos.unrealized_pnl = pos.quantity * (pos.current_price - pos.avg_cost)

    def record_daily_return(self) -> float:
        """Mark end of trading day. Returns the daily return pct."""
        daily_return = (self.current_balance - self._daily_start_balance) / self._daily_start_balance
        self._daily_returns.append(daily_return)

        # Count wins/losses
        if daily_return > 0:
            self._num_wins += 1
        elif daily_return < 0:
            self._num_losses += 1

        # Track peak
        bal = self.current_balance
        if bal > self._peak_balance:
            self._peak_balance = bal

        # Reset daily start
        self._daily_start_balance = self.current_balance
        return daily_return

    def get_metrics(self) -> dict:
        """Get current metrics snapshot."""
        balance = self.current_balance
        total_pnl = balance - self.initial_balance
        total_return = total_pnl / self.initial_balance if self.initial_balance else 0.0

        return {
            "balance": balance,
            "total_pnl": round(total_pnl, 2),
            "total_return": round(total_return, 4),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "drawdown": round(self.current_drawdown, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "win_rate": round(self.win_rate, 4),
            "num_trades": self.num_trades,
            "num_positions": len([p for p in self._positions.values() if p.quantity != 0]),
        }

    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio from daily returns."""
        if len(self._daily_returns) < 2:
            return 0.0
        df = pl.Series(self._daily_returns)
        mean_ret = df.mean()
        std_ret = df.std()
        if std_ret == 0:
            return 0.0
        return float((mean_ret / std_ret) * math.sqrt(self.annualization))

    @property
    def win_rate(self) -> float:
        """Win rate from daily returns."""
        total = self._num_wins + self._num_losses
        if total == 0:
            return 0.0
        return self._num_wins / total

    def daily_summary(self, date_str: Optional[str] = None) -> DailySummary:
        """Generate a DailySummary for the current/last day."""
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        total_trades = self._num_wins + self._num_losses
        return DailySummary(
            date=date_str,
            starting_balance=self._daily_start_balance,
            ending_balance=self.current_balance,
            daily_pnl=round(self.current_balance - self._daily_start_balance, 2),
            daily_return=round(
                (self.current_balance - self._daily_start_balance) / self._daily_start_balance, 4
            ) if self._daily_start_balance else 0.0,
            num_trades=self.num_trades,
            num_wins=self._num_wins,
            num_losses=self._num_losses,
            win_rate=round(self.win_rate, 4),
            current_drawdown=round(self.current_drawdown, 4),
            max_drawdown=round(self.max_drawdown, 4),
            sharpe_ratio=round(self.sharpe_ratio, 3),
            positions_count=len([p for p in self._positions.values() if p.quantity != 0]),
            alerts_count=0,  # Updated by AlertManager
        )

    def to_dataframe(self) -> pl.DataFrame:
        """Export all fills as a Polars DataFrame."""
        if not self._fills:
            return pl.DataFrame(
                schema=["fill_id", "ticker", "side", "quantity", "price", "order_id", "timestamp", "commission"]
            )
        return pl.DataFrame([
            {
                "fill_id": f.fill_id,
                "ticker": f.ticker,
                "side": f.side.value,
                "quantity": f.quantity,
                "price": f.price,
                "order_id": f.order_id,
                "timestamp": f.timestamp,
                "commission": f.commission,
            }
            for f in self._fills
        ])

    def export_metrics_history(self) -> pl.DataFrame:
        """Export metrics history as a Polars DataFrame."""
        return pl.DataFrame({
            "daily_return": self._daily_returns,
        }) if self._daily_returns else pl.DataFrame(schema={"daily_return": pl.Float64})