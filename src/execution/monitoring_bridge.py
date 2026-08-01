"""Bridge between execution engine and monitoring layer.

Converts execution models (Order, Position, Fill) into monitoring
models (TradeEvent, Fill, Position) and feeds them to the metrics
tracker and alert manager.

Usage:
    from src.execution.monitoring_bridge import MonitoringBridge

    bridge = MonitoringBridge(initial_balance=100_000.0)
    engine = ExecutionEngine(config, broker=broker, monitoring_bridge=bridge)

    # After each execution step:
    engine.update_monitoring()

    # Access monitoring:
    metrics = bridge.get_metrics()
    alerts = bridge.get_alerts()
    summary = bridge.record_end_of_day()
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from .models import Fill as ExecFill, Order, Position as ExecPosition, Side

if TYPE_CHECKING:
    from src.monitoring import (
        Alert,
        AlertManager,
        DailySummary,
        Fill as MonFill,
        MetricsTracker,
        MonitoringConfig,
        Position as MonPosition,
        TradeEvent,
    )

logger = logging.getLogger(__name__)


def exec_position_to_monitoring(pos: ExecPosition) -> dict:
    """Convert an execution Position to monitoring Position fields."""
    return {
        "ticker": pos.ticker,
        "quantity": pos.quantity,
        "avg_cost": pos.avg_cost,
        "current_price": pos.current_price,
        "unrealized_pnl": pos.unrealized_pnl,
        "realized_pnl": pos.realized_pnl,
    }


def exec_fill_to_monitoring(fill: ExecFill) -> dict:
    """Convert an execution Fill to monitoring Fill fields."""
    from src.monitoring import OrderSide

    side = OrderSide.BUY if fill.side == Side.BUY else OrderSide.SELL
    return {
        "ticker": fill.ticker,
        "side": side,
        "quantity": fill.quantity,
        "price": fill.price,
        "order_id": fill.order_id,
        "timestamp": fill.timestamp,
        "commission": fill.commission,
    }


class MonitoringBridge:
    """Bridge between execution engine and monitoring layer.

    Converts execution models to monitoring models and feeds them
    to the metrics tracker and alert manager.

    Args:
        initial_balance: Starting portfolio balance for metrics.
        monitoring_config: Optional monitoring configuration.
    """

    def __init__(
        self,
        initial_balance: float = 100_000.0,
        monitoring_config: Optional["MonitoringConfig"] = None,
    ):
        from src.monitoring import (
            AlertManager,
            MetricsTracker,
            MonitoringConfig,
        )

        self.config = monitoring_config or MonitoringConfig()
        self.metrics = MetricsTracker(initial_balance=initial_balance)
        self.alerts = AlertManager(self.config)

        # Track which fills we've already processed
        self._processed_fills: set[str] = set()

    def update_from_engine(
        self,
        positions: dict[str, ExecPosition],
        fills: list[ExecFill],
        portfolio_value: float,
        daily_pnl: float = 0.0,
        peak_equity: float = 0.0,
    ) -> list["Alert"]:
        """Update monitoring from execution engine state.

        Args:
            positions: Current positions from the engine.
            fills: Recent fills from the engine.
            portfolio_value: Total portfolio value.
            daily_pnl: Daily P&L (if available).
            peak_equity: Peak equity value (for drawdown calc).

        Returns:
            List of new alerts fired during this update.
        """
        from src.monitoring import (
            Fill as MonFill,
            Position as MonPosition,
            TradeEvent,
        )

        new_alerts = []

        # Update positions in metrics tracker
        mon_positions = [
            MonPosition(**exec_position_to_monitoring(pos))
            for pos in positions.values()
        ]
        self.metrics.update_positions(mon_positions)

        # Process new fills
        for fill in fills:
            if fill.fill_id not in self._processed_fills:
                self._processed_fills.add(fill.fill_id)
                mon_fill = MonFill(**exec_fill_to_monitoring(fill))
                self.metrics.record_fill(mon_fill)

                # Create a trade event
                event = TradeEvent(
                    event_type="FILL",
                    ticker=fill.ticker,
                    fill=mon_fill,
                )
                event_alerts = self.alerts.process_trade_event(event)
                new_alerts.extend(event_alerts)

        # Check drawdown
        if peak_equity > 0:
            drawdown = (peak_equity - portfolio_value) / peak_equity
            alert = self.alerts.check_drawdown(drawdown)
            if alert:
                new_alerts.append(alert)

        # Check daily loss
        if daily_pnl < 0 and portfolio_value > 0:
            daily_loss_pct = daily_pnl / portfolio_value
            alert = self.alerts.check_daily_loss(daily_loss_pct)
            if alert:
                new_alerts.append(alert)

        # Check position limits
        for ticker, pos in positions.items():
            if portfolio_value > 0:
                position_pct = abs(pos.market_value) / portfolio_value
                alert = self.alerts.check_position_limit(position_pct, ticker)
                if alert:
                    new_alerts.append(alert)

        return new_alerts

    def record_end_of_day(self) -> "DailySummary":
        """Mark end of trading day and generate summary."""
        daily_return = self.metrics.record_daily_return()
        summary = self.metrics.daily_summary()
        return summary

    def get_metrics(self) -> dict:
        """Get current metrics snapshot."""
        return self.metrics.get_metrics()

    def get_alerts(self) -> list["Alert"]:
        """Get all alerts."""
        return self.alerts.alerts

    def get_unacknowledged_alerts(self) -> list["Alert"]:
        """Get unacknowledged alerts."""
        return self.alerts.get_unacknowledged()

    def get_recent_alerts(self, minutes: int = 60) -> list["Alert"]:
        """Get alerts from the last N minutes."""
        return self.alerts.get_recent(minutes)

    @property
    def active_alert_count(self) -> int:
        """Number of unacknowledged alerts."""
        return self.alerts.active_count