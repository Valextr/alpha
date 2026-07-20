"""Alert engine with configurable thresholds.

Monitors metrics and fires alerts when thresholds are breached.
The alert manager is updated on every metrics refresh and can be
plugged into the dashboard for real-time notification.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable, Optional

from loguru import logger

from .config import MonitoringConfig
from .data_models import Alert, AlertSeverity, AlertType, TradeEvent


class AlertManager:
    """Monitor metrics and fire alerts when thresholds are breached.

    Usage:
        mgr = AlertManager(config)
        mgr.add_callback(send_notification)
        mgr.check_drawdown(0.08)   # fires if > config.max_drawdown
        mgr.check_daily_loss(0.03)  # fires if > config.max_daily_loss

    Rate limiting:
        Each alert type has a cooldown period (default 5 minutes) to
        prevent alert storms during volatile periods.
    """

    COOLDOWN_SECONDS = 300  # 5 minutes between same-type alerts

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.alerts: list[Alert] = []
        self._callbacks: list[Callable[[Alert], None]] = []
        self._last_fired: dict[str, float] = {}

    def add_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add a callback to be invoked when an alert fires."""
        self._callbacks.append(callback)

    def _fire(self, alert: Alert) -> None:
        """Fire an alert after cooldown check."""
        # Rate limiting
        last_fired = self._last_fired.get(alert.alert_type.value, 0.0)
        if time.time() - last_fired < self.COOLDOWN_SECONDS:
            logger.debug(f"Alert rate-limited: {alert.alert_type.value}")
            return

        self.alerts.append(alert)
        self._last_fired[alert.alert_type.value] = time.time()
        logger.warning(
            f"ALERT [{alert.severity.value}] {alert.alert_type.value}: "
            f"{alert.message}"
        )

        # Notify callbacks
        for cb in self._callbacks:
            try:
                cb(alert)
            except Exception:
                logger.exception(f"Alert callback failed for {alert.alert_type.value}")

    def check_drawdown(self, drawdown: float) -> Optional[Alert]:
        """Check drawdown against thresholds."""
        thresholds = self.config.alerts

        if drawdown >= thresholds.kill_switch_drawdown:
            alert = Alert(
                alert_type=AlertType.KILL_SWITCH,
                severity=AlertSeverity.CRITICAL,
                message=f"Kill switch triggered: drawdown {drawdown:.1%} exceeds {thresholds.kill_switch_drawdown:.1%}",
                value=drawdown,
                threshold=thresholds.kill_switch_drawdown,
            )
            self._fire(alert)
            return alert

        if drawdown >= thresholds.max_drawdown:
            alert = Alert(
                alert_type=AlertType.DRAWDOWN_BREACH,
                severity=AlertSeverity.WARNING,
                message=f"Drawdown {drawdown:.1%} exceeds threshold {thresholds.max_drawdown:.1%}",
                value=drawdown,
                threshold=thresholds.max_drawdown,
            )
            self._fire(alert)
            return alert

        return None

    def check_daily_loss(self, daily_loss_pct: float) -> Optional[Alert]:
        """Check daily loss percentage against threshold."""
        threshold = self.config.alerts.max_daily_loss
        if daily_loss_pct <= -threshold:
            alert = Alert(
                alert_type=AlertType.DAILY_LOSS_LIMIT,
                severity=AlertSeverity.WARNING,
                message=f"Daily loss {daily_loss_pct:.1%} exceeds threshold {threshold:.1%}",
                value=daily_loss_pct,
                threshold=-threshold,
            )
            self._fire(alert)
            return alert
        return None

    def check_position_limit(
        self, position_pct: float, ticker: str
    ) -> Optional[Alert]:
        """Check position size against portfolio percentage limit."""
        threshold = self.config.alerts.max_position_pct
        if position_pct > threshold:
            alert = Alert(
                alert_type=AlertType.POSITION_LIMIT_BREACH,
                severity=AlertSeverity.WARNING,
                message=f"Position {ticker} at {position_pct:.1%} exceeds limit {threshold:.1%}",
                ticker=ticker,
                value=position_pct,
                threshold=threshold,
            )
            self._fire(alert)
            return alert
        return None

    def check_leverage(self, leverage: float) -> Optional[Alert]:
        """Check leverage against maximum."""
        threshold = self.config.alerts.max_leverage
        if leverage > threshold:
            alert = Alert(
                alert_type=AlertType.LEVERAGE_EXCEEDED,
                severity=AlertSeverity.CRITICAL,
                message=f"Leverage {leverage:.2f}x exceeds max {threshold:.2f}x",
                value=leverage,
                threshold=threshold,
            )
            self._fire(alert)
            return alert
        return None

    def check_fill_slippage(
        self, slippage_pct: float, ticker: str
    ) -> Optional[Alert]:
        """Check fill slippage against threshold."""
        threshold = self.config.alerts.max_fill_slippage_pct
        if slippage_pct > threshold:
            alert = Alert(
                alert_type=AlertType.FILL_MISMATCH,
                severity=AlertSeverity.INFO,
                message=f"Slippage on {ticker}: {slippage_pct:.2f}% (threshold: {threshold}%)",
                ticker=ticker,
                value=slippage_pct,
                threshold=threshold,
            )
            self._fire(alert)
            return alert
        return None

    def process_trade_event(self, event: TradeEvent) -> list[Alert]:
        """Process a trade event and return any new alerts."""
        new_alerts = []

        if event.event_type == "FILL" and event.fill:
            # Could check slippage here if expected price was in metadata
            pass

        elif event.event_type == "ERROR":
            alert = Alert(
                alert_type=AlertType.CONNECTION_ERROR,
                severity=AlertSeverity.WARNING,
                message=event.error or "Unknown error from execution engine",
            )
            self._fire(alert)
            new_alerts.append(alert)

        return new_alerts

    def acknowledge(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def get_unacknowledged(self) -> list[Alert]:
        """Get all unacknowledged alerts."""
        return [a for a in self.alerts if not a.acknowledged]

    def get_recent(self, minutes: int = 60) -> list[Alert]:
        """Get alerts from the last N minutes."""
        cutoff = time.time() - minutes * 60
        return [
            a for a in self.alerts
            if (a.timestamp.timestamp() if hasattr(a.timestamp, "timestamp") else 0) > cutoff
        ]

    @property
    def active_count(self) -> int:
        """Number of unacknowledged alerts."""
        return len(self.get_unacknowledged())