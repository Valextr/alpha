"""Pydantic data models for monitoring, execution, and alerts.

These models define the contract between the execution engine and
the monitoring layer. The execution engine pushes events; monitoring
consumes them and produces metrics and alerts.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    BRACKET = "BRACKET"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertType(str, Enum):
    # Risk alerts
    DRAWDOWN_BREACH = "DRAWDOWN_BREACH"
    POSITION_LIMIT_BREACH = "POSITION_LIMIT_BREACH"
    LEVERAGE_EXCEEDED = "LEVERAGE_EXCEEDED"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    KILL_SWITCH = "KILL_SWITCH"

    # Operational alerts
    CONNECTION_ERROR = "CONNECTION_ERROR"
    FILL_MISMATCH = "FILL_MISMATCH"
    RECONCILIATION_ERROR = "RECONCILIATION_ERROR"

    # Performance alerts
    ANOMALOUS_PERF = "ANOMALOUS_PERF"
    SIGNAL_DEGRADATION = "SIGNAL_DEGRADATION"


class Position(BaseModel):
    """Current position for a single ticker."""

    ticker: str
    quantity: int
    avg_cost: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    model_config = {"extra": "forbid"}

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def notional(self) -> float:
        return abs(self.market_value)


class Fill(BaseModel):
    """Record of an order fill."""

    fill_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    ticker: str
    side: OrderSide
    quantity: int
    price: float
    order_id: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    commission: float = 0.0

    model_config = {"extra": "forbid"}


class TradeEvent(BaseModel):
    """Event pushed by the execution engine to monitoring."""

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    event_type: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Optional fields depending on event_type
    ticker: Optional[str] = None
    fill: Optional[Fill] = None
    position: Optional[Position] = None
    error: Optional[str] = None
    metadata: dict = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class Alert(BaseModel):
    """Alert fired by the alert manager."""

    alert_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    ticker: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None
    acknowledged: bool = False

    model_config = {"extra": "allow"}


class DailySummary(BaseModel):
    """End-of-day summary report."""

    date: str
    starting_balance: float
    ending_balance: float
    daily_pnl: float
    daily_return: float
    num_trades: int
    num_wins: int
    num_losses: int
    win_rate: float
    current_drawdown: float
    max_drawdown: float
    sharpe_ratio: float
    positions_count: int
    alerts_count: int

    model_config = {"extra": "allow"}