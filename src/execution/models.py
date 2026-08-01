"""Execution data models.

Core types:
- Order: a request to buy/sell an asset
- Position: current exposure to a ticker
- Fill: execution of an order (partial or full)
- PortfolioSnapshot: point-in-time portfolio state
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    BRACKET = "BRACKET"


class OrderStatus(Enum):
    PENDING = "PENDING"       # submitted, not yet filled
    FILLED = "FILLED"         # fully filled
    PARTIAL = "PARTIAL"       # partially filled
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class FillStatus(Enum):
    PARTIAL = "PARTIAL"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True)
class Order:
    """An order request."""

    ticker: str
    side: Side
    order_type: OrderType
    quantity: int
    limit_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    filled_qty: int = 0
    avg_fill_price: float = 0.0
    # optional: signal strength that generated this order
    signal_strength: float = 0.0
    # optional: portfolio target weight
    target_weight: float = 0.0

    @property
    def remaining(self) -> int:
        return self.quantity - self.filled_qty

    @property
    def is_complete(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED)


@dataclass(frozen=True)
class Position:
    """Current exposure to a ticker."""

    ticker: str
    quantity: int
    avg_cost: float
    current_price: float = 0.0
    # unrealized P&L
    unrealized_pnl: float = 0.0
    # realized P&L (from closed positions)
    realized_pnl: float = 0.0
    # total commissions paid
    commissions: float = 0.0
    # timestamp of last update
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_cost

    @property
    def equity(self) -> float:
        return self.market_value - self.commissions

    @property
    def return_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return (self.market_value - self.cost_basis) / self.cost_basis


@dataclass(frozen=True)
class Fill:
    """Execution of an order (partial or full)."""

    fill_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    order_id: str = ""
    ticker: str = ""
    side: Side = Side.BUY
    quantity: int = 0
    price: float = 0.0
    commission: float = 0.0
    fill_status: FillStatus = FillStatus.PARTIAL
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Point-in-time portfolio state."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    positions: dict[str, Position] = field(default_factory=dict)
    cash: float = 0.0
    total_equity: float = 0.0
    daily_pnl: float = 0.0
    drawdown: float = 0.0
    # metadata
    num_positions: int = 0
    num_orders_today: int = 0
    num_trades_today: int = 0