"""Risk guardrails for the execution engine.

Guardrails:
- Position limits (max positions, max concentration)
- Daily P&L circuit breaker (kill switch on drawdown)
- Max leverage
- Pre-trade validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .models import Order, Position, Side


@dataclass
class RiskConfig:
    """Risk guardrail configuration."""

    # Maximum number of concurrent positions
    max_positions: int = 20

    # Maximum portfolio weight per position (e.g., 0.10 = 10%)
    max_position_weight: float = 0.10

    # Maximum total portfolio leverage (1.0 = no leverage)
    max_leverage: float = 1.0

    # Daily P&L circuit breaker — halt trading if daily loss exceeds this %
    daily_loss_limit_pct: float = 0.05

    # Maximum drawdown from peak equity — kill switch
    max_drawdown_pct: float = 0.15

    # Minimum cash reserve (fraction of portfolio)
    min_cash_reserve: float = 0.05

    # Max order size relative to portfolio
    max_order_size_pct: float = 0.05

    # Whether the kill switch has been triggered
    _kill_switch_active: bool = field(default=False, init=False)
    # When the kill switch was triggered
    _kill_switch_time: Optional[datetime] = field(default=None, init=False)

    def activate_kill_switch(self) -> None:
        self._kill_switch_active = True
        self._kill_switch_time = datetime.now(timezone.utc)

    def deactivate_kill_switch(self) -> None:
        self._kill_switch_active = False
        self._kill_switch_time = None

    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch_active

    @property
    def kill_switch_triggered_at(self) -> Optional[datetime]:
        return self._kill_switch_time


class RiskGuard:
    """Pre-trade and intra-trade risk validation.

    Args:
        config: Risk configuration.
        portfolio_value: Current total portfolio value.
        positions: Current positions dict.
        cash: Current cash balance.
        daily_pnl: Current daily P&L so far.
        peak_equity: Peak portfolio equity for drawdown calculation.
    """

    def __init__(
        self,
        config: RiskConfig,
        *,
        portfolio_value: float = 0.0,
        positions: dict[str, Position] | None = None,
        cash: float = 0.0,
        daily_pnl: float = 0.0,
        peak_equity: float = 0.0,
    ):
        self.config = config
        self.portfolio_value = portfolio_value
        self.positions = positions or {}
        self.cash = cash
        self.daily_pnl = daily_pnl
        self.peak_equity = peak_equity

    def validate_order(self, order: Order) -> tuple[bool, str]:
        """Validate an order against all risk guardrails.

        Returns:
            (approved, reason) — if not approved, reason explains why.
        """
        if self.config.kill_switch_active:
            return False, "KILL_SWITCH_ACTIVE: Trading halted"

        checks = [
            self._check_max_positions,
            self._check_position_weight,
            self._check_order_size,
            self._check_daily_loss_limit,
            self._check_drawdown,
            self._check_cash_reserve,
            self._check_leverage,
        ]

        for check in checks:
            approved, reason = check(order)
            if not approved:
                return False, reason

        return True, "APPROVED"

    def _check_max_positions(self, order: Order) -> tuple[bool, str]:
        """Check if adding this order would exceed max positions."""
        if order.side == Side.BUY and order.ticker not in self.positions:
            if len(self.positions) >= self.config.max_positions:
                return False, f"MAX_POSITIONS: Cannot open {len(self.positions) + 1}th position (max {self.config.max_positions})"
        return True, "APPROVED"

    def _check_position_weight(self, order: Order) -> tuple[bool, str]:
        """Check if this position would exceed max weight."""
        if not self.portfolio_value or self.portfolio_value <= 0:
            return True, "APPROVED"

        # Estimate order value
        if order.side == Side.BUY:
            current_qty = self.positions.get(order.ticker, Position(ticker=order.ticker, quantity=0, avg_cost=0.0))
            # Approximate: use order quantity as estimate
            order_value = order.quantity * (order.avg_fill_price or 0)
            new_weight = order_value / self.portfolio_value if self.portfolio_value else 0
            if new_weight > self.config.max_position_weight:
                return False, f"POSITION_WEIGHT: Order would represent {new_weight:.1%} of portfolio (max {self.config.max_position_weight:.1%})"
        return True, "APPROVED"

    def _check_order_size(self, order: Order) -> tuple[bool, str]:
        """Check if order size exceeds max order size relative to portfolio."""
        if not self.portfolio_value or self.portfolio_value <= 0:
            return True, "APPROVED"

        # Use average fill price if available, otherwise can't check precisely
        if order.avg_fill_price:
            order_value = order.quantity * order.avg_fill_price
            if order_value > self.portfolio_value * self.config.max_order_size_pct:
                return False, f"ORDER_SIZE: Order value ${order_value:,.0f} exceeds {self.config.max_order_size_pct:.1%} of portfolio (max ${self.portfolio_value * self.config.max_order_size_pct:,.0f})"
        return True, "APPROVED"

    def _check_daily_loss_limit(self, order: Order) -> tuple[bool, str]:
        """Check if daily P&L has exceeded the loss limit."""
        if not self.portfolio_value or self.portfolio_value <= 0:
            return True, "APPROVED"

        daily_loss_pct = -self.daily_pnl / self.portfolio_value
        if daily_loss_pct >= self.config.daily_loss_limit_pct:
            self.config.activate_kill_switch()
            return False, f"DAILY_LOSS: Daily loss {daily_loss_pct:.1%} exceeds limit {self.config.daily_loss_limit_pct:.1%} — kill switch activated"
        return True, "APPROVED"

    def _check_drawdown(self, order: Order) -> tuple[bool, str]:
        """Check if portfolio drawdown exceeds max drawdown."""
        if self.peak_equity <= 0:
            return True, "APPROVED"

        drawdown = (self.peak_equity - self.portfolio_value) / self.peak_equity
        if drawdown >= self.config.max_drawdown_pct:
            self.config.activate_kill_switch()
            return False, f"DRAWDOWN: Portfolio drawdown {drawdown:.1%} exceeds limit {self.config.max_drawdown_pct:.1%} — kill switch activated"
        return True, "APPROVED"

    def _check_cash_reserve(self, order: Order) -> tuple[bool, str]:
        """Check if this order would violate minimum cash reserve."""
        if order.side == Side.BUY and order.avg_fill_price:
            order_cost = order.quantity * order.avg_fill_price
            remaining_cash = self.cash - order_cost
            if remaining_cash < self.portfolio_value * self.config.min_cash_reserve:
                return False, f"CASH_RESERVE: Order would leave ${remaining_cash:,.0f} below minimum reserve (${self.portfolio_value * self.config.min_cash_reserve:,.0f})"
        return True, "APPROVED"

    def _check_leverage(self, order: Order) -> tuple[bool, str]:
        """Check if this order would exceed max leverage."""
        if not self.portfolio_value or self.portfolio_value <= 0:
            return True, "APPROVED"

        current_leverage = sum(pos.market_value for pos in self.positions.values()) / self.portfolio_value
        if order.side == Side.BUY and order.avg_fill_price:
            additional_exposure = order.quantity * order.avg_fill_price
            new_leverage = (current_leverage * self.portfolio_value + additional_exposure) / self.portfolio_value
            if new_leverage > self.config.max_leverage:
                return False, f"LEVERAGE: New leverage {new_leverage:.2f}x exceeds limit {self.config.max_leverage}x"
        return True, "APPROVED"