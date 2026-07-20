"""Broker abstraction and implementations.

Broker ABC:
- normalize_signals() — convert signal DataFrame into portfolio targets
- generate_orders() — convert portfolio targets into executable orders
- execute_orders() — submit orders and return fills

PaperBroker:
- Simulated broker for backtesting and paper trading
- Supports deterministic and stochastic fill simulation

IBBroker:
- Interactive Brokers integration via ib_insync
- Paper and live trading modes
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Protocol

import polars as pl

from .models import (
    Fill,
    FillStatus,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Side,
)


class Broker(ABC):
    """Abstract broker interface."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to broker."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close broker connection."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the broker is connected."""

    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """Submit an order and return the broker order ID."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""

    @abstractmethod
    def get_positions(self) -> dict[str, Position]:
        """Get current positions."""

    @abstractmethod
    def get_portfolio_value(self) -> float:
        """Get total portfolio value (cash + positions)."""

    @abstractmethod
    def get_current_price(self, ticker: str) -> Optional[float]:
        """Get current market price for a ticker."""

    @abstractmethod
    def place_market_order(
        self,
        ticker: str,
        side: Side,
        quantity: int,
        signal_strength: float = 0.0,
        target_weight: float = 0.0,
    ) -> Order:
        """Place a market order."""

    @abstractmethod
    def place_limit_order(
        self,
        ticker: str,
        side: Side,
        quantity: int,
        limit_price: float,
        signal_strength: float = 0.0,
        target_weight: float = 0.0,
    ) -> Order:
        """Place a limit order."""


@dataclass
class PaperBroker(Broker):
    """Simulated broker for paper trading.

    Uses historical data or simulated prices for fill generation.
    Supports slippage and commission simulation.

    Args:
        initial_cash: Starting cash balance.
        commission_per_share: Commission per share traded.
        slippage_bps: Slippage in basis points (e.g., 10 = 0.1%).
        price_source: Optional price data (ticker -> DataFrame).
    """

    initial_cash: float = 100_000.0
    commission_per_share: float = 0.005
    slippage_bps: float = 10.0
    # Price source: ticker -> pl.DataFrame with (date, close) columns
    price_source: Optional[dict[str, pl.DataFrame]] = None

    _connected: bool = field(default=False, init=False)
    _positions: dict[str, Position] = field(default_factory=dict, init=False)
    _cash: float = field(default=0.0, init=False)
    _orders: dict[str, Order] = field(default_factory=dict, init=False)
    _fills: list[Fill] = field(default_factory=list, init=False)
    _daily_pnl: list[float] = field(default_factory=list, init=False)
    _prev_eod_equity: float = field(default=0.0, init=False)

    def __post_init__(self):
        self._cash = self.initial_cash
        self._prev_eod_equity = self.initial_cash

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def _get_current_price(self, ticker: str, ref_date: Optional[datetime] = None) -> Optional[float]:
        if self.price_source and ticker in self.price_source:
            df = self.price_source[ticker]
            if ref_date:
                row = df.filter(pl.col("date") == ref_date.date())
                if len(row):
                    return float(row["close"].item())
            else:
                if len(df):
                    return float(df["close"].slice(len(df) - 1).item())
        # Default price if no source available
        return None

    def get_current_price(self, ticker: str) -> Optional[float]:
        return self._get_current_price(ticker)

    def _apply_slippage(self, price: float, side: Side) -> float:
        """Apply slippage to fill price."""
        multiplier = 1 + (self.slippage_bps / 10000) if side == Side.BUY else 1 - (self.slippage_bps / 10000)
        return price * multiplier

    def _compute_commission(self, quantity: int) -> float:
        return quantity * self.commission_per_share

    def submit_order(self, order: Order) -> str:
        if not self._connected:
            raise RuntimeError("PaperBroker not connected")

        self._orders[order.order_id] = order
        return order.order_id

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if not order or order.is_complete:
            return False
        # Update the order in place (frozen dataclass, so recreate)
        cancelled = Order(
            ticker=order.ticker,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            limit_price=order.limit_price,
            status=OrderStatus.CANCELLED,
            order_id=order.order_id,
            created_at=order.created_at,
            signal_strength=order.signal_strength,
            target_weight=order.target_weight,
        )
        self._orders[order_id] = cancelled
        return True

    def get_positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def get_portfolio_value(self) -> float:
        positions_value = sum(pos.market_value for pos in self._positions.values())
        return self._cash + positions_value

    def place_market_order(
        self,
        ticker: str,
        side: Side,
        quantity: int,
        signal_strength: float = 0.0,
        target_weight: float = 0.0,
    ) -> Order:
        if not self._connected:
            raise RuntimeError("PaperBroker not connected")

        price = self._get_current_price(ticker)
        if price is None:
            raise ValueError(f"No price available for {ticker}")

        fill_price = self._apply_slippage(price, side)
        commission = self._compute_commission(quantity)

        # Create order
        order = Order(
            ticker=ticker,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            status=OrderStatus.FILLED,
            filled_qty=quantity,
            avg_fill_price=fill_price,
            signal_strength=signal_strength,
            target_weight=target_weight,
        )
        self._orders[order.order_id] = order

        # Process fill
        commission_cost = self._compute_commission(quantity)
        total_cost = fill_price * quantity + commission_cost

        if side == Side.BUY:
            self._cash -= total_cost
            # Update or create position
            if ticker in self._positions:
                pos = self._positions[ticker]
                total_shares = pos.quantity + quantity
                new_avg = (pos.avg_cost * pos.quantity + fill_price * quantity) / total_shares
                self._positions[ticker] = Position(
                    ticker=ticker,
                    quantity=total_shares,
                    avg_cost=new_avg,
                    current_price=fill_price,
                    realized_pnl=pos.realized_pnl,
                    commissions=pos.commissions + commission_cost,
                )
            else:
                self._positions[ticker] = Position(
                    ticker=ticker,
                    quantity=quantity,
                    avg_cost=fill_price,
                    current_price=fill_price,
                    commissions=commission_cost,
                )
        else:
            # SELL
            proceeds = fill_price * quantity - commission_cost
            self._cash += proceeds
            if ticker in self._positions:
                pos = self._positions[ticker]
                remaining = pos.quantity - quantity
                if remaining <= 0:
                    # Full close — calculate realized P&L
                    realized = (fill_price - pos.avg_cost) * pos.quantity
                    del self._positions[ticker]
                else:
                    realized = (fill_price - pos.avg_cost) * quantity
                    self._positions[ticker] = Position(
                        ticker=ticker,
                        quantity=remaining,
                        avg_cost=pos.avg_cost,
                        current_price=fill_price,
                        realized_pnl=pos.realized_pnl + realized,
                        commissions=pos.commissions + commission_cost,
                    )

        # Record fill
        fill = Fill(
            order_id=order.order_id,
            ticker=ticker,
            side=side,
            quantity=quantity,
            price=fill_price,
            commission=commission_cost,
            fill_status=FillStatus.COMPLETE,
        )
        self._fills.append(fill)

        return order

    def place_limit_order(
        self,
        ticker: str,
        side: Side,
        quantity: int,
        limit_price: float,
        signal_strength: float = 0.0,
        target_weight: float = 0.0,
    ) -> Order:
        if not self._connected:
            raise RuntimeError("PaperBroker not connected")

        price = self._get_current_price(ticker)
        if price is None:
            raise ValueError(f"No price available for {ticker}")

        # Limit order fills if price is favorable
        fills = False
        if side == Side.BUY and price <= limit_price:
            fills = True
        elif side == Side.SELL and price >= limit_price:
            fills = True

        if not fills:
            order = Order(
                ticker=ticker,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                limit_price=limit_price,
                status=OrderStatus.PENDING,
                signal_strength=signal_strength,
                target_weight=target_weight,
            )
            self._orders[order.order_id] = order
            return order

        # Fill at limit price (or better)
        fill_price = min(price, limit_price) if side == Side.BUY else max(price, limit_price)
        commission = self._compute_commission(quantity)

        order = Order(
            ticker=ticker,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            limit_price=limit_price,
            status=OrderStatus.FILLED,
            filled_qty=quantity,
            avg_fill_price=fill_price,
            signal_strength=signal_strength,
            target_weight=target_weight,
        )
        self._orders[order.order_id] = order

        # Update cash and positions (reuse market order logic)
        commission_cost = self._compute_commission(quantity)
        if side == Side.BUY:
            total_cost = fill_price * quantity + commission_cost
            self._cash -= total_cost
            if ticker in self._positions:
                pos = self._positions[ticker]
                total_shares = pos.quantity + quantity
                new_avg = (pos.avg_cost * pos.quantity + fill_price * quantity) / total_shares
                self._positions[ticker] = Position(
                    ticker=ticker,
                    quantity=total_shares,
                    avg_cost=new_avg,
                    current_price=fill_price,
                    realized_pnl=pos.realized_pnl,
                    commissions=pos.commissions + commission_cost,
                )
            else:
                self._positions[ticker] = Position(
                    ticker=ticker,
                    quantity=quantity,
                    avg_cost=fill_price,
                    current_price=fill_price,
                    commissions=commission_cost,
                )
        else:
            proceeds = fill_price * quantity - commission_cost
            self._cash += proceeds
            if ticker in self._positions:
                pos = self._positions[ticker]
                remaining = pos.quantity - quantity
                if remaining <= 0:
                    realized = (fill_price - pos.avg_cost) * pos.quantity
                    del self._positions[ticker]
                else:
                    realized = (fill_price - pos.avg_cost) * quantity
                    self._positions[ticker] = Position(
                        ticker=ticker,
                        quantity=remaining,
                        avg_cost=pos.avg_cost,
                        current_price=fill_price,
                        realized_pnl=pos.realized_pnl + realized,
                        commissions=pos.commissions + commission_cost,
                    )

        fill = Fill(
            order_id=order.order_id,
            ticker=ticker,
            side=side,
            quantity=quantity,
            price=fill_price,
            commission=commission_cost,
            fill_status=FillStatus.COMPLETE,
        )
        self._fills.append(fill)

        return order

    def get_fills(self) -> list[Fill]:
        return list(self._fills)

    def get_order_history(self) -> dict[str, Order]:
        return dict(self._orders)

    def get_daily_pnl(self) -> list[float]:
        return list(self._daily_pnl)

    def record_eod(self) -> float:
        """Record end-of-day P&L. Call once per trading day."""
        equity = self.get_portfolio_value()
        daily_pnl = equity - self._prev_eod_equity
        self._daily_pnl.append(daily_pnl)
        self._prev_eod_equity = equity
        return daily_pnl


# IBBroker is implemented in a separate module to avoid ib_insync
# dependency when running paper trading.