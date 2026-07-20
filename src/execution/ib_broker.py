"""Interactive Brokers broker via ib_insync.

This module requires the `ib` optional dependency:
    uv sync --extra ib

Usage:
    config = IBConfig(
        tws_host="127.0.0.1",
        tws_port=4002,  # paper trading
        client_id=1,
        paper_trading=True,
    )
    broker = IBBroker(config)
    broker.connect()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .broker import Broker
from .models import Fill, FillStatus, Order, OrderStatus, OrderType, Position, Side

logger = logging.getLogger(__name__)


@dataclass
class IBConfig:
    """Interactive Brokers connection configuration."""

    tws_host: str = "127.0.0.1"
    tws_port: int = 4002  # paper trading default
    client_id: int = 1
    paper_trading: bool = True
    connect_timeout: float = 30.0
    account: str = ""  # empty = auto-detect


class IBBroker(Broker):
    """Interactive Brokers broker via ib_insync.

    Requires TWS or IB Gateway running and configured for API access.

    Args:
        config: IB connection configuration.
    """

    def __init__(self, config: IBConfig | None = None):
        self.config = config or IBConfig()
        self._ib = None
        self._connected: bool = False
        self._positions_cache: dict[str, Position] = {}
        self._orders: dict[str, Order] = {}
        self._fills: list[Fill] = []

    def _get_ib(self):
        """Lazy import ib_insync."""
        if self._ib is None:
            try:
                from ib_insync import IB
            except ImportError:
                raise ImportError(
                    "ib_insync is required for IBBroker. "
                    "Install with: uv sync --extra ib"
                ) from None
            self._ib = IB()
        return self._ib

    def connect(self) -> None:
        """Connect to TWS/Gateway."""
        ib = self._get_ib()
        try:
            ib.connect(
                self.config.tws_host,
                self.config.tws_port,
                clientId=self.config.client_id,
                timeout=self.config.connect_timeout,
            )
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            raise ConnectionError(
                f"Cannot connect to IB at {self.config.tws_host}:{self.config.tws_port}"
            ) from e

        # Verify connection
        accounts = ib.accounts()
        if not accounts:
            raise ConnectionError("No IB accounts available")

        if not self.config.account or self.config.account not in accounts:
            self.config.account = accounts[0]

        self._connected = True
        logger.info(
            f"Connected to IB (paper={self.config.paper_trading}), "
            f"account={self.config.account}"
        )

    def disconnect(self) -> None:
        """Disconnect from TWS/Gateway."""
        if self._ib:
            try:
                self._ib.disconnect()
            except Exception:
                pass
            self._ib = None
        self._connected = False
        logger.info("Disconnected from IB")

    def is_connected(self) -> bool:
        if self._ib is None:
            return False
        return self._connected and self._ib.isConnected()

    def submit_order(self, order: Order) -> str:
        if not self.is_connected():
            raise RuntimeError("IBBroker not connected")

        contract = self._create_contract(order.ticker)
        ib_order = self._create_ib_order(order)

        ib = self._get_ib()
        try:
            ib_order_id = ib.placeOrder(contract, ib_order)
            self._orders[order.order_id] = order
            logger.info(
                f"Submitted order {order.order_id}: "
                f"{order.side.value} {order.quantity} {order.ticker}"
            )
            return str(ib_order_id)
        except Exception as e:
            logger.error(f"Failed to submit order {order.order_id}: {e}")
            raise

    def cancel_order(self, order_id: str) -> bool:
        if not self.is_connected():
            return False

        order = self._orders.get(order_id)
        if not order or order.is_complete:
            return False

        try:
            self._orders[order_id] = Order(
                ticker=order.ticker,
                side=order.side,
                order_type=order.order_type,
                quantity=order.quantity,
                limit_price=order.limit_price,
                status=OrderStatus.CANCELLED,
                order_id=order_id,
                created_at=order.created_at,
                signal_strength=order.signal_strength,
                target_weight=order.target_weight,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_positions(self) -> dict[str, Position]:
        if not self.is_connected():
            return dict(self._positions_cache)

        ib = self._get_ib()
        try:
            pos_list = ib.positions()
            positions = {}
            for pos in pos_list:
                ticker = pos.contract.symbol
                avg_cost = pos.avgCost
                current_price = self.get_current_price(ticker) or avg_cost
                positions[ticker] = Position(
                    ticker=ticker,
                    quantity=pos.position,
                    avg_cost=avg_cost,
                    current_price=current_price,
                )
            self._positions_cache = positions
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")

        return dict(self._positions_cache)

    def get_portfolio_value(self) -> float:
        if not self.is_connected():
            return sum(pos.market_value for pos in self._positions_cache.values())

        ib = self._get_ib()
        try:
            accounts = ib.accountValues()
            net_liquid = next(
                (v for v in accounts if v.tag == "NetLiquidValue"), None
            )
            if net_liquid:
                return float(net_liquid.value)
        except Exception as e:
            logger.error(f"Failed to get portfolio value: {e}")

        # Fallback
        return sum(pos.market_value for pos in self.get_positions().values())

    def get_current_price(self, ticker: str) -> Optional[float]:
        if not self.is_connected():
            return None

        ib = self._get_ib()
        contract = self._create_contract(ticker)
        try:
            tick = ib.reqTickers(contract)
            if tick:
                price = tick[0].close
                if price:
                    return float(price)
        except Exception as e:
            logger.error(f"Failed to get price for {ticker}: {e}")
        return None

    def _create_contract(self, ticker: str):
        """Create an IB Stock contract for a ticker."""
        from ib_insync import Stock
        return Stock(ticker, "SMART", "USD")

    def _create_ib_order(self, order: Order):
        """Create an ib_insync order from our Order model."""
        from ib_insync import LimitOrder, MarketOrder

        action = "BUY" if order.side == Side.BUY else "SELL"

        if order.order_type == OrderType.MARKET:
            return MarketOrder(action, order.quantity)
        elif order.order_type == OrderType.LIMIT and order.limit_price:
            return LimitOrder(action, order.quantity, order.limit_price)
        else:
            return MarketOrder(action, order.quantity)

    def place_market_order(
        self,
        ticker: str,
        side: Side,
        quantity: int,
        signal_strength: float = 0.0,
        target_weight: float = 0.0,
    ) -> Order:
        order = Order(
            ticker=ticker,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            signal_strength=signal_strength,
            target_weight=target_weight,
        )
        self.submit_order(order)
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
        order = Order(
            ticker=ticker,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            limit_price=limit_price,
            signal_strength=signal_strength,
            target_weight=target_weight,
        )
        self.submit_order(order)
        return order