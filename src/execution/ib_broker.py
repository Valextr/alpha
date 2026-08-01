"""Interactive Brokers broker via IBKRClient (ib_async v2.x).

This module wraps IBKRClient to implement the Broker ABC, making it
compatible with the ExecutionEngine. The old ib_insync-based IBBroker
was removed because ib_insync doesn't support Python 3.14.

Usage:
    from src.execution.ib_broker import IBBroker

    broker = IBBroker()  # loads config from .env
    engine = ExecutionEngine(config, broker=broker)
    engine.start()

Requires the `ib` optional dependency:
    uv sync --extra ib
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .broker import Broker
from .ibkr_client import IBKRClient, IBKRConfig
from .models import Fill, FillStatus, Order, OrderStatus, OrderType, Position, Side

logger = logging.getLogger(__name__)


@dataclass
class IBConfig:
    """Interactive Brokers connection configuration.

    Thin wrapper around IBKRConfig for backward compatibility.
    """

    tws_host: str = "127.0.0.1"
    tws_port: int = 4002  # paper trading default
    client_id: int = 1
    paper_trading: bool = True
    connect_timeout: float = 30.0
    account: str = ""  # empty = auto-detect

    def to_ibkr_config(self) -> IBKRConfig:
        """Convert to IBKRConfig."""
        return IBKRConfig(
            host=self.tws_host,
            port=self.tws_port,
            client_id=self.client_id,
            trade_mode="PAPER" if self.paper_trading else "LIVE",
            account_code=self.account,
        )


class IBBroker(Broker):
    """Interactive Brokers broker via IBKRClient (ib_async v2.x).

    Wraps IBKRClient to implement the Broker ABC. This allows the
    execution engine to use live IBKR connectivity while maintaining
    the same interface as PaperBroker.

    Requires IB Gateway or TWS running and configured for API access.

    Args:
        config: Optional IB configuration. If None, loads from .env.
    """

    def __init__(self, config: IBConfig | None = None):
        if config:
            self._client = IBKRClient(config.to_ibkr_config(), auto_load_env=False)
        else:
            self._client = IBKRClient(auto_load_env=True)

        self._positions_cache: dict[str, Position] = {}
        self._orders: dict[str, Order] = {}
        self._fills: list[Fill] = []

    def connect(self) -> None:
        """Establish connection to IB Gateway/TWS."""
        try:
            self._client.connect(timeout=int(30))
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            raise ConnectionError(
                f"Cannot connect to IBKR at {self._client.config.host}:{self._client.config.port}"
            ) from e

        # Verify account
        if not self._client.account:
            raise ConnectionError("No IBKR account available")

        logger.info(
            f"Connected to IBKR (mode={self._client.config.trade_mode}), "
            f"account={self._client.account}"
        )

    def disconnect(self) -> None:
        """Close IBKR connection."""
        self._client.disconnect()
        logger.info("Disconnected from IBKR")

    def is_connected(self) -> bool:
        """Check if connected to IBKR."""
        return self._client.connected

    def submit_order(self, order: Order) -> str:
        """Submit an order and return the broker order ID.

        Converts our Order model to an IBKR order and submits it.
        """
        if not self.is_connected():
            raise RuntimeError("IBBroker not connected")

        # Create IBKR contract
        contract = self._client.stock(order.ticker)

        # Create IBKR order
        if order.order_type == OrderType.MARKET:
            action = "BUY" if order.side == Side.BUY else "SELL"
            ib_order = self._client.market_order(action, order.quantity)
        elif order.order_type == OrderType.LIMIT and order.limit_price:
            action = "BUY" if order.side == Side.BUY else "SELL"
            ib_order = self._client.limit_order(
                action, order.quantity, order.limit_price
            )
        else:
            action = "BUY" if order.side == Side.BUY else "SELL"
            ib_order = self._client.market_order(action, order.quantity)

        # Place the order
        placed_order = self._client.place_order(contract, ib_order, pre_check=True)

        # Store our order mapping
        self._orders[order.order_id] = order
        broker_order_id = str(placed_order.order.orderId)

        logger.info(
            f"Submitted order {order.order_id}: "
            f"{order.side.value} {order.quantity} {order.ticker} "
            f"(IBKR orderId={broker_order_id})"
        )

        return broker_order_id

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if not self.is_connected():
            return False

        order = self._orders.get(order_id)
        if not order or order.is_complete:
            return False

        try:
            # Update local state
            cancelled = Order(
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
            self._orders[order_id] = cancelled
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_positions(self) -> dict[str, Position]:
        """Get current positions as dict[str, Position]."""
        if not self.is_connected():
            return dict(self._positions_cache)

        try:
            ib_positions = self._client.get_positions()
            positions = {}

            for pos in ib_positions:
                # Get current price
                current_price = self.get_current_price(pos.symbol)
                if current_price is None:
                    current_price = pos.avg_price

                positions[pos.symbol] = Position(
                    ticker=pos.symbol,
                    quantity=int(pos.quantity),
                    avg_cost=pos.avg_price,
                    current_price=current_price,
                    unrealized_pnl=pos.unrealized_pnl,
                    realized_pnl=pos.realized_pnl,
                )

            self._positions_cache = positions
            logger.debug(f"Retrieved {len(positions)} positions from IBKR")
        except Exception as e:
            logger.error(f"Failed to get positions from IBKR: {e}")

        return dict(self._positions_cache)

    def get_portfolio_value(self) -> float:
        """Get total portfolio value (cash + positions)."""
        if not self.is_connected():
            # Fallback: sum position market values
            return sum(pos.market_value for pos in self._positions_cache.values())

        try:
            pnl = self._client.get_account_pnl()
            net_liquidation = pnl.get("net_liquidation", 0.0)
            if net_liquidation > 0:
                return net_liquidation
        except Exception as e:
            logger.error(f"Failed to get portfolio value from IBKR: {e}")

        # Fallback
        return sum(pos.market_value for pos in self.get_positions().values())

    def get_current_price(self, ticker: str) -> Optional[float]:
        """Get current market price for a ticker."""
        if not self.is_connected():
            return None

        try:
            # Use snapshot for efficiency (no subscription overhead)
            snap = self._client.get_snapshot(ticker)
            if snap and snap.last:
                price = float(snap.last)
                if price > 0:
                    return price
        except Exception as e:
            logger.warning(f"IBKR snapshot failed for {ticker}: {e}")

        # Fallback: use yfinance for price lookup
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            if not hist.empty and "Close" in hist.columns:
                price = float(hist["Close"].iloc[-1])
                if price > 0:
                    logger.info(f"Using yfinance price for {ticker}: ${price:.2f}")
                    return price
        except Exception as e:
            logger.warning(f"yfinance fallback failed for {ticker}: {e}")

        return None

    def place_market_order(
        self,
        ticker: str,
        side: Side,
        quantity: int,
        signal_strength: float = 0.0,
        target_weight: float = 0.0,
        fill_price: Optional[float] = None,
    ) -> Order:
        """Place a market order via IBKR."""
        order = Order(
            ticker=ticker,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            signal_strength=signal_strength,
            target_weight=target_weight,
        )

        # Submit to IBKR
        broker_id = self.submit_order(order)

        # Update order with broker ID info
        # Note: IBKR orders are asynchronous — we don't know the fill
        # price until the fill callback fires. For now, mark as PENDING.
        logger.info(
            f"Market order {order.order_id}: {side.value} {quantity} {ticker} "
            f"submitted (broker_id={broker_id})"
        )

        return order

    def place_limit_order(
        self,
        ticker: str,
        side: Side,
        quantity: int,
        limit_price: float,
        signal_strength: float = 0.0,
        target_weight: float = 0.0,
        fill_price: Optional[float] = None,
    ) -> Order:
        """Place a limit order via IBKR."""
        order = Order(
            ticker=ticker,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            limit_price=limit_price,
            signal_strength=signal_strength,
            target_weight=target_weight,
        )

        # Submit to IBKR
        broker_id = self.submit_order(order)

        logger.info(
            f"Limit order {order.order_id}: {side.value} {quantity} {ticker} "
            f"@ {limit_price:.2f} (broker_id={broker_id})"
        )

        return order

    def get_fills(self) -> list[Fill]:
        """Get recent fills from IBKR."""
        if not self.is_connected():
            return list(self._fills)

        try:
            ib_fills = self._client.get_fills()
            fills = []

            for fill in ib_fills:
                side = Side.BUY if fill.action == "BUY" else Side.SELL
                fills.append(Fill(
                    ticker=fill.symbol,
                    side=side,
                    quantity=int(fill.quantity),
                    price=float(fill.price),
                    commission=float(fill.commission),
                    fill_status=FillStatus.COMPLETE,
                    order_id=str(fill.date),  # Use date as identifier since IBKR FillRecord lacks order_id
                ))

            self._fills = fills
            logger.debug(f"Retrieved {len(fills)} fills from IBKR")
        except Exception as e:
            logger.error(f"Failed to get fills from IBKR: {e}")

        return list(self._fills)

    def get_order_history(self) -> dict[str, Order]:
        """Get all orders tracked by this broker."""
        return dict(self._orders)