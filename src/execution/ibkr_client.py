"""Interactive Brokers integration for the Alpha trading system.

Provides a thin wrapper around ib_insync for paper and live trading:
- Connection management (auto-reconnect, health checks)
- Market data subscriptions (real-time quotes, historical data)
- Order placement and management
- Position tracking and P&L reconciliation

Architecture:
    Signals -> Ensemble -> Portfolio -> IBKRClient -> IB Gateway -> IBKR

Usage:
    from src.execution.ibkr_client import IBKRClient

    client = IBKRClient()
    client.connect()

    # Real-time data
    ticker = client.get_ticker("AAPL")
    print(ticker.close())

    # Place order
    contract = client.stock("AAPL")
    order = client.market_order("BUY", 100)
    client.place_order(contract, order)

    # Positions
    positions = client.get_positions()
    pnl = client.get_account_pnl()
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

try:
    from ib_async import (
        IB,
        AccountValue,
        Contract,
        Fill,
        Order,
        PortfolioItem,
        Position,
        Ticker,
        util,
        BarDataList,
        TagValue,
    )
    from ib_async.contract import Stock, Option, CFD, Forex
    HAS_IB = True
except ImportError:
    HAS_IB = False
    logger.warning(
        "ib_async not installed. Install with: uv add 'ib_async>=2.0'"
    )


@dataclass
class IBKRConfig:
    """Configuration for IBKR connection.

    Reads from environment variables or .env file.
    """
    host: str = "127.0.0.1"
    port: int = 4001
    client_id: int = 1
    trade_mode: str = "PAPER"  # PAPER or LIVE
    username: str = ""
    password: str = ""
    max_reconnect_attempts: int = 3
    reconnect_delay_seconds: int = 5
    readonly: bool = False
    account_code: str = ""

    @classmethod
    def from_env(cls) -> "IBKRConfig":
        """Load configuration from environment variables."""
        return cls(
            host=os.environ.get("IBKR_HOST", "127.0.0.1"),
            port=int(os.environ.get("IBKR_PORT", "4001")),
            client_id=int(os.environ.get("IBKR_CLIENT_ID", "1")),
            trade_mode=os.environ.get("IBKR_TRADE_MODE", "PAPER").upper(),
            username=os.environ.get("IBKR_USERNAME", ""),
            password=os.environ.get("IBKR_PASSWORD", ""),
        )


@dataclass
class FillRecord:
    """Represents a completed trade fill."""
    symbol: str
    action: str  # BUY or SELL
    quantity: float
    price: float
    date: str
    commission: float
    currency: str


@dataclass
class PositionRecord:
    """Represents a current position."""
    symbol: str
    quantity: float
    avg_price: float
    unrealized_pnl: float
    realized_pnl: float


class IBKRClient:
    """Thin wrapper around ib_insync for the Alpha trading system.

    Handles connection lifecycle, error handling, and provides
    typed interfaces for market data, orders, and account state.

    Safety features:
    - Trade mode guard (PAPER vs LIVE)
    - Position limit enforcement before order submission
    - Auto-reconnect on disconnection
    - Fill tracking and reconciliation
    """

    def __init__(
        self,
        config: Optional[IBKRConfig] = None,
        auto_load_env: bool = True,
    ):
        """Initialize the IBKR client.

        Args:
            config: Configuration override. If None, loads from environment.
            auto_load_env: If True and config is None, load from env vars.
        """
        if not HAS_IB:
            raise ImportError(
                "ib_async is required. Install with: uv add 'ib_async>=2.0'"
            )

        if config is None and auto_load_env:
            config = IBKRConfig.from_env()
        self.config = config or IBKRConfig()

        # Normalize trade mode to uppercase
        self.config.trade_mode = self.config.trade_mode.upper()

        self._ib: Optional[IB] = None
        self._connected = False
        self._account: str = ""
        self._reconnect_attempts: int = 0

        logger.info(
            f"IBKRClient initialized: mode={self.config.trade_mode}, "
            f"host={self.config.host}:{self.config.port}, "
            f"client_id={self.config.client_id}"
        )

    @property
    def connected(self) -> bool:
        """Whether the client is currently connected to IB Gateway."""
        return self._connected

    @property
    def account(self) -> str:
        """The current account number."""
        return self._account

    def connect(self, timeout: int = 30) -> bool:
        """Connect to IB Gateway / TWS.

        Args:
            timeout: Seconds to wait for connection establishment.

        Returns:
            True if connected successfully.

        Raises:
            ConnectionError: If connection fails after retries.
        """
        if self._connected:
            logger.debug("Already connected to IBKR")
            return True

        if not self.config.username or not self.config.password:
            logger.warning(
                "No credentials provided. Ensure IB Gateway is already logged in."
            )

        try:
            self._ib = IB()
            self._ib.connect(
                self.config.host,
                self.config.port,
                clientId=self.config.client_id,
                timeout=timeout,
            )

            self._connected = True
            self._reconnect_attempts = 0

            # Discover account
            self._account = self._ib.managedAccounts()[0] if self._ib.managedAccounts() else ""

            logger.info(
                f"Connected to IBKR: account={self._account}, "
                f"mode={self.config.trade_mode}"
            )

            # Warm-up period — IB Gateway needs ~30 seconds before
            # market data subscriptions work reliably
            logger.debug("Waiting for IB Gateway warm-up period...")
            time.sleep(2)

            return True

        except Exception as e:
            self._reconnect_attempts += 1
            error_msg = f"Connection attempt {self._reconnect_attempts} failed: {e}"

            if self._reconnect_attempts < self.config.max_reconnect_attempts:
                logger.warning(
                    f"{error_msg} Retrying in {self.config.reconnect_delay_seconds}s..."
                )
                time.sleep(self.config.reconnect_delay_seconds)
                return self.connect(timeout=timeout)
            else:
                self._connected = False
                logger.error(f"Failed to connect after {self._reconnect_attempts} attempts")
                raise ConnectionError(error_msg)

    def disconnect(self) -> None:
        """Disconnect from IB Gateway."""
        if self._ib and self._connected:
            self._ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IBKR")

    def reconnect(self) -> bool:
        """Attempt to reconnect after disconnection."""
        logger.warning("Attempting to reconnect to IBKR...")
        self.disconnect()
        return self.connect()

    # ------------------------------------------------------------------
    # Contract helpers
    # ------------------------------------------------------------------

    def stock(
        self,
        symbol: str,
        exchange: str = "SMART",
        currency: str = "USD",
        primary_exchange: str = "NASDAQ",
    ) -> Stock:
        """Create a stock contract.

        Args:
            symbol: Ticker symbol (e.g., "AAPL")
            exchange: Trading exchange (default: SMART for best routing)
            currency: Settlement currency
            primary_exchange: Primary listing exchange
        """
        return Stock(
            symbol,
            exchange=exchange,
            currency=currency,
            primaryExchange=primary_exchange,
        )

    def forex(
        self,
        pair: str,
        exchange: str = "IDEALPRO",
        currency: str = "",
    ) -> Forex:
        """Create a forex contract.

        Args:
            pair: Currency pair (e.g., "EUR.USD")
            exchange: Exchange (default: IDEALPRO for forex)
        """
        return Forex(pair, exchange=exchange, currency=currency)

    def cfd(
        self,
        symbol: str,
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> CFD:
        """Create a CFD contract."""
        return CFD(symbol, exchange=exchange, currency=currency)

    # ------------------------------------------------------------------
    # Order helpers
    # ------------------------------------------------------------------

    def market_order(
        self,
        action: str,
        quantity: float,
        account: Optional[str] = None,
    ) -> Order:
        """Create a market order.

        Args:
            action: "BUY" or "SELL"
            quantity: Number of shares
        """
        order = Order(
            action=action,
            orderType="MKT",
            totalQuantity=quantity,
        )
        if account:
            order.account = account
        return order

    def limit_order(
        self,
        action: str,
        quantity: float,
        limit_price: float,
        tif: str = "GTC",
        account: Optional[str] = None,
    ) -> Order:
        """Create a limit order.

        Args:
            action: "BUY" or "SELL"
            quantity: Number of shares
            limit_price: Limit price
            tif: Time in force (GTC, DAY, IOC, etc.)
        """
        order = Order(
            action=action,
            orderType="LMT",
            totalQuantity=quantity,
            lmtPrice=limit_price,
            tif=tif,
        )
        if account:
            order.account = account
        return order

    def stop_order(
        self,
        action: str,
        quantity: float,
        stop_price: float,
        account: Optional[str] = None,
    ) -> Order:
        """Create a stop market order.

        Args:
            action: "BUY" or "SELL"
            quantity: Number of shares
            stop_price: Stop trigger price
        """
        order = Order(
            action=action,
            orderType="STP",
            totalQuantity=quantity,
            auxPrice=stop_price,
        )
        if account:
            order.account = account
        return order

    def bracket_order(
        self,
        quantity: float,
        limit_price: float,
        take_profit: float,
        stop_loss: float,
        account: Optional[str] = None,
    ) -> list[Order]:
        """Create a bracket order (entry + TP + SL).

        Returns three orders: parent limit + OCO child group.
        """
        parent = self.limit_order("BUY", quantity, limit_price, account=account)

        tp = self.limit_order("SELL", quantity, take_profit, account=account)

        sl = self.stop_order("SELL", quantity, stop_loss, account=account)

        # Parent order carries OCO children
        parent.ocoGroup = True
        tp.ocaName = f"bracket_{int(time.time())}"
        tp.ocoGroup = True
        sl.ocaName = tp.ocaName
        sl.ocoGroup = True

        return [parent, tp, sl]

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def place_order(
        self,
        contract: Contract,
        order: Order,
        pre_check: bool = True,
    ) -> Order:
        """Place an order with optional pre-submission checks.

        Args:
            contract: The contract to trade
            order: The order specification
            pre_check: If True, run safety checks before submission

        Returns:
            The submitted order with server-assigned orderID.

        Raises:
            ValueError: If pre-check fails (e.g., trade mode mismatch)
            ConnectionError: If not connected to IB Gateway
        """
        if not self._connected:
            raise ConnectionError("Not connected to IBKR. Call connect() first.")

        if pre_check:
            self._pre_order_check(contract, order)

        logger.info(
            f"Placing order: {order.action} {order.totalQuantity} "
            f"{contract.symbol} @ {order.orderType}"
        )

        try:
            placed_order = self._ib.placeOrder(contract, order)

            logger.info(
                f"Order placed: orderId={placed_order.orderId}, "
                f"status={placed_order.status}"
            )
            return placed_order

        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            raise

    def cancel_order(self, order: Order) -> bool:
        """Cancel a pending order."""
        if not self._connected:
            raise ConnectionError("Not connected to IBKR.")

        logger.info(f"Cancelling order: orderId={order.orderId}")
        try:
            self._ib.cancelOrder(order)
            return True
        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            return False

    def get_open_orders(self) -> list[Order]:
        """Get all open (pending) orders."""
        if not self._connected:
            raise ConnectionError("Not connected to IBKR.")
        return self._ib.openOrders()

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_ticker(
        self,
        symbol: str,
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> Optional[Ticker]:
        """Get real-time ticker data for a symbol.

        Subscribes to live data and returns immediately (async fill).

        Args:
            symbol: Ticker symbol
            exchange: Exchange routing

        Returns:
            Ticker object with real-time data (may be partially filled).
        """
        if not self._connected:
            raise ConnectionError("Not connected to IBKR.")

        contract = self.stock(symbol, exchange=exchange, currency=currency)
        ticker = self._ib.reqMktData(contract, regexSnapshot="")

        # Wait briefly for initial data
        self._ib.sleep(1)

        if ticker and ticker.last:
            logger.debug(
                f"Ticker {symbol}: last={ticker.last}, "
                f"bid={ticker.bid}, ask={ticker.ask}"
            )
        else:
            logger.warning(f"No ticker data available for {symbol}")

        return ticker

    def cancel_ticker(
        self,
        symbol: str,
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> None:
        """Cancel a market data subscription."""
        if not self._connected:
            return
        contract = self.stock(symbol, exchange=exchange, currency=currency)
        self._ib.cancelMktData(contract)

    def get_historical_data(
        self,
        symbol: str,
        duration: str = "30 D",
        bar_size: str = "1 day",
        what_to_show: str = "trades",
        exchange: str = "SMART",
    ) -> BarDataList:
        """Fetch historical bar data.

        Args:
            symbol: Ticker symbol
            duration: Duration string (e.g., "30 D", "1 Y", "6 M")
            bar_size: Bar size (e.g., "1 min", "5 mins", "1 day")
            what_to_show: "trades", "bids", "midpoint", etc.
            exchange: Exchange routing

        Returns:
            List of BarData objects.
        """
        if not self._connected:
            raise ConnectionError("Not connected to IBKR.")

        contract = self.stock(symbol, exchange=exchange)
        bars = self._ib.reqHistoricalData(
            contract,
            durationStr=duration,
            barSizeStr=bar_size,
            whatToShow=what_to_show,
            useRTH=True,  # Regular trading hours only
            formatDate=1,
        )

        logger.info(f"Received {len(bars)} bars for {symbol}")
        return bars

    def get_snapshot(
        self,
        symbol: str,
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> Optional[Ticker]:
        """Get a one-time price snapshot (no subscription).

        Useful for batch price lookups without maintaining subscriptions.
        """
        if not self._connected:
            raise ConnectionError("Not connected to IBKR.")

        contract = self.stock(symbol, exchange=exchange, currency=currency)
        ticker = self._ib.reqMktData(contract)
        self._ib.sleep(1)
        self._ib.cancelMktData(contract)
        return ticker

    # ------------------------------------------------------------------
    # Positions and account
    # ------------------------------------------------------------------

    def get_positions(self) -> list[PositionRecord]:
        """Get all current positions.

        Returns:
            List of PositionRecord objects with symbol, quantity, and P&L.
        """
        if not self._connected:
            raise ConnectionError("Not connected to IBKR.")

        positions = self._ib.positions()
        result = []

        for pos in positions:
            portfolio = self._ib.portfolio()
            pos_record = PositionRecord(
                symbol=pos.contract.symbol,
                quantity=pos.position,
                avg_price=pos.avgCost,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
            )

            # Look up P&L from portfolio
            for port_item in portfolio:
                if port_item.contract.symbol == pos.contract.symbol:
                    pos_record.unrealized_pnl = port_item.unrealizedPNL
                    break

            result.append(pos_record)

        logger.info(f"Retrieved {len(result)} positions")
        return result

    def get_fills(self) -> list[FillRecord]:
        """Get recent trade fills.

        Returns:
            List of FillRecord objects.
        """
        if not self._connected:
            raise ConnectionError("Not connected to IBKR.")

        fills = self._ib.fills()
        result = []

        for fill in fills:
            result.append(
                FillRecord(
                    symbol=fill.contract.symbol,
                    action=fill.order.action,
                    quantity=fill.fill.quantity,
                    price=fill.fill.price,
                    date=str(fill.fill.time),
                    commission=fill.fill.commissionReport.commission
                    if fill.fill.commissionReport
                    else 0.0,
                    currency=fill.fill.commissionReport.currency
                    if fill.fill.commissionReport
                    else "USD",
                )
            )

        logger.info(f"Retrieved {len(result)} fills")
        return result

    def get_account_summary(self) -> dict[str, AccountValue]:
        """Get account summary values.

        Returns:
            Dict mapping account value names to AccountValue objects.
            Common keys: NetLiquidation, TotalCashValue, BuyingPower,
            AvailableFunds, EquityWithLoanValue, etc.
        """
        if not self._connected:
            raise ConnectionError("Not connected to IBKR.")

        values = self._ib.accountValues()
        summary = {av.tag: av for av in values}

        logger.debug(
            f"Account summary: {len(summary)} values retrieved"
        )
        return summary

    def get_account_pnl(self) -> dict:
        """Get key account P&L metrics.

        Returns:
            Dict with net_liquidation, total_cash, buying_power,
            available_funds, equity, and daily_pnl_change.
        """
        summary = self.get_account_summary()

        def _get(tag: str, default: float = 0.0) -> float:
            try:
                return float(summary.get(tag, AccountValue(value=default)).value)
            except (ValueError, TypeError):
                return default

        return {
            "net_liquidation": _get("NetLiquidation"),
            "total_cash": _get("TotalCashValue"),
            "buying_power": _get("BuyingPower"),
            "available_funds": _get("AvailableFunds"),
            "equity": _get("EquityWithLoanValue"),
            "daily_change": _get("DailyAccountPnL"),
        }

    # ------------------------------------------------------------------
    # Risk checks
    # ------------------------------------------------------------------

    def check_position_limit(
        self,
        symbol: str,
        max_position: float,
        current_positions: Optional[list[PositionRecord]] = None,
    ) -> bool:
        """Check if adding a position would exceed the limit.

        Args:
            symbol: Ticker symbol
            max_position: Maximum allowed position size
            current_positions: Optional cache of positions

        Returns:
            True if within limits.
        """
        if current_positions is None:
            current_positions = self.get_positions()

        for pos in current_positions:
            if pos.symbol == symbol and abs(pos.quantity) >= max_position:
                logger.warning(
                    f"Position limit check failed for {symbol}: "
                    f"current={pos.quantity}, limit={max_position}"
                )
                return False

        return True

    def _pre_order_check(
        self,
        contract: Contract,
        order: Order,
    ) -> None:
        """Run safety checks before order submission.

        Raises:
            ValueError: If any check fails.
        """
        # Trade mode warning
        if self.config.trade_mode != "PAPER":
            logger.warning(
                f"LIVE TRADE MODE: {order.action} {order.totalQuantity} "
                f"{contract.symbol}"
            )

        # Quantity validation
        if order.totalQuantity <= 0:
            raise ValueError(
                f"Invalid order quantity: {order.totalQuantity}"
            )

        # Whole-share check for stocks
        if isinstance(contract, Stock) and order.totalQuantity != int(order.totalQuantity):
            raise ValueError(
                f"Stock orders require whole shares: {order.totalQuantity}"
            )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def run(self, *coros, timeout: Optional[int] = None) -> list:
        """Run async coroutines in the IB event loop.

        Proxy to ib_insync's util.run for scheduling async work.
        """
        if not HAS_IB:
            raise ImportError("ib_insync not available")
        return util.run(*coros, timeout=timeout)

    def wait_until_disconnected(self) -> None:
        """Block until the IB connection drops."""
        if self._ib:
            self._ib.waitUntilDisconnected()

    def __enter__(self) -> "IBKRClient":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.disconnect()
        return False

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return (
            f"IBKRClient({status}, mode={self.config.trade_mode}, "
            f"account={self._account})"
        )