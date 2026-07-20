"""Execution engine — converts portfolio targets into orders and manages lifecycle.

Pipeline:
    Signals → Portfolio targets → ExecutionEngine → Broker → Market

The engine:
1. Receives portfolio targets (ticker → desired weight/quantity)
2. Compares against current positions
3. Generates orders to reconcile the gap
4. Validates orders against risk guardrails
5. Submits approved orders to the broker
6. Tracks fills and reconciles state
7. Monitors daily P&L and drawdown
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import polars as pl

from .broker import Broker, PaperBroker
from .models import (
    Fill,
    Order,
    OrderStatus,
    OrderType,
    PortfolioSnapshot,
    Position,
    Side,
)
from .risk import RiskConfig, RiskGuard

logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    """Execution engine configuration."""

    # Initial cash for paper trading
    initial_cash: float = 100_000.0

    # Commission per share
    commission_per_share: float = 0.005

    # Slippage in basis points
    slippage_bps: float = 10.0

    # Order type: MARKET or LIMIT
    default_order_type: OrderType = OrderType.MARKET

    # Max positions allowed
    max_positions: int = 20

    # Max portfolio weight per position
    max_position_weight: float = 0.10

    # Daily loss circuit breaker (% of portfolio)
    daily_loss_limit_pct: float = 0.05

    # Max drawdown kill switch (% from peak)
    max_drawdown_pct: float = 0.15

    # Minimum cash reserve (% of portfolio)
    min_cash_reserve: float = 0.05

    # Max leverage
    max_leverage: float = 1.0

    # Max order size relative to portfolio
    max_order_size_pct: float = 0.05


class ExecutionEngine:
    """Orchestrates signal-to-order execution with risk guardrails.

    Usage (paper trading):
        engine = ExecutionEngine(ExecutionConfig(initial_cash=100_000))
        engine.start()
        targets = engine.generate_targets_from_signals(signal_df)
        engine.execute_targets(targets)

    Usage (live with IB):
        from src.execution.ib_broker import IBBroker
        broker = IBBroker(ib_config)
        engine = ExecutionEngine(
            ExecutionConfig(),
            broker=broker,
        )
        engine.start()

    Args:
        config: Execution configuration.
        broker: Optional broker instance (creates PaperBroker if not provided).
    """

    def __init__(
        self,
        config: ExecutionConfig,
        broker: Broker | None = None,
    ):
        self.config = config
        self.broker = broker or PaperBroker(
            initial_cash=config.initial_cash,
            commission_per_share=config.commission_per_share,
            slippage_bps=config.slippage_bps,
        )

        self.risk_config = RiskConfig(
            max_positions=config.max_positions,
            max_position_weight=config.max_position_weight,
            max_leverage=config.max_leverage,
            daily_loss_limit_pct=config.daily_loss_limit_pct,
            max_drawdown_pct=config.max_drawdown_pct,
            min_cash_reserve=config.min_cash_reserve,
            max_order_size_pct=config.max_order_size_pct,
        )

        # Internal state
        self._orders: dict[str, Order] = {}
        self._fills: list[Fill] = []
        self._peak_equity: float = config.initial_cash
        self._daily_pnl: float = 0.0
        self._num_orders_today: int = 0
        self._num_trades_today: int = 0
        self._running: bool = False

    def start(self) -> None:
        """Start the execution engine (connect broker)."""
        self.broker.connect()
        self._running = True
        self._peak_equity = self.broker.get_portfolio_value()
        logger.info("Execution engine started")

    def stop(self) -> None:
        """Stop the execution engine (disconnect broker)."""
        if self.risk_config.kill_switch_active:
            logger.warning("Kill switch was active — all trading halted")
        self.broker.disconnect()
        self._running = False
        logger.info("Execution engine stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def kill_switch_active(self) -> bool:
        return self.risk_config.kill_switch_active

    def get_portfolio_value(self) -> float:
        """Get current total portfolio value."""
        return self.broker.get_portfolio_value()

    def get_positions(self) -> dict[str, Position]:
        """Get current positions."""
        return self.broker.get_positions()

    def get_snapshot(self) -> PortfolioSnapshot:
        """Get current portfolio snapshot."""
        positions = self.broker.get_positions()
        portfolio_value = self.broker.get_portfolio_value()

        # Update peak equity
        if portfolio_value > self._peak_equity:
            self._peak_equity = portfolio_value

        # Calculate drawdown
        drawdown = 0.0
        if self._peak_equity > 0:
            drawdown = (self._peak_equity - portfolio_value) / self._peak_equity

        return PortfolioSnapshot(
            timestamp=datetime.now(timezone.utc),
            positions=positions,
            cash=0.0,
            total_equity=portfolio_value,
            daily_pnl=self._daily_pnl,
            drawdown=drawdown,
            num_positions=len(positions),
            num_orders_today=self._num_orders_today,
            num_trades_today=self._num_trades_today,
        )

    def generate_targets_from_signals(
        self,
        signal_df: pl.DataFrame,
        *,
        signal_column: str = "ensemble_signal",
        top_n: int | None = None,
        min_signal_strength: float = 0.1,
    ) -> dict[str, float]:
        """Generate portfolio targets from a signal DataFrame.

        Converts signal values into target weights using a simple
        proportional allocation scheme.

        Args:
            signal_df: DataFrame with (ticker, date, signal_column).
            signal_column: Name of the signal column.
            top_n: Only trade top N signals by absolute strength.
            min_signal_strength: Minimum absolute signal to trade.

        Returns:
            Dict mapping ticker -> target_weight (in [-1, +1]).
        """
        if signal_df.is_empty() or signal_column not in signal_df.columns:
            logger.warning("Empty signal DataFrame or missing signal column")
            return {}

        # Filter by minimum signal strength
        filtered = signal_df.filter(
            pl.col(signal_column).abs() >= min_signal_strength
        )

        if filtered.is_empty():
            return {}

        # Get the most recent date
        latest_date = filtered["date"].max()
        latest = filtered.filter(pl.col("date") == latest_date)

        # Top N by absolute signal strength
        if top_n:
            latest = latest.sort(
                by=signal_column, descending=True
            ).head(top_n)

        # Convert signal strength to target weights
        # Normalize so that sum of absolute weights = 1 (fully invested)
        signals = latest.select(
            pl.col("ticker"),
            pl.col(signal_column).alias("signal"),
        )

        total_abs = float(signals["signal"].abs().sum())
        if total_abs == 0:
            return {}

        targets = {}
        for row in signals.iter_rows(named=True):
            weight = row["signal"] / total_abs
            targets[row["ticker"]] = round(weight, 4)

        return targets

    def execute_targets(
        self,
        targets: dict[str, float],
        price_source: dict[str, pl.DataFrame] | None = None,
    ) -> list[Order]:
        """Execute portfolio targets.

        Compares targets against current positions and generates
        orders to reconcile the gap.

        Args:
            targets: Dict mapping ticker -> target_weight.
            price_source: Optional price data for position sizing.

        Returns:
            List of orders that were submitted.
        """
        if not self._running:
            raise RuntimeError("Execution engine not running")

        if self.risk_config.kill_switch_active:
            logger.error("Kill switch active — skipping execution")
            return []

        if price_source and isinstance(self.broker, PaperBroker):
            self.broker.price_source = price_source

        portfolio_value = self.get_portfolio_value()
        current_positions = self.get_positions()

        # Calculate desired quantities
        orders = []
        for ticker, target_weight in targets.items():
            desired_value = abs(target_weight) * portfolio_value

            # Get current price
            price = self.broker.get_current_price(ticker)
            if price is None or price <= 0:
                logger.warning(f"No price for {ticker}, skipping")
                continue

            # Calculate desired quantity (in whole shares)
            desired_qty = max(1, int(desired_value / price))

            # Current position
            current_qty = 0
            if ticker in current_positions:
                current_qty = current_positions[ticker].quantity

            # Calculate gap
            gap = desired_qty - current_qty

            # Determine side and quantity
            side: Optional[Side] = None
            qty = 0

            if target_weight == 0 or (current_qty > 0 and target_weight == 0):
                # Close position
                if current_qty > 0:
                    side = Side.SELL
                    qty = current_qty
            elif gap > 0:
                side = Side.BUY
                qty = gap
            else:
                side = Side.SELL
                qty = abs(gap)

            if qty <= 0 or side is None:
                continue  # No action needed

            # Create order with price info for risk checks
            order = Order(
                ticker=ticker,
                side=side,
                order_type=self.config.default_order_type,
                quantity=qty,
                limit_price=price if self.config.default_order_type == OrderType.LIMIT else None,
                signal_strength=abs(target_weight),
                target_weight=target_weight,
                avg_fill_price=price,  # For risk validation
            )

            # Validate against risk guardrails
            risk_guard = RiskGuard(
                self.risk_config,
                portfolio_value=portfolio_value,
                positions=current_positions,
                cash=portfolio_value - sum(
                    pos.market_value for pos in current_positions.values()
                ),
                daily_pnl=self._daily_pnl,
                peak_equity=self._peak_equity,
            )

            approved, reason = risk_guard.validate_order(order)
            if not approved:
                logger.warning(f"Order REJECTED for {ticker}: {reason}")
                continue

            # Execute the order
            if self.config.default_order_type == OrderType.LIMIT:
                executed = self.broker.place_limit_order(
                    ticker=ticker,
                    side=side,
                    quantity=qty,
                    limit_price=order.limit_price or price,
                    signal_strength=order.signal_strength,
                    target_weight=order.target_weight,
                )
            else:
                executed = self.broker.place_market_order(
                    ticker=ticker,
                    side=side,
                    quantity=qty,
                    signal_strength=order.signal_strength,
                    target_weight=order.target_weight,
                )

            # Track the order
            self._orders[executed.order_id] = executed
            self._num_orders_today += 1

            if executed.filled_qty > 0:
                self._num_trades_today += 1

            orders.append(executed)
            logger.info(
                f"Order {executed.order_id}: {side.value} {qty} {ticker} "
                f"@ {executed.avg_fill_price:.2f}"
            )

        return orders

    def record_end_of_day(self) -> float:
        """Record end-of-day P&L. Call once per trading day."""
        if isinstance(self.broker, PaperBroker):
            return self.broker.record_eod()

        # For live brokers, calculate manually
        current_value = self.get_portfolio_value()
        self._daily_pnl = current_value - self._peak_equity
        return self._daily_pnl

    def reset_daily_counters(self) -> None:
        """Reset daily counters at the start of a new trading day."""
        self._daily_pnl = 0.0
        self._num_orders_today = 0
        self._num_trades_today = 0

    def cancel_all_orders(self) -> int:
        """Cancel all pending orders (emergency stop)."""
        cancelled = 0
        for order_id, order in self._orders.items():
            if not order.is_complete:
                if self.broker.cancel_order(order_id):
                    cancelled += 1
        return cancelled

    def emergency_stop(self) -> None:
        """Emergency stop: cancel all orders, activate kill switch, disconnect."""
        logger.critical("EMERGENCY STOP activated")
        self.risk_config.activate_kill_switch()
        self.cancel_all_orders()
        self.stop()

    def get_order_history(self) -> dict[str, Order]:
        """Get all orders processed by this engine."""
        return dict(self._orders)

    def get_fill_history(self) -> list[Fill]:
        """Get all fills processed by this engine."""
        if isinstance(self.broker, PaperBroker):
            return self.broker.get_fills()
        return self._fills