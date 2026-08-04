"""Intraday execution engine for bar-by-bar trading.

Designed for intraday mean reversion strategies with:
- Bar-by-bar signal processing via Broker ABC
- IBKR integration for live trading (via IBBroker)
- Paper trading support (via PaperBroker)
- Risk management (stop loss, take profit, position limits)
- Session management (market open/close, EOD position closure)
- Intraday P&L tracking and metrics

Architecture:
    Bar stream -> compute_intraday_features -> signal -> IntradayEngine -> Broker -> Market

Usage (paper):
    from src.execution.intraday_engine import IntradayEngine, IntradayConfig
    from src.execution.broker import PaperBroker

    config = IntradayConfig(ticker="MSFT")
    engine = IntradayEngine(config, broker=PaperBroker(...))
    engine.start_session()
    for bar in bars:
        engine.process_bar(bar)
    engine.end_session()
    print(engine.get_metrics())

Usage (live with IBKR):
    from src.execution.intraday_engine import IntradayEngine, IntradayConfig
    from src.execution.ib_broker import IBBroker

    config = IntradayConfig(ticker="MSFT", use_ibkr=True)
    engine = IntradayEngine(config, broker=IBBroker())
    engine.connect()
    engine.start_session()
    # Stream bars from IBKR and call process_bar() each
    engine.end_session()
    engine.disconnect()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from typing import Optional

import polars as pl

from .broker import Broker, PaperBroker
from .models import Fill, Order, OrderStatus, OrderType, Position, Side

log = logging.getLogger(__name__)


# US market times (UTC)
MARKET_OPEN_UTC = time(14, 30)   # 14:30 UTC = 09:30 ET (winter)
MARKET_CLOSE_UTC = time(21, 0)   # 21:00 UTC = 16:00 ET (winter)
MARKET_OPEN_ET = time(9, 30)
MARKET_CLOSE_ET = time(16, 0)


@dataclass
class IntradayConfig:
    """Configuration for intraday trading."""

    ticker: str = "MSFT"
    timeframe: str = "1min"  # Bar timeframe
    signal_column: str = "signal_vwap_mean_reversion_filtered"
    max_position_shares: int = 10000  # Max shares to hold
    stop_loss_pct: float = 0.005  # 0.5% stop loss
    take_profit_pct: float = 0.0075  # 0.75% take profit
    commission_per_share: float = 0.005
    slippage_bps: float = 10.0  # 0.1%
    initial_cash: float = 1_000_000.0
    use_ibkr: bool = False  # Use IBKR for live trading
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 4002

    # Session management
    market_open_utc: time = MARKET_OPEN_UTC
    market_close_utc: time = MARKET_CLOSE_UTC
    close_positions_at_end: bool = True  # Auto-close at market close

    # Risk
    max_daily_loss_pct: float = 0.02  # 2% daily loss circuit breaker
    kill_switch_active: bool = False


@dataclass
class SessionMetrics:
    """Metrics for a single trading session."""

    session_date: Optional[datetime] = None
    initial_equity: float = 0.0
    final_equity: float = 0.0
    peak_equity: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_commissions: float = 0.0
    avg_trade_pnl: float = 0.0
    max_position: int = 0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0


class IntradayEngine:
    """Bar-by-bar intraday execution engine.

    Processes streaming bars, computes signals, and executes trades
    through a Broker (PaperBroker for backtesting, IBBroker for live).

    The engine manages:
    - Position state (shares, entry price, P&L)
    - Risk management (stop loss, take profit, daily loss limit)
    - Session lifecycle (start, process bars, end, close positions)
    - Metrics collection (trades, equity curve, P&L)

    Args:
        config: Intraday trading configuration.
        broker: Broker instance (PaperBroker or IBBroker). If None,
            creates a PaperBroker with config defaults.
    """

    def __init__(
        self,
        config: Optional[IntradayConfig] = None,
        broker: Optional[Broker] = None,
    ):
        self.config = config or IntradayConfig()
        self.broker = broker or PaperBroker(
            initial_cash=self.config.initial_cash,
            commission_per_share=self.config.commission_per_share,
            slippage_bps=self.config.slippage_bps,
            price_source={},
        )

        # Position state
        self._position: int = 0
        self._entry_price: float = 0.0
        self._peak_equity: float = self.config.initial_cash

        # Live-broker async-fill state (pitfall 30)
        self._pending_order: Optional[dict] = None
        self._reject_strikes: int = 0
        self._order_pause: int = 0

        # Session state
        self._session_active: bool = False
        self._session_date: Optional[datetime] = None

        # Metrics
        self._trades: list[dict] = []
        self._equity_curve: list[dict] = []
        self._fills: list[Fill] = []
        self._daily_pnl: float = 0.0
        self._winning_pnl: float = 0.0
        self._losing_pnl: float = 0.0

        log.info(
            f"IntradayEngine initialized: ticker={self.config.ticker}, "
            f"timeframe={self.config.timeframe}, broker={type(self.broker).__name__}"
        )

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to the broker."""
        self.broker.connect()
        log.info("Connected to broker")

    def disconnect(self) -> None:
        """Disconnect from the broker."""
        if self.broker.is_connected():
            self.broker.disconnect()
            log.info("Disconnected from broker")

    @property
    def is_connected(self) -> bool:
        """Whether the broker is connected."""
        return self.broker.is_connected()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(self, session_date: Optional[datetime] = None) -> None:
        """Start a new trading session.

        Resets daily counters and initializes session state.

        Args:
            session_date: Override the session date (defaults to now).
        """
        self._session_active = True
        self._session_date = session_date or datetime.now(timezone.utc)
        self._daily_pnl = 0.0
        self._peak_equity = self.config.initial_cash

        # Reset daily kill switch
        self.config.kill_switch_active = False

        # Reset live-broker async-fill state and adopt the broker's actual
        # book (restarting with an open position must not assume flat).
        self._pending_order = None
        self._reject_strikes = 0
        self._order_pause = 0
        self._reconcile_position(record_trade=False)

        initial_equity = self.broker.get_portfolio_value()
        if initial_equity > 0:
            self._peak_equity = initial_equity

        self._equity_curve.append({
            "datetime": self._session_date,
            "portfolio_value": initial_equity,
        })

        log.info(
            f"Session started: {self._session_date.date()}, "
            f"initial_equity=${initial_equity:,.2f}"
        )

    def end_session(self, current_price: Optional[float] = None) -> dict:
        """End the current trading session.

        If close_positions_at_end is True, closes all open positions.
        Returns session metrics.

        Args:
            current_price: Price to use for closing positions.
                If None, fetches from broker.

        Returns:
            Dict with session metrics.
        """
        if not self._session_active:
            log.warning("No active session to end")
            return {}

        # Close positions if configured
        if self.config.close_positions_at_end and self._position != 0:
            if current_price is None:
                current_price = self.broker.get_current_price(self.config.ticker)
            if current_price:
                self._close_position(current_price, "end_of_session")

        # Final equity snapshot (reconcile first so async closes are counted)
        self._reconcile_position()
        final_equity = self.broker.get_portfolio_value()

        self._session_active = False
        log.info(
            f"Session ended: final_equity=${final_equity:,.2f}, "
            f"trades={len(self._trades)}"
        )

        return self.get_session_metrics()

    # ------------------------------------------------------------------
    # Bar processing
    # ------------------------------------------------------------------

    def process_bar(self, bar: dict) -> Optional[dict]:
        """Process a single bar and execute trades if needed.

        Pipeline:
        1. Check kill switch
        2. Check stop loss / take profit
        3. Compute target position from signal
        4. Execute trade through broker if position needs adjustment

        Args:
            bar: Dictionary with bar data including signal and price.
                Required keys: close, datetime, signal_column.

        Returns:
            Trade information if a trade was executed, None otherwise.
        """
        # Kill switch check
        if self.config.kill_switch_active:
            log.debug("Kill switch active — skipping bar")
            return None

        signal = bar.get(self.config.signal_column)
        close = bar.get("close")
        datetime_val = bar.get("datetime")

        if signal is None or close is None:
            return None

        # Validate price
        if not isinstance(close, (int, float)) or close <= 0:
            return None

        # Adopt the broker's actual position (async fills on live brokers)
        self._reconcile_position()

        # Check stop loss / take profit first
        self.check_stop_loss(close)

        # Backoff after repeated unfilled orders (rejected/margin-blocked)
        if self._order_pause > 0:
            self._order_pause -= 1
            log.debug(f"{self.config.ticker}: order pause active ({self._order_pause} bars left)")
            return None

        # One outstanding live order at a time — wait for its fill to be
        # reconciled before placing anything else (prevents order stacking).
        if self._pending_order is not None:
            log.debug(f"{self.config.ticker}: awaiting fill of {self._pending_order['order_id']}")
            return None

        # Calculate target position based on signal strength
        target_position = int(signal * self.config.max_position_shares)

        # How many shares do we need to trade?
        shares_to_trade = target_position - self._position

        # Execute trade if there's a meaningful change
        if abs(shares_to_trade) > 0:
            trade = self._execute_trade(shares_to_trade, close, datetime_val, signal)
            if trade:
                # Record equity
                equity = self.broker.get_portfolio_value()
                self._equity_curve.append({
                    "datetime": datetime_val,
                    "portfolio_value": equity,
                })

                # Check daily loss circuit breaker
                self._check_daily_loss_limit(equity)

                return trade

        return None

    def _reconcile_position(self, record_trade: bool = True, count_strike: bool = True) -> Optional[dict]:
        """Adopt the broker's actual position as truth.

        Live brokers (IBKR) fill asynchronously: ``place_market_order``
        returns PENDING and the fill arrives via the gateway's position
        updates. The synchronous FILLED check in ``_execute_trade`` never
        fires for them, so without reconciliation the engine re-submits the
        full target every bar (pitfall 30 — 27 same-side orders on Aug 4,
        2026). This adopts the broker book, resolves pending orders, records
        fills, and enforces a pause after repeated unfilled orders.

        Args:
            record_trade: Record a trade entry for a detected position change.
                False for session-start adoption of a pre-existing position.
            count_strike: Count an unresolved pending order as a rejection
                strike. False for the immediate post-placement reconcile —
                an order placed this same call has not yet had a bar to fill.

        Returns:
            Trade record dict when a position change was detected, else None.
        """
        if not hasattr(self.broker, "get_positions"):
            return None
        try:
            pos = self.broker.get_positions().get(self.config.ticker)
        except Exception as e:
            log.debug(f"Position reconciliation failed: {e}")
            return None

        actual = int(pos.quantity) if pos else 0

        if actual == self._position:
            # Unresolved pending order across bars means it likely failed
            # (rejected, margin-blocked). Strike toward a pause.
            if self._pending_order is not None and count_strike:
                self._reject_strikes += 1
                if self._reject_strikes >= 3:
                    log.warning(
                        f"{self.config.ticker}: 3 consecutive orders without a "
                        "fill — pausing new orders for 5 bars"
                    )
                    self._order_pause = 5
                    self._pending_order = None
                    self._reject_strikes = 0
            return None

        # Position moved — record the fill (delta vs our previous view).
        delta = actual - self._position
        pending = self._pending_order or {}
        fill_price = pos.avg_cost if pos and pos.avg_cost > 0 else pending.get("price", 0.0)
        commission = abs(delta) * self.config.commission_per_share

        trade_record = None
        if record_trade:
            trade_record = {
                "datetime": pending.get("datetime") or datetime.now(timezone.utc),
                "shares": delta,
                "price": fill_price,
                "commission": commission,
                "signal": pending.get("signal", 0.0),
                "position_after": actual,
                "order_id": pending.get("order_id", ""),
                "pnl": 0.0,
                "reconciled": True,
            }
            self._trades.append(trade_record)
        self._position = actual
        self._entry_price = (pos.avg_cost if pos else 0.0) if actual != 0 else 0.0
        self._pending_order = None
        self._reject_strikes = 0
        return trade_record

    def _execute_trade(
        self,
        shares: int,
        price: float,
        datetime_val: Optional[datetime],
        signal: float,
    ) -> Optional[dict]:
        """Execute a trade through the broker.

        Args:
            shares: Number of shares to trade (positive = buy, negative = sell).
            price: Current bar price.
            datetime_val: Bar timestamp.
            signal: Signal strength that triggered this trade.

        Returns:
            Trade record dict, or None if the broker rejected the trade.
        """
        side = Side.BUY if shares > 0 else Side.SELL
        qty = abs(shares)

        try:
            order = self.broker.place_market_order(
                ticker=self.config.ticker,
                side=side,
                quantity=int(qty),
                signal_strength=abs(signal),
                fill_price=price,
            )

            # Update internal position tracking
            if order.status == OrderStatus.FILLED or order.filled_qty > 0:
                fill_qty = order.filled_qty if order.filled_qty > 0 else qty
                fill_price = order.avg_fill_price or price

                if side == Side.BUY:
                    self._position += fill_qty
                    self._entry_price = fill_price
                else:
                    # SELL: reduce position
                    self._position -= fill_qty
                    if self._position == 0:
                        self._entry_price = 0.0

                commission = fill_qty * self.config.commission_per_share

                # Calculate P&L for this trade
                trade_pnl = 0.0
                if side == Side.SELL and self._trades:
                    # Find the matching buy
                    for prev_trade in reversed(self._trades):
                        if prev_trade.get("shares", 0) > 0:
                            trade_pnl = (fill_price - prev_trade["price"]) * fill_qty - commission
                            break

                trade_record = {
                    "datetime": datetime_val,
                    "shares": shares,
                    "price": fill_price,
                    "commission": commission,
                    "signal": signal,
                    "position_after": self._position,
                    "order_id": order.order_id,
                    "pnl": trade_pnl,
                }
                self._trades.append(trade_record)

                if trade_pnl > 0:
                    self._winning_pnl += trade_pnl
                else:
                    self._losing_pnl += abs(trade_pnl)

                return trade_record
            else:
                # Live broker: the order is async (PENDING) — the synchronous
                # FILLED check above never fires for IBBroker. Remember the
                # order; _reconcile_position() adopts the actual fill at the
                # next bar (or immediately, if the gateway already reflects it).
                self._pending_order = {
                    "shares": shares,
                    "price": price,
                    "signal": signal,
                    "datetime": datetime_val,
                    "order_id": order.order_id,
                }
                # Best-effort immediate adopt (paper fills land in ms); do not
                # count this same-call check as a rejection strike.
                return self._reconcile_position(count_strike=False)

        except (ValueError, RuntimeError) as e:
            log.warning(f"Trade execution failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Risk management
    # ------------------------------------------------------------------

    def check_stop_loss(self, current_price: float) -> bool:
        """Check if stop loss or take profit is triggered.

        Args:
            current_price: Current market price.

        Returns:
            True if position was closed, False otherwise.
        """
        if self._position == 0:
            return False

        if self._entry_price <= 0:
            return False

        # Calculate P&L percentage
        pnl_pct = (current_price - self._entry_price) / self._entry_price

        # For long positions: SL if price drops, TP if price rises
        # For short positions: inverted logic
        if self._position > 0:
            if pnl_pct <= -self.config.stop_loss_pct:
                self._close_position(current_price, "stop_loss")
                log.warning(f"Stop loss triggered: {pnl_pct:.2%}")
                return True
            if pnl_pct >= self.config.take_profit_pct:
                self._close_position(current_price, "take_profit")
                log.info(f"Take profit triggered: {pnl_pct:.2%}")
                return True
        else:
            # Short position
            if pnl_pct >= self.config.stop_loss_pct:
                self._close_position(current_price, "stop_loss")
                return True
            if pnl_pct <= -self.config.take_profit_pct:
                self._close_position(current_price, "take_profit")
                return True

        return False

    def _close_position(self, price: float, reason: str) -> None:
        """Close current position through the broker.

        Args:
            price: Price to close at.
            reason: Reason for closing (stop_loss, take_profit, end_of_session).
        """
        if self._position == 0:
            return

        side = Side.SELL if self._position > 0 else Side.BUY
        qty = abs(self._position)

        try:
            order = self.broker.place_market_order(
                ticker=self.config.ticker,
                side=side,
                quantity=int(qty),
                fill_price=price,
            )

            if order.status == OrderStatus.FILLED or order.filled_qty > 0:
                fill_price = order.avg_fill_price or price
                commission = qty * self.config.commission_per_share

                # Calculate realized P&L
                pnl = (fill_price - self._entry_price) * self._position - commission

                self._trades.append({
                    "datetime": datetime.now(timezone.utc),
                    "shares": -self._position,
                    "price": fill_price,
                    "commission": commission,
                    "reason": reason,
                    "position_after": 0,
                    "pnl": pnl,
                })

                if pnl > 0:
                    self._winning_pnl += pnl
                else:
                    self._losing_pnl += abs(pnl)

                self._position = 0
                self._entry_price = 0.0

                log.info(
                    f"Position closed ({reason}): price=${fill_price:.2f}, "
                    f"P&L=${pnl:.2f}"
                )

        except Exception as e:
            log.error(f"Failed to close position: {e}")

    def _check_daily_loss_limit(self, current_equity: float) -> None:
        """Check if daily loss exceeds the circuit breaker limit.

        Args:
            current_equity: Current portfolio equity.
        """
        if current_equity <= 0:
            return

        daily_loss_pct = (self._peak_equity - current_equity) / self._peak_equity
        if daily_loss_pct >= self.config.max_daily_loss_pct:
            self.config.kill_switch_active = True
            log.error(
                f"Daily loss limit triggered: {daily_loss_pct:.2%} >= "
                f"{self.config.max_daily_loss_pct:.2%} — kill switch ON"
            )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_equity(self, current_price: Optional[float] = None) -> float:
        """Get current equity from the broker.

        Args:
            current_price: Optional price to include unrealized P&L.

        Returns:
            Current equity value.
        """
        return self.broker.get_portfolio_value()

    @property
    def position(self) -> int:
        """Current position in shares."""
        return self._position

    @property
    def entry_price(self) -> float:
        """Entry price for current position."""
        return self._entry_price

    @property
    def session_active(self) -> bool:
        """Whether a session is currently active."""
        return self._session_active

    def get_session_metrics(self) -> dict:
        """Calculate session-level performance metrics.

        Returns:
            Dict with comprehensive session metrics.
        """
        if not self._equity_curve:
            return {}

        equity = pl.DataFrame(self._equity_curve)
        initial_value = self.config.initial_cash
        final_value = float(equity["portfolio_value"].max()) if len(equity) > 0 else initial_value

        # Total return
        total_return = (final_value - initial_value) / initial_value * 100 if initial_value > 0 else 0
        total_pnl = final_value - initial_value

        # Win rate
        winning_trades = sum(1 for t in self._trades if t.get("pnl", 0) > 0)
        losing_trades = sum(1 for t in self._trades if t.get("pnl", 0) < 0)
        total_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

        # Average trade P&L
        avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0

        # Profit factor
        profit_factor = (
            self._winning_pnl / self._losing_pnl
            if self._losing_pnl > 0 else float("inf")
        )

        # Sharpe ratio (annualized from intraday returns)
        returns = equity["portfolio_value"].pct_change().drop_nulls()
        if len(returns) > 1 and returns.std() > 0:
            # For intraday, annualize based on bars per day * trading days
            bars_per_day = 390  # 6.5 hours of 1-min bars
            sharpe = float(
                returns.mean() / returns.std() * (bars_per_day * 252 ** 0.5)
            )
        else:
            sharpe = 0.0

        # Max drawdown
        if len(equity) > 0:
            peak = equity["portfolio_value"].cum_max()
            drawdown = (equity["portfolio_value"] - peak) / peak
            max_drawdown = float(drawdown.min() * 100)
        else:
            max_drawdown = 0.0

        # Total commissions
        total_commissions = sum(t.get("commission", 0) for t in self._trades)

        return {
            "initial_equity": initial_value,
            "final_equity": final_value,
            "total_return_pct": total_return,
            "total_pnl": total_pnl,
            "total_trades": len(self._trades),
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate_pct": win_rate,
            "avg_trade_pnl": avg_trade_pnl,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_drawdown,
            "total_commissions": total_commissions,
            "total_bars": len(self._equity_curve),
            "position": self._position,
        }

    def get_equity_curve(self) -> pl.DataFrame:
        """Get the equity curve as a DataFrame."""
        return pl.DataFrame(self._equity_curve)

    def get_trades(self) -> list[dict]:
        """Get all trades executed during this session."""
        return list(self._trades)

    def reset(self) -> None:
        """Reset engine state (for backtesting multiple sessions)."""
        self._position = 0
        self._entry_price = 0.0
        self._peak_equity = self.config.initial_cash
        self._session_active = False
        self._session_date = None
        self._trades = []
        self._equity_curve = []
        self._fills = []
        self._daily_pnl = 0.0
        self._winning_pnl = 0.0
        self._losing_pnl = 0.0
        self.config.kill_switch_active = False