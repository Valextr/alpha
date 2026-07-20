"""Tests for execution engine."""

import pytest
import polars as pl
from datetime import date

from src.execution.models import Order, Position, Fill, Side, OrderType, OrderStatus
from src.execution.broker import PaperBroker
from src.execution.risk import RiskConfig, RiskGuard
from src.execution.engine import ExecutionEngine, ExecutionConfig


@pytest.fixture
def price_source():
    data = []
    for ticker in ["AAPL", "MSFT", "GOOGL"]:
        for i in range(10):
            d = date(2024, 1, i + 1)
            price = 150.0 + i * 0.5
            data.append({"date": d, "close": price, "ticker": ticker})
    return {
        "AAPL": pl.DataFrame(data[:10]),
        "MSFT": pl.DataFrame(data[10:20]),
        "GOOGL": pl.DataFrame(data[20:30]),
    }


class TestPaperBroker:
    def test_connect_disconnect(self):
        broker = PaperBroker(initial_cash=100_000)
        assert not broker.is_connected()
        broker.connect()
        assert broker.is_connected()
        broker.disconnect()
        assert not broker.is_connected()

    def test_initial_portfolio_value(self):
        broker = PaperBroker(initial_cash=100_000)
        broker.connect()
        assert broker.get_portfolio_value() == 100_000.0
        broker.disconnect()

    def test_buy_order(self, price_source):
        broker = PaperBroker(initial_cash=100_000, price_source=price_source)
        broker.connect()

        order = broker.place_market_order("AAPL", Side.BUY, 10)
        assert order.status == OrderStatus.FILLED
        assert order.filled_qty == 10

        positions = broker.get_positions()
        assert "AAPL" in positions
        assert positions["AAPL"].quantity == 10

    def test_sell_order(self, price_source):
        broker = PaperBroker(initial_cash=100_000, price_source=price_source)
        broker.connect()

        broker.place_market_order("AAPL", Side.BUY, 10)
        order = broker.place_market_order("AAPL", Side.SELL, 5)
        assert order.status == OrderStatus.FILLED

        positions = broker.get_positions()
        assert positions["AAPL"].quantity == 5

    def test_full_close_position(self, price_source):
        broker = PaperBroker(initial_cash=100_000, price_source=price_source)
        broker.connect()

        broker.place_market_order("AAPL", Side.BUY, 10)
        broker.place_market_order("AAPL", Side.SELL, 10)

        positions = broker.get_positions()
        assert "AAPL" not in positions

    def test_commission_deduction(self, price_source):
        broker = PaperBroker(
            initial_cash=100_000,
            commission_per_share=0.01,
            price_source=price_source,
        )
        broker.connect()

        initial = broker.get_portfolio_value()
        broker.place_market_order("AAPL", Side.BUY, 100)
        final = broker.get_portfolio_value()

        assert final < initial

    def test_limit_order_fill(self, price_source):
        broker = PaperBroker(initial_cash=100_000, price_source=price_source)
        broker.connect()

        current = broker.get_current_price("AAPL")
        assert current is not None

        order = broker.place_limit_order("AAPL", Side.BUY, 5, limit_price=current + 10)
        assert order.status == OrderStatus.FILLED

    def test_limit_order_pending(self, price_source):
        broker = PaperBroker(initial_cash=100_000, price_source=price_source)
        broker.connect()

        current = broker.get_current_price("AAPL")
        order = broker.place_limit_order("AAPL", Side.BUY, 5, limit_price=current - 10)
        assert order.status == OrderStatus.PENDING

    def test_cancel_order(self, price_source):
        broker = PaperBroker(initial_cash=100_000, price_source=price_source)
        broker.connect()

        current = broker.get_current_price("AAPL")
        order = broker.place_limit_order("AAPL", Side.BUY, 5, limit_price=current - 10)
        assert order.status == OrderStatus.PENDING

        cancelled = broker.cancel_order(order.order_id)
        assert cancelled

    def test_no_price_available(self):
        broker = PaperBroker(initial_cash=100_000)
        broker.connect()

        with pytest.raises(ValueError, match="No price available"):
            broker.place_market_order("UNKNOWN", Side.BUY, 10)

    def test_not_connected_error(self):
        broker = PaperBroker(initial_cash=100_000)

        with pytest.raises(RuntimeError, match="not connected"):
            broker.place_market_order("AAPL", Side.BUY, 10)


class TestRiskGuard:
    @pytest.fixture
    def risk_config(self):
        return RiskConfig(
            max_positions=5,
            max_position_weight=0.2,
            max_leverage=1.0,
            daily_loss_limit_pct=0.05,
            max_drawdown_pct=0.15,
            min_cash_reserve=0.05,
            max_order_size_pct=0.1,
        )

    @pytest.fixture
    def guard(self, risk_config):
        return RiskGuard(
            risk_config,
            portfolio_value=100_000,
            positions={},
            cash=100_000,
            daily_pnl=0,
            peak_equity=100_000,
        )

    def test_approve_normal_order(self, guard):
        order = Order(
            ticker="AAPL",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=10,
            avg_fill_price=150.0,
        )
        approved, reason = guard.validate_order(order)
        assert approved

    def test_kill_switch_blocks(self, guard, risk_config):
        risk_config.activate_kill_switch()
        order = Order(
            ticker="AAPL",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=10,
            avg_fill_price=150.0,
        )
        approved, reason = guard.validate_order(order)
        assert not approved
        assert "KILL_SWITCH" in reason

    def test_daily_loss_limit(self, risk_config):
        guard = RiskGuard(
            risk_config,
            portfolio_value=100_000,
            positions={},
            cash=100_000,
            daily_pnl=-6_000,
            peak_equity=100_000,
        )
        order = Order(
            ticker="AAPL",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=10,
            avg_fill_price=150.0,
        )
        approved, reason = guard.validate_order(order)
        assert not approved
        assert "DAILY_LOSS" in reason
        assert risk_config.kill_switch_active

    def test_drawdown_limit(self, risk_config):
        guard = RiskGuard(
            risk_config,
            portfolio_value=80_000,
            positions={},
            cash=80_000,
            daily_pnl=-2_000,
            peak_equity=100_000,
        )
        order = Order(
            ticker="AAPL",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=10,
            avg_fill_price=150.0,
        )
        approved, reason = guard.validate_order(order)
        assert not approved
        assert "DRAWDOWN" in reason
        assert risk_config.kill_switch_active

    def test_max_positions(self, risk_config):
        positions = {
            f"TICKER{i}": Position(
                ticker=f"TICKER{i}", quantity=10, avg_cost=100.0, current_price=100.0
            )
            for i in range(5)
        }
        guard = RiskGuard(
            risk_config,
            portfolio_value=100_000,
            positions=positions,
            cash=10_000,
            daily_pnl=0,
            peak_equity=100_000,
        )
        order = Order(
            ticker="NEW_TICKER",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=10,
            avg_fill_price=100.0,
        )
        approved, reason = guard.validate_order(order)
        assert not approved
        assert "MAX_POSITIONS" in reason


class TestExecutionEngine:
    def test_start_stop(self):
        config = ExecutionConfig(initial_cash=100_000)
        engine = ExecutionEngine(config)

        assert not engine.is_running
        engine.start()
        assert engine.is_running
        engine.stop()
        assert not engine.is_running

    def test_portfolio_value(self):
        config = ExecutionConfig(initial_cash=100_000)
        engine = ExecutionEngine(config)
        engine.start()

        assert engine.get_portfolio_value() == 100_000.0
        engine.stop()

    def test_generate_targets_from_signals(self):
        df = pl.DataFrame({
            "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN"],
            "date": [date(2024, 1, 1)] * 4,
            "ensemble_signal": [0.5, -0.3, 0.2, -0.1],
        })

        config = ExecutionConfig()
        engine = ExecutionEngine(config)

        targets = engine.generate_targets_from_signals(df)
        assert "AAPL" in targets
        assert "MSFT" in targets
        assert abs(targets["AAPL"]) > abs(targets["AMZN"])

    def test_filter_weak_signals(self):
        df = pl.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "date": [date(2024, 1, 1)] * 2,
            "ensemble_signal": [0.05, 0.5],
        })

        config = ExecutionConfig()
        engine = ExecutionEngine(config)

        targets = engine.generate_targets_from_signals(df, min_signal_strength=0.1)
        assert "AAPL" not in targets
        assert "MSFT" in targets

    def test_top_n_signals(self):
        df = pl.DataFrame({
            "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN"],
            "date": [date(2024, 1, 1)] * 4,
            "ensemble_signal": [0.5, -0.3, 0.2, -0.1],
        })

        config = ExecutionConfig()
        engine = ExecutionEngine(config)

        targets = engine.generate_targets_from_signals(df, top_n=2)
        assert len(targets) <= 2

    def test_execute_targets_empty_when_not_running(self):
        config = ExecutionConfig()
        engine = ExecutionEngine(config)

        with pytest.raises(RuntimeError, match="not running"):
            engine.execute_targets({"AAPL": 0.5})

    def test_execute_with_price_source(self, price_source):
        config = ExecutionConfig(
            initial_cash=100_000,
            max_order_size_pct=0.5,
            max_position_weight=0.5,
        )
        engine = ExecutionEngine(config)
        engine.start()

        targets = {"AAPL": 0.3}
        orders = engine.execute_targets(targets, price_source=price_source)

        assert len(orders) > 0
        assert orders[0].ticker == "AAPL"

        engine.stop()

    def test_snapshot(self):
        config = ExecutionConfig(initial_cash=100_000)
        engine = ExecutionEngine(config)
        engine.start()

        snap = engine.get_snapshot()
        assert snap.total_equity == 100_000.0
        assert snap.num_positions == 0

        engine.stop()

    def test_end_of_day(self):
        config = ExecutionConfig(initial_cash=100_000)
        engine = ExecutionEngine(config)
        engine.start()

        pnl = engine.record_end_of_day()
        assert pnl == 0.0

        engine.stop()

    def test_emergency_stop(self):
        config = ExecutionConfig(initial_cash=100_000)
        engine = ExecutionEngine(config)
        engine.start()

        engine.emergency_stop()
        assert engine.kill_switch_active
        assert not engine.is_running

    def test_kill_switch_blocks_execution(self):
        config = ExecutionConfig(initial_cash=100_000)
        engine = ExecutionEngine(config)
        engine.start()

        engine.risk_config.activate_kill_switch()
        orders = engine.execute_targets({"AAPL": 0.3})
        assert len(orders) == 0

        engine.stop()