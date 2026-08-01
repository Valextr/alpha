"""Integration tests for Phase 7: execution + monitoring + IBKR bridge.

Tests:
- MonitoringBridge: model conversion, metrics update, alert firing
- ExecutionEngine with monitoring_bridge attached
- IBBroker: Broker ABC compliance (structural, no live connection)
- End-to-end paper trading pipeline with monitoring
"""

import pytest
from datetime import datetime, timezone

import polars as pl

from src.execution.broker import Broker, PaperBroker
from src.execution.engine import ExecutionConfig, ExecutionEngine
from src.execution.ib_broker import IBConfig, IBBroker
from src.execution.models import Fill, Order, OrderStatus, OrderType, Position, Side
from src.execution.monitoring_bridge import (
    MonitoringBridge,
    exec_fill_to_monitoring,
    exec_position_to_monitoring,
)
from src.monitoring import (
    AlertManager,
    AlertSeverity,
    AlertType,
    MonitoringConfig,
)


class TestMonitoringBridgeConversion:
    """Test model conversion between execution and monitoring."""

    def test_exec_position_to_monitoring(self):
        pos = Position(
            ticker="AAPL",
            quantity=100,
            avg_cost=150.0,
            current_price=155.0,
            unrealized_pnl=500.0,
            realized_pnl=0.0,
        )
        result = exec_position_to_monitoring(pos)
        assert result["ticker"] == "AAPL"
        assert result["quantity"] == 100
        assert result["avg_cost"] == 150.0
        assert result["current_price"] == 155.0
        assert result["unrealized_pnl"] == 500.0

    def test_exec_fill_to_monitoring_buy(self):
        fill = Fill(
            order_id="ord1",
            ticker="MSFT",
            side=Side.BUY,
            quantity=50,
            price=300.0,
            commission=0.25,
        )
        result = exec_fill_to_monitoring(fill)
        assert result["ticker"] == "MSFT"
        assert result["side"].value == "BUY"
        assert result["quantity"] == 50
        assert result["price"] == 300.0

    def test_exec_fill_to_monitoring_sell(self):
        fill = Fill(
            order_id="ord2",
            ticker="GOOGL",
            side=Side.SELL,
            quantity=25,
            price=140.0,
            commission=0.125,
        )
        result = exec_fill_to_monitoring(fill)
        assert result["side"].value == "SELL"


class TestMonitoringBridge:
    """Test MonitoringBridge initialization and updates."""

    def test_bridge_creation(self):
        bridge = MonitoringBridge(initial_balance=100_000.0)
        assert bridge.metrics.initial_balance == 100_000.0
        assert isinstance(bridge.alerts, AlertManager)

    def test_bridge_custom_config(self):
        config = MonitoringConfig()
        config.alerts.kill_switch_drawdown = 0.10
        bridge = MonitoringBridge(initial_balance=50_000.0, monitoring_config=config)
        assert bridge.metrics.initial_balance == 50_000.0
        assert bridge.config.alerts.kill_switch_drawdown == 0.10

    def test_update_from_engine_no_positions(self):
        bridge = MonitoringBridge(initial_balance=100_000.0)
        alerts = bridge.update_from_engine(
            positions={},
            fills=[],
            portfolio_value=100_000.0,
            daily_pnl=0.0,
            peak_equity=100_000.0,
        )
        assert alerts == []

    def test_update_from_engine_with_fills(self):
        bridge = MonitoringBridge(initial_balance=100_000.0)

        fill = Fill(
            order_id="ord1",
            ticker="AAPL",
            side=Side.BUY,
            quantity=100,
            price=150.0,
            commission=0.5,
        )

        pos = Position(
            ticker="AAPL",
            quantity=100,
            avg_cost=150.0,
            current_price=150.0,
        )

        alerts = bridge.update_from_engine(
            positions={"AAPL": pos},
            fills=[fill],
            portfolio_value=100_000.0,
            daily_pnl=0.0,
            peak_equity=100_000.0,
        )
        assert bridge.metrics.num_trades == 1

    def test_update_from_engine_deduplicates_fills(self):
        bridge = MonitoringBridge(initial_balance=100_000.0)

        fill = Fill(
            order_id="ord1",
            ticker="AAPL",
            side=Side.BUY,
            quantity=100,
            price=150.0,
            commission=0.5,
        )

        # Call twice with same fill
        bridge.update_from_engine(
            positions={},
            fills=[fill],
            portfolio_value=100_000.0,
        )
        bridge.update_from_engine(
            positions={},
            fills=[fill],
            portfolio_value=100_000.0,
        )

        # Should only count once
        assert bridge.metrics.num_trades == 1

    def test_drawdown_alert(self):
        config = MonitoringConfig()
        config.alerts.max_drawdown = 0.05
        bridge = MonitoringBridge(initial_balance=100_000.0, monitoring_config=config)

        alerts = bridge.update_from_engine(
            positions={},
            fills=[],
            portfolio_value=90_000.0,
            daily_pnl=0.0,
            peak_equity=100_000.0,
        )
        # 10% drawdown exceeds 5% threshold
        assert len(alerts) > 0
        assert alerts[0].alert_type == AlertType.DRAWDOWN_BREACH

    def test_kill_switch_alert(self):
        config = MonitoringConfig()
        config.alerts.kill_switch_drawdown = 0.10
        bridge = MonitoringBridge(initial_balance=100_000.0, monitoring_config=config)

        alerts = bridge.update_from_engine(
            positions={},
            fills=[],
            portfolio_value=80_000.0,
            daily_pnl=0.0,
            peak_equity=100_000.0,
        )
        # 20% drawdown exceeds 10% kill switch
        assert len(alerts) > 0
        assert alerts[0].alert_type == AlertType.KILL_SWITCH
        assert alerts[0].severity == AlertSeverity.CRITICAL

    def test_position_limit_alert(self):
        config = MonitoringConfig()
        config.alerts.max_position_pct = 0.20
        bridge = MonitoringBridge(initial_balance=100_000.0, monitoring_config=config)

        pos = Position(
            ticker="AAPL",
            quantity=600,
            avg_cost=150.0,
            current_price=150.0,
        )

        alerts = bridge.update_from_engine(
            positions={"AAPL": pos},
            fills=[],
            portfolio_value=100_000.0,
            daily_pnl=0.0,
            peak_equity=100_000.0,
        )
        # 600 * 150 = 90,000 = 90% of portfolio, exceeds 20% limit
        assert len(alerts) > 0
        position_alerts = [a for a in alerts if a.alert_type == AlertType.POSITION_LIMIT_BREACH]
        assert len(position_alerts) > 0

    def test_get_metrics(self):
        bridge = MonitoringBridge(initial_balance=100_000.0)
        metrics = bridge.get_metrics()
        assert "balance" in metrics
        assert metrics["balance"] == 100_000.0

    def test_record_end_of_day(self):
        bridge = MonitoringBridge(initial_balance=100_000.0)
        summary = bridge.record_end_of_day()
        assert summary.starting_balance == 100_000.0
        assert summary.ending_balance == 100_000.0
        assert summary.daily_pnl == 0.0


class TestIBBrokerABC:
    """Test IBBroker implements Broker ABC (structural, no live connection)."""

    def test_ib_broker_is_broker_subclass(self):
        assert issubclass(IBBroker, Broker)

    def test_ib_broker_has_required_methods(self):
        broker = IBBroker()
        assert callable(broker.connect)
        assert callable(broker.disconnect)
        assert callable(broker.is_connected)
        assert callable(broker.submit_order)
        assert callable(broker.cancel_order)
        assert callable(broker.get_positions)
        assert callable(broker.get_portfolio_value)
        assert callable(broker.get_current_price)
        assert callable(broker.place_market_order)
        assert callable(broker.place_limit_order)

    def test_ib_broker_default_config(self):
        broker = IBBroker()
        assert broker._client.config.host == "127.0.0.1"
        assert broker._client.config.port == 4002

    def test_ib_broker_custom_config(self):
        config = IBConfig(tws_host="localhost", tws_port=4003, client_id=99)
        broker = IBBroker(config=config)
        assert broker._client.config.host == "localhost"
        assert broker._client.config.port == 4003
        assert broker._client.config.client_id == 99

    def test_ib_broker_config_conversion(self):
        config = IBConfig(paper_trading=False)
        broker = IBBroker(config=config)
        assert broker._client.config.trade_mode == "LIVE"

    def test_ib_broker_not_connected(self):
        broker = IBBroker()
        assert not broker.is_connected()

    def test_ib_broker_submit_requires_connection(self):
        broker = IBBroker()
        order = Order(
            ticker="AAPL",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
        )
        with pytest.raises(RuntimeError, match="not connected"):
            broker.submit_order(order)

    def test_ib_broker_get_positions_returns_cache_when_disconnected(self):
        broker = IBBroker()
        positions = broker.get_positions()
        assert isinstance(positions, dict)


class TestExecutionEngineWithMonitoring:
    """Test ExecutionEngine with monitoring_bridge attached."""

    def test_engine_accepts_monitoring_bridge(self):
        config = ExecutionConfig(initial_cash=100_000.0)
        bridge = MonitoringBridge(initial_balance=100_000.0)
        engine = ExecutionEngine(config, monitoring_bridge=bridge)
        assert engine.monitoring is bridge

    def test_engine_without_monitoring(self):
        config = ExecutionConfig(initial_cash=100_000.0)
        engine = ExecutionEngine(config)
        assert engine.monitoring is None

    def test_update_monitoring_noop_when_none(self):
        config = ExecutionConfig(initial_cash=100_000.0)
        engine = ExecutionEngine(config)
        engine.start()
        # Should not raise
        engine.update_monitoring()

    def test_update_monitoring_with_bridge(self):
        config = ExecutionConfig(initial_cash=100_000.0)
        bridge = MonitoringBridge(initial_balance=100_000.0)
        engine = ExecutionEngine(config, monitoring_bridge=bridge)
        engine.start()
        engine.update_monitoring()
        metrics = bridge.get_metrics()
        assert metrics["balance"] == 100_000.0

    def test_end_of_day_updates_monitoring(self):
        config = ExecutionConfig(initial_cash=100_000.0)
        bridge = MonitoringBridge(initial_balance=100_000.0)
        engine = ExecutionEngine(config, monitoring_bridge=bridge)
        engine.start()
        engine.record_end_of_day()
        # Monitoring should have recorded the daily return
        summary = bridge.metrics.daily_summary()
        assert summary.daily_pnl == 0.0


class TestEndToEndPaperTrading:
    """Full pipeline: signals -> engine -> monitoring -> alerts."""

    def _make_price_source(self):
        """Create a price source for PaperBroker."""
        return {
            "AAPL": pl.DataFrame({
                "date": [datetime(2025, 1, 1).date()],
                "close": [150.0],
            }),
            "MSFT": pl.DataFrame({
                "date": [datetime(2025, 1, 1).date()],
                "close": [300.0],
            }),
        }

    def _make_signal_df(self):
        """Create a signal DataFrame."""
        return pl.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "date": [datetime(2025, 1, 1)] * 2,
            "ensemble_signal": [0.5, -0.3],
        })

    def test_full_pipeline(self):
        """Signals -> targets -> execution -> monitoring -> metrics."""
        config = ExecutionConfig(
            initial_cash=100_000.0,
            max_position_weight=0.5,
            max_order_size_pct=0.5,
        )
        bridge = MonitoringBridge(initial_balance=100_000.0)
        price_source = self._make_price_source()

        broker = PaperBroker(
            initial_cash=100_000.0,
            price_source=price_source,
        )
        engine = ExecutionEngine(config, broker=broker, monitoring_bridge=bridge)
        engine.start()

        # Generate targets from signals
        signal_df = self._make_signal_df()
        targets = engine.generate_targets_from_signals(signal_df)
        assert len(targets) > 0

        # Execute targets
        orders = engine.execute_targets(targets, price_source=price_source)
        assert len(orders) > 0

        # Update monitoring
        engine.update_monitoring()

        # Check metrics
        metrics = bridge.get_metrics()
        assert metrics["num_trades"] > 0
        assert metrics["num_positions"] > 0

        engine.stop()

    def test_pipeline_triggers_alerts_on_large_position(self):
        """When a single position exceeds the limit, an alert fires."""
        config = ExecutionConfig(
            initial_cash=100_000.0,
            max_position_weight=1.0,
            max_order_size_pct=1.0,
        )
        mon_config = MonitoringConfig()
        mon_config.alerts.max_position_pct = 0.05  # 5% threshold
        bridge = MonitoringBridge(
            initial_balance=100_000.0,
            monitoring_config=mon_config,
        )
        price_source = self._make_price_source()

        broker = PaperBroker(
            initial_cash=100_000.0,
            price_source=price_source,
        )
        engine = ExecutionEngine(config, broker=broker, monitoring_bridge=bridge)
        engine.start()

        # Force a large target
        targets = {"AAPL": 0.8}  # 80% of portfolio
        orders = engine.execute_targets(targets, price_source=price_source)
        engine.update_monitoring()

        # Should trigger position limit alert
        alerts = bridge.get_alerts()
        position_alerts = [a for a in alerts if a.alert_type == AlertType.POSITION_LIMIT_BREACH]
        assert len(position_alerts) > 0

        engine.stop()

    def test_pipeline_end_of_day(self):
        """End-of-day recording updates monitoring summary."""
        config = ExecutionConfig(
            initial_cash=100_000.0,
            max_position_weight=0.5,
            max_order_size_pct=0.5,
        )
        bridge = MonitoringBridge(initial_balance=100_000.0)
        price_source = self._make_price_source()

        broker = PaperBroker(
            initial_cash=100_000.0,
            price_source=price_source,
        )
        engine = ExecutionEngine(config, broker=broker, monitoring_bridge=bridge)
        engine.start()

        # Execute a trade
        targets = {"AAPL": 0.1}
        engine.execute_targets(targets, price_source=price_source)
        engine.update_monitoring()

        # Record end of day
        daily_pnl = engine.record_end_of_day()
        summary = bridge.metrics.daily_summary()

        assert summary.positions_count > 0
        assert isinstance(daily_pnl, float)

        engine.stop()


class TestIBConfigConversion:
    """Test IBConfig to IBKRConfig conversion."""

    def test_paper_trading_conversion(self):
        config = IBConfig(paper_trading=True, tws_port=4002)
        ibkr_config = config.to_ibkr_config()
        assert ibkr_config.trade_mode == "PAPER"
        assert ibkr_config.port == 4002

    def test_live_trading_conversion(self):
        config = IBConfig(paper_trading=False)
        ibkr_config = config.to_ibkr_config()
        assert ibkr_config.trade_mode == "LIVE"

    def test_custom_host_port(self):
        config = IBConfig(tws_host="192.168.1.100", tws_port=4004, client_id=42)
        ibkr_config = config.to_ibkr_config()
        assert ibkr_config.host == "192.168.1.100"
        assert ibkr_config.port == 4004
        assert ibkr_config.client_id == 42