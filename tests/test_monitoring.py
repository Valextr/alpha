"""Tests for the monitoring module."""

from __future__ import annotations

import tempfile
import time
from datetime import datetime, timezone

import polars as pl
import pytest

from src.monitoring import (
    Alert,
    AlertManager,
    AlertSeverity,
    AlertType,
    Dashboard,
    DailySummary,
    Fill,
    MonitoringConfig,
    MetricsTracker,
    OrderSide,
    Position,
    ReportGenerator,
    TradeEvent,
)


# --- Data models ---


class TestDataModels:
    def test_position_market_value(self):
        pos = Position(
            ticker="AAPL",
            quantity=100,
            avg_cost=150.0,
            current_price=155.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
        )
        assert pos.market_value == 15500.0
        assert pos.notional == 15500.0

    def test_position_short(self):
        pos = Position(
            ticker="AAPL",
            quantity=-50,
            avg_cost=150.0,
            current_price=155.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
        )
        assert pos.market_value == -7750.0
        assert pos.notional == 7750.0

    def test_fill_creation(self):
        fill = Fill(
            ticker="SPY",
            side=OrderSide.BUY,
            quantity=10,
            price=450.0,
            order_id="test-order-1",
        )
        assert len(fill.fill_id) == 12
        assert isinstance(fill.timestamp, datetime)

    def test_trade_event(self):
        event = TradeEvent(
            event_type="FILL",
            ticker="SPY",
            fill=Fill(
                ticker="SPY",
                side=OrderSide.SELL,
                quantity=5,
                price=448.0,
                order_id="test-order-2",
            ),
        )
        assert event.event_type == "FILL"
        assert event.fill is not None

    def test_alert_creation(self):
        alert = Alert(
            alert_type=AlertType.DRAWDOWN_BREACH,
            severity=AlertSeverity.WARNING,
            message="Test alert",
            value=0.08,
            threshold=0.10,
        )
        assert alert.acknowledged is False
        assert len(alert.alert_id) == 12


# --- Config ---


class TestConfig:
    def test_default_config(self):
        config = MonitoringConfig()
        assert config.alerts.max_drawdown == 0.10
        assert config.alerts.max_daily_loss == 0.02
        assert config.dashboard_refresh_seconds == 5.0

    def test_custom_config(self):
        from src.monitoring.config import AlertThresholds
        config = MonitoringConfig(
            alerts=AlertThresholds(
                max_drawdown=0.15,
                kill_switch_drawdown=0.20,
            )
        )
        assert config.alerts.max_drawdown == 0.15
        assert config.alerts.kill_switch_drawdown == 0.20


# --- Metrics ---


class TestMetricsTracker:
    def test_initial_balance(self):
        tracker = MetricsTracker(initial_balance=100_000.0)
        assert tracker.current_balance == 100_000.0
        assert tracker.realized_pnl == 0.0
        assert tracker.unrealized_pnl == 0.0

    def test_buy_fill_reduces_balance(self):
        tracker = MetricsTracker(initial_balance=100_000.0)
        fill = Fill(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=150.0,
            order_id="buy-1",
            commission=1.0,
        )
        tracker.record_fill(fill)
        assert tracker.current_balance == pytest.approx(100_000.0 - 15_000.0 - 1.0, abs=0.1)

    def test_sell_fill_increases_balance(self):
        tracker = MetricsTracker(initial_balance=100_000.0)
        tracker.record_fill(
            Fill(ticker="AAPL", side=OrderSide.BUY, quantity=100, price=150.0, order_id="b1", commission=0.0)
        )
        tracker.record_fill(
            Fill(ticker="AAPL", side=OrderSide.SELL, quantity=100, price=160.0, order_id="s1", commission=0.0)
        )
        assert tracker.current_balance > 100_000.0

    def test_unrealized_pnl(self):
        tracker = MetricsTracker(initial_balance=100_000.0)
        tracker.record_fill(
            Fill(ticker="AAPL", side=OrderSide.BUY, quantity=100, price=150.0, order_id="b1", commission=0.0)
        )
        # Update price upward
        tracker.update_positions([
            Position(ticker="AAPL", quantity=100, avg_cost=150.0, current_price=160.0, unrealized_pnl=0.0, realized_pnl=0.0)
        ])
        assert tracker.unrealized_pnl == pytest.approx(1000.0)

    def test_drawdown(self):
        tracker = MetricsTracker(initial_balance=100_000.0)
        # Record some daily returns to establish peak
        tracker.record_daily_return()  # day 1, flat
        # Simulate position update to establish peak
        tracker._positions["CASH"] = Position(
            ticker="CASH", quantity=1, avg_cost=0.0, current_price=100_000.0, unrealized_pnl=0.0, realized_pnl=0.0
        )
        tracker._daily_start_balance = tracker.current_balance
        tracker.record_daily_return()
        # Drop the balance
        tracker._balance = 90_000.0
        tracker._positions.clear()
        assert tracker.current_drawdown > 0.0

    def test_sharpe_ratio(self):
        tracker = MetricsTracker(initial_balance=100_000.0)
        # Simulate daily returns
        tracker._daily_returns = [0.01, -0.005, 0.02, -0.01, 0.015]
        assert isinstance(tracker.sharpe_ratio, float)

    def test_win_rate(self):
        tracker = MetricsTracker(initial_balance=100_000.0)
        tracker._daily_start_balance = 100_000.0
        tracker._balance = 100_000.0
        tracker._positions.clear()
        # Simulate wins and losses
        tracker._num_wins = 3
        tracker._num_losses = 2
        assert tracker.win_rate == pytest.approx(0.6)

    def test_to_dataframe(self):
        tracker = MetricsTracker(initial_balance=100_000.0)
        tracker.record_fill(
            Fill(ticker="AAPL", side=OrderSide.BUY, quantity=10, price=150.0, order_id="b1", commission=0.0)
        )
        df = tracker.to_dataframe()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 1
        assert df["ticker"][0] == "AAPL"

    def test_daily_summary(self):
        tracker = MetricsTracker(initial_balance=100_000.0)
        summary = tracker.daily_summary("2026-07-19")
        assert isinstance(summary, DailySummary)
        assert summary.date == "2026-07-19"
        assert summary.starting_balance == 100_000.0


# --- Alerts ---


class TestAlertManager:
    def test_drawdown_below_threshold(self):
        config = MonitoringConfig()
        mgr = AlertManager(config)
        result = mgr.check_drawdown(0.05)  # 5% < 10% threshold
        assert result is None
        assert len(mgr.alerts) == 0

    def test_drawdown_above_threshold(self):
        config = MonitoringConfig()
        mgr = AlertManager(config)
        result = mgr.check_drawdown(0.12)  # 12% > 10% threshold
        assert result is not None
        assert result.alert_type == AlertType.DRAWDOWN_BREACH
        assert result.severity == AlertSeverity.WARNING

    def test_kill_switch(self):
        config = MonitoringConfig()
        mgr = AlertManager(config)
        result = mgr.check_drawdown(0.20)  # 20% > 15% kill switch
        assert result is not None
        assert result.alert_type == AlertType.KILL_SWITCH
        assert result.severity == AlertSeverity.CRITICAL

    def test_daily_loss_below_threshold(self):
        config = MonitoringConfig()
        mgr = AlertManager(config)
        result = mgr.check_daily_loss(-0.01)  # 1% < 2% threshold
        assert result is None

    def test_daily_loss_above_threshold(self):
        config = MonitoringConfig()
        mgr = AlertManager(config)
        result = mgr.check_daily_loss(-0.03)  # 3% > 2% threshold
        assert result is not None
        assert result.alert_type == AlertType.DAILY_LOSS_LIMIT

    def test_position_limit(self):
        config = MonitoringConfig()
        mgr = AlertManager(config)
        result = mgr.check_position_limit(0.30, "AAPL")  # 30% > 25% limit
        assert result is not None
        assert result.ticker == "AAPL"

    def test_position_within_limit(self):
        config = MonitoringConfig()
        mgr = AlertManager(config)
        result = mgr.check_position_limit(0.20, "AAPL")  # 20% < 25% limit
        assert result is None

    def test_leverage_check(self):
        config = MonitoringConfig()
        mgr = AlertManager(config)
        result = mgr.check_leverage(2.0)  # 2x > 1x max
        assert result is not None
        assert result.alert_type == AlertType.LEVERAGE_EXCEEDED

    def test_rate_limiting(self):
        config = MonitoringConfig()
        mgr = AlertManager(config)
        mgr.check_drawdown(0.12)  # First alert fires
        assert len(mgr.alerts) > 0
        # Immediately fire again — should be rate-limited
        old_cd = AlertManager.COOLDOWN_SECONDS
        AlertManager.COOLDOWN_SECONDS = 99999
        mgr2 = AlertManager(config)
        mgr2.check_drawdown(0.12)
        assert len(mgr2.alerts) > 0
        mgr2._last_fired[AlertType.DRAWDOWN_BREACH.value] = time.time()
        result = mgr2.check_drawdown(0.12)
        # Should be rate-limited
        assert result is None or len(mgr2.alerts) == 1
        AlertManager.COOLDOWN_SECONDS = old_cd

    def test_callback(self):
        config = MonitoringConfig()
        mgr = AlertManager(config)
        received = []
        mgr.add_callback(lambda a: received.append(a))
        mgr.check_drawdown(0.12)
        assert len(received) > 0

    def test_acknowledge(self):
        config = MonitoringConfig()
        mgr = AlertManager(config)
        mgr.check_drawdown(0.12)
        alert = mgr.alerts[-1]
        assert mgr.acknowledge(alert.alert_id) is True
        assert alert.acknowledged is True
        assert mgr.acknowledge("nonexistent") is False

    def test_active_count(self):
        config = MonitoringConfig()
        mgr = AlertManager(config)
        mgr.check_drawdown(0.12)
        assert mgr.active_count >= 1

    def test_process_trade_event_error(self):
        config = MonitoringConfig()
        mgr = AlertManager(config)
        event = TradeEvent(
            event_type="ERROR",
            error="Connection timeout",
        )
        mgr.process_trade_event(event)
        assert len(mgr.alerts) > 0

    def test_get_unacknowledged(self):
        config = MonitoringConfig()
        mgr = AlertManager(config)
        mgr.check_drawdown(0.12)
        unacked = mgr.get_unacknowledged()
        assert len(unacked) > 0
        # Acknowledge all
        for a in unacked:
            mgr.acknowledge(a.alert_id)
        assert len(mgr.get_unacknowledged()) == 0


# --- Dashboard ---


class TestDashboard:
    def test_render(self):
        config = MonitoringConfig()
        dash = Dashboard(config)
        metrics = {
            "balance": 100_000.0,
            "total_pnl": 1_500.0,
            "total_return": 0.015,
            "realized_pnl": 1_000.0,
            "unrealized_pnl": 500.0,
            "drawdown": 0.02,
            "max_drawdown": 0.05,
            "sharpe_ratio": 1.5,
            "win_rate": 0.6,
            "num_trades": 15,
            "num_positions": 2,
        }
        dash.update(metrics)
        output = dash.render()
        assert isinstance(output, str)
        assert "Portfolio Overview" in output or "ALPHA MONITORING" in output

    def test_render_with_positions(self):
        config = MonitoringConfig()
        dash = Dashboard(config)
        metrics = {"balance": 100_000.0, "total_pnl": 0.0, "total_return": 0.0, "realized_pnl": 0.0, "unrealized_pnl": 0.0, "drawdown": 0.0, "max_drawdown": 0.0, "sharpe_ratio": 0.0, "win_rate": 0.0, "num_trades": 0, "num_positions": 1}
        positions = [
            Position(ticker="AAPL", quantity=100, avg_cost=150.0, current_price=155.0, unrealized_pnl=500.0, realized_pnl=0.0),
        ]
        dash.update(metrics, positions=positions)
        output = dash.render()
        assert "AAPL" in output

    def test_render_with_alerts(self):
        config = MonitoringConfig()
        dash = Dashboard(config)
        metrics = {"balance": 100_000.0, "total_pnl": 0.0, "total_return": 0.0, "realized_pnl": 0.0, "unrealized_pnl": 0.0, "drawdown": 0.0, "max_drawdown": 0.0, "sharpe_ratio": 0.0, "win_rate": 0.0, "num_trades": 0, "num_positions": 0}
        alerts = [
            Alert(alert_type=AlertType.DRAWDOWN_BREACH, severity=AlertSeverity.WARNING, message="Test alert"),
        ]
        dash.update(metrics, alerts=alerts)
        output = dash.render()
        assert "DRAWDOWN_BREACH" in output


# --- Report Generator ---


class TestReportGenerator:
    def test_generate_daily(self):
        config = MonitoringConfig()
        tracker = MetricsTracker(initial_balance=100_000.0)
        gen = ReportGenerator(config, tracker)
        summary = gen.generate_daily("2026-07-19")
        assert isinstance(summary, DailySummary)
        assert summary.date == "2026-07-19"

    def test_save_text(self):
        from pathlib import Path
        config = MonitoringConfig()
        tracker = MetricsTracker(initial_balance=100_000.0)
        with tempfile.TemporaryDirectory() as tmp:
            config.report_output_dir = Path(tmp)
            gen = ReportGenerator(config, tracker)
            summary = gen.generate_daily("2026-07-19")
            path = gen.save(summary, format="text")
            assert path.exists()
            content = path.read_text()
            assert "2026-07-19" in content
            assert "Daily P&L" in content

    def test_save_parquet(self):
        from pathlib import Path
        config = MonitoringConfig()
        tracker = MetricsTracker(initial_balance=100_000.0)
        with tempfile.TemporaryDirectory() as tmp:
            config.report_output_dir = Path(tmp)
            gen = ReportGenerator(config, tracker)
            summary = gen.generate_daily("2026-07-19")
            path = gen.save(summary, format="parquet")
            assert path.exists()
            df = pl.read_parquet(path)
            assert len(df) == 1

    def test_save_all_reports(self):
        from pathlib import Path
        config = MonitoringConfig()
        tracker = MetricsTracker(initial_balance=100_000.0)
        with tempfile.TemporaryDirectory() as tmp:
            config.report_output_dir = Path(tmp)
            gen = ReportGenerator(config, tracker)
            gen.generate_daily("2026-07-18")
            gen.generate_daily("2026-07-19")
            path = gen.save_all_reports()
            assert path.exists()
            df = pl.read_parquet(path)
            assert len(df) == 2

    def test_get_reports(self):
        config = MonitoringConfig()
        tracker = MetricsTracker(initial_balance=100_000.0)
        gen = ReportGenerator(config, tracker)
        gen.generate_daily("2026-07-19")
        reports = gen.get_reports()
        assert len(reports) == 1