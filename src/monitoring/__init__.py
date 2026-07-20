"""Monitoring dashboard and alerts for paper trading.

Submodules:
    data_models  — Pydantic models (Position, Fill, TradeEvent, Alert, etc.)
    config       — MonitoringConfig with alert thresholds
    metrics      — P&L, drawdown, Sharpe, win rate (Polars-based)
    alerts       — Alert engine with configurable thresholds
    dashboard    — Terminal-based Rich dashboard
    report       — Daily summary report generator

Architecture:
    The execution engine pushes events (fills, position changes, errors).
    The monitoring module consumes those events and:
      1. Updates metrics in real time
      2. Fires alerts when thresholds are breached
      3. Renders a live dashboard (terminal-based, Rich/Textual)
      4. Generates daily summary reports
"""

from __future__ import annotations

from .data_models import (
    Alert,
    AlertSeverity,
    AlertType,
    DailySummary,
    Fill,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TradeEvent,
)
from .config import MonitoringConfig
from .metrics import MetricsTracker
from .alerts import AlertManager
from .dashboard import Dashboard
from .report import ReportGenerator

__all__ = [
    # Data models
    "Alert",
    "AlertSeverity",
    "AlertType",
    "DailySummary",
    "Fill",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "Position",
    "TradeEvent",
    # Config
    "MonitoringConfig",
    # Metrics
    "MetricsTracker",
    # Alerts
    "AlertManager",
    # Dashboard
    "Dashboard",
    # Reports
    "ReportGenerator",
]