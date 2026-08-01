"""Monitoring configuration with alert thresholds.

Central configuration for the monitoring module. Uses pydantic for
validation and supports environment variable overrides via ALPHA_ prefix.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class AlertThresholds(BaseModel):
    """Thresholds for various alert types."""

    # Risk thresholds
    max_drawdown: float = Field(
        default=0.10,
        description="Max portfolio drawdown before alert (10% default)"
    )
    max_daily_loss: float = Field(
        default=0.02,
        description="Max daily loss percentage (2% default)"
    )
    max_position_pct: float = Field(
        default=0.25,
        description="Max position as pct of portfolio (25% default)"
    )
    max_leverage: float = Field(
        default=1.0,
        description="Max leverage (1.0 = cash only, 2.0 = 2x)"
    )
    kill_switch_drawdown: float = Field(
        default=0.15,
        description="Drawdown level that triggers kill switch (15% default)"
    )

    # Operational thresholds
    max_fill_slippage_pct: float = Field(
        default=2.0,
        description="Max fill slippage in pct before alert"
    )
    reconciliation_stale_minutes: int = Field(
        default=30,
        description="Minutes before reconciliation is considered stale"
    )

    # Performance thresholds
    anomalous_return_sigma: float = Field(
        default=3.0,
        description="Number of sigmas for anomalous performance alert"
    )
    ic_degradation_threshold: float = Field(
        default=0.3,
        description="IC drop pct before signal degradation alert"
    )


class MonitoringConfig(BaseModel):
    """Central configuration for monitoring."""

    # Alert thresholds
    alerts: AlertThresholds = Field(
        default_factory=AlertThresholds,
        description="Alert threshold configuration"
    )

    # Dashboard
    dashboard_refresh_seconds: float = Field(
        default=5.0,
        description="Dashboard refresh interval in seconds"
    )

    # Reporting
    report_output_dir: Path = Field(
        default=Path("data/reports"),
        description="Directory for daily report outputs"
    )

    # Metrics
    sharpe_annualization: int = Field(
        default=252,
        description="Trading days per year for Sharpe calculation"
    )
    drawdown_window: int = Field(
        default=21,
        description="Lookback window for drawdown computation (trading days)"
    )

    # Storage
    metrics_store_path: Path = Field(
        default=Path("data/metrics.parquet"),
        description="Parquet file for persistent metrics storage"
    )

    model_config = {"env_prefix": "ALPHA_"}

    def model_post_init(self, __context) -> None:
        """Ensure output directories exist."""
        self.report_output_dir.mkdir(parents=True, exist_ok=True)