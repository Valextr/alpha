"""Live trading configuration.

Defines the universe, data source, execution settings, and monitoring
configuration for the daily trading loop.

Usage:
    from src.live.config import LiveConfig

    config = LiveConfig()  # loads defaults + .env overrides
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv

# Auto-load .env from project root
load_dotenv(Path(__file__).resolve().parents[3] / ".env")


@dataclass
class LiveConfig:
    """Configuration for the live trading runner.

    Args:
        tickers: Universe of tickers to trade.
        data_dir: Path to the gold data directory.
        lookback_days: Minimum history required for feature computation.
        ensemble_mode: "ic_weighted" or "lightgbm".
        ic_lookback: Rolling window for IC computation (trading days).
        ic_target_horizon: Forward return horizon for IC target.
        top_n: Only trade top N signals by absolute strength (None = all).
        min_signal_strength: Minimum absolute signal to trade.
        max_positions: Max concurrent positions.
        max_position_weight: Max portfolio weight per position.
        daily_loss_limit_pct: Daily loss circuit breaker (% of portfolio).
        max_drawdown_pct: Max drawdown kill switch (% from peak).
        report_dir: Directory for daily reports.
    """

    # Universe
    tickers: list[str] = field(
        default_factory=lambda: ["AAPL", "AMZN", "GOOGL", "META", "MSFT"]
    )

    # Data
    data_dir: Path = field(default_factory=lambda: Path("data"))
    lookback_days: int = 252  # ~1 year of history needed for features

    # Ensemble
    ensemble_mode: Literal["ic_weighted", "lightgbm"] = "ic_weighted"
    ic_lookback: int = 63
    ic_target_horizon: int = 5

    # Signal filtering
    top_n: Optional[int] = None
    min_signal_strength: float = 0.1

    # Risk
    max_positions: int = 10
    max_position_weight: float = 0.25
    daily_loss_limit_pct: float = 0.05
    max_drawdown_pct: float = 0.15

    # Reports
    report_dir: Path = field(default_factory=lambda: Path("reports/live"))

    def __post_init__(self):
        self.report_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "LiveConfig":
        """Load configuration from environment variables."""
        return cls(
            tickers=os.getenv(
                "ALPHA_TICKERS", "AAPL,AMZN,GOOGL,META,MSFT"
            ).split(","),
            data_dir=Path(os.getenv("ALPHA_DATA_DIR", "data")),
            lookback_days=int(os.getenv("ALPHA_LOOKBACK_DAYS", "252")),
            ensemble_mode=os.getenv("ALPHA_ENSEMBLE_MODE", "ic_weighted"),
            ic_lookback=int(os.getenv("ALPHA_IC_LOOKBACK", "63")),
            ic_target_horizon=int(os.getenv("ALPHA_IC_TARGET_HORIZON", "5")),
            top_n=int(os.getenv("ALPHA_TOP_N", "0")) if os.getenv("ALPHA_TOP_N") else None,
            min_signal_strength=float(os.getenv("ALPHA_MIN_SIGNAL", "0.1")),
            max_positions=int(os.getenv("ALPHA_MAX_POSITIONS", "10")),
            max_position_weight=float(os.getenv("ALPHA_MAX_POSITION_WEIGHT", "0.25")),
            daily_loss_limit_pct=float(os.getenv("ALPHA_DAILY_LOSS_LIMIT", "0.05")),
            max_drawdown_pct=float(os.getenv("ALPHA_MAX_DRAWDOWN", "0.15")),
            report_dir=Path(os.getenv("ALPHA_REPORT_DIR", "reports/live")),
        )