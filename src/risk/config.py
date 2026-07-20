"""Risk management configuration.

Central config object defining all risk thresholds. Uses Pydantic for
validation — every constraint is typed and checked at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RiskConfig:
    """Risk management configuration.

    All values are conservative defaults suitable for paper trading.
    Tighten or relax based on validation results (Phase 6).

    Attributes:
        max_position_pct: Maximum weight for any single position (fraction
            of total portfolio). Default 0.10 (10%).
        max_sector_pct: Maximum combined weight for any single sector.
            Default 0.30 (30%).
        max_leverage: Maximum gross leverage. 1.0 = long-only, 2.0 =
            100% long + 100% short. Default 1.5.
        target_vol: Target annualized portfolio volatility. Default 0.10
            (10%). Actual scaling is applied via volatility_target().
        lookback_vol: Rolling window for realized volatility estimation
            (trading days). Default 63 (~3 months).
        max_drawdown: Maximum portfolio drawdown from peak. When breached,
            leverage scales down proportionally. Default 0.20 (20%).
        drawdown_recovery_pct: Fraction of drawdown that must recover before
            leverage scales back up. Default 0.50 (50% recovery required).
        correlation_window: Rolling window for correlation matrix estimation
            (trading days). Default 63.
        correlation_spike_threshold: Average pairwise correlation above which
            position counts are reduced. Default 0.70.
        min_positions_after_spike: Minimum positions retained after correlation
            spike reduction. Default 2.
        fraction_positions_on_spike: Fraction of positions to keep during
            correlation spikes. Default 0.50 (keep half).
    """

    max_position_pct: float = 0.10
    max_sector_pct: float = 0.30
    max_leverage: float = 1.5
    target_vol: float = 0.10
    lookback_vol: int = 63
    max_drawdown: float = 0.20
    drawdown_recovery_pct: float = 0.50
    correlation_window: int = 63
    correlation_spike_threshold: float = 0.70
    min_positions_after_spike: int = 2
    fraction_positions_on_spike: float = 0.50

    def __post_init__(self) -> None:
        # Validate ranges
        if not 0 < self.max_position_pct <= 1.0:
            raise ValueError(f"max_position_pct must be in (0, 1]: {self.max_position_pct}")
        if not 0 < self.max_sector_pct <= 1.0:
            raise ValueError(f"max_sector_pct must be in (0, 1]: {self.max_sector_pct}")
        if self.max_leverage < 0:
            raise ValueError(f"max_leverage must be >= 0: {self.max_leverage}")
        if not 0 < self.target_vol <= 1.0:
            raise ValueError(f"target_vol must be in (0, 1]: {self.target_vol}")
        if self.lookback_vol < 10:
            raise ValueError(f"lookback_vol must be >= 10: {self.lookback_vol}")
        if not 0 < self.max_drawdown <= 1.0:
            raise ValueError(f"max_drawdown must be in (0, 1]: {self.max_drawdown}")
        if not 0 <= self.drawdown_recovery_pct <= 1.0:
            raise ValueError(f"drawdown_recovery_pct must be in [0, 1]: {self.drawdown_recovery_pct}")
        if self.correlation_window < 10:
            raise ValueError(f"correlation_window must be >= 10: {self.correlation_window}")
        if not 0.0 <= self.correlation_spike_threshold <= 1.0:
            raise ValueError(f"correlation_spike_threshold must be in [0, 1]: {self.correlation_spike_threshold}")
        if self.min_positions_after_spike < 1:
            raise ValueError(f"min_positions_after_spike must be >= 1: {self.min_positions_after_spike}")
        if not 0 < self.fraction_positions_on_spike <= 1.0:
            raise ValueError(f"fraction_positions_on_spike must be in (0, 1]: {self.fraction_positions_on_spike}")

    @classmethod
    def conservative(cls) -> RiskConfig:
        """Very conservative settings for initial paper trading."""
        return cls(
            max_position_pct=0.05,
            max_sector_pct=0.20,
            max_leverage=1.0,
            target_vol=0.05,
            lookback_vol=126,
            max_drawdown=0.10,
            drawdown_recovery_pct=0.75,
            correlation_spike_threshold=0.60,
            fraction_positions_on_spike=0.33,
        )

    @classmethod
    def aggressive(cls) -> RiskConfig:
        """More aggressive settings for validated strategies."""
        return cls(
            max_position_pct=0.20,
            max_sector_pct=0.50,
            max_leverage=2.0,
            target_vol=0.20,
            lookback_vol=21,
            max_drawdown=0.30,
            drawdown_recovery_pct=0.25,
        )