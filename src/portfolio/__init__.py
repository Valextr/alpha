"""Portfolio construction and risk management.

Submodules:
    kelly         — Kelly criterion position sizing (fractional Kelly).
    construction  — Risk parity, HRP, Ledoit-Wolf covariance shrinkage.
    integration   — End-to-end portfolio pipeline (Kelly → Risk → Construction).
"""

from .kelly import KellyConfig, compute_kelly_fraction, compute_position_weights, compute_position_weights_from_signal_scores
from .construction import (
    PortfolioWeights,
    RiskContribution,
    ledoit_wolf_shrinkage,
    risk_parity_weights,
    risk_contributions,
    hrp_weights,
    hrp_with_clusters,
    returns_from_prices,
    pivot_returns,
    weights_to_dataframe,
    portfolio_volatility,
    portfolio_sharpe,
)
from .integration import PipelineConfig, PortfolioPipeline

__all__ = [
    # Kelly
    "KellyConfig",
    "compute_kelly_fraction",
    "compute_position_weights",
    "compute_position_weights_from_signal_scores",
    # Construction
    "PortfolioWeights",
    "RiskContribution",
    "ledoit_wolf_shrinkage",
    "risk_parity_weights",
    "risk_contributions",
    "hrp_weights",
    "hrp_with_clusters",
    "returns_from_prices",
    "pivot_returns",
    "weights_to_dataframe",
    "portfolio_volatility",
    "portfolio_sharpe",
    # Integration
    "PipelineConfig",
    "PortfolioPipeline",
]