"""Risk management controls.

Hard constraint layer enforcing:
- Position size caps
- Sector exposure limits
- Leverage limits
- Volatility targeting
- Drawdown circuit breakers
- Correlation spike detection

Sits between Kelly sizing (Phase 5.1) and portfolio construction (Phase 5.3).
Kelly produces raw weights; risk management enforces constraints before
portfolio construction allocates final capital.
"""

from .config import RiskConfig
from .constraints import enforce_position_limits, enforce_sector_limits, enforce_leverage
from .vol_targeting import volatility_target
from .drawdown import drawdown_circuit_breaker
from .correlation import correlation_spike_detector
from .pipeline import enforce_all_constraints

__all__ = [
    "RiskConfig",
    "enforce_position_limits",
    "enforce_sector_limits",
    "enforce_leverage",
    "volatility_target",
    "drawdown_circuit_breaker",
    "correlation_spike_detector",
    "enforce_all_constraints",
]