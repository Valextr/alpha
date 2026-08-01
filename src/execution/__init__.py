"""Signal-to-order pipeline, position tracking, and fill reconciliation.

The execution engine sits between the ensemble/portfolio layer and the broker:

    Signals → Ensemble → Portfolio → ExecutionEngine → Broker → Market

Responsibilities:
- Convert portfolio targets into executable orders
- Track positions, fills, and P&L
- Reconcile broker fills against internal state
- Enforce risk guardrails (position limits, kill switch, daily P&L caps)
- Support paper trading simulation and live trading via Interactive Brokers
- Intraday bar-by-bar trading with IBKR integration
"""

from .intraday_engine import IntradayEngine, IntradayConfig
from .bar_builder import BarBuilder, Bar, Tick
from .live_runner import LiveIntradayRunner, RunnerConfig

__all__ = [
    "IntradayEngine",
    "IntradayConfig",
    "BarBuilder",
    "Bar",
    "Tick",
    "LiveIntradayRunner",
    "RunnerConfig",
]