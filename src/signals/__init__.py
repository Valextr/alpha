"""Phase 3: Signal Factory

Composable signal modules, each independently validated.

Exports:
    registry: SignalRegistry singleton
    generate_all: generate all registered signals on a DataFrame
    generate_all_with_forward_returns: generate signals + attach forward returns
"""

from .registry import SignalRegistry, registry
from .base import (
    compute_forward_returns,
    rank_ic,
    ic_decay,
    win_rate,
    signal_summary,
)
from .pipeline import generate_all, generate_all_with_forward_returns

# Eagerly import signal modules so their @registry.register decorators fire.
from . import mean_reversion, momentum  # noqa: F401

__all__ = [
    "SignalRegistry",
    "registry",
    "generate_all",
    "generate_all_with_forward_returns",
    "compute_forward_returns",
    "rank_ic",
    "ic_decay",
    "win_rate",
    "signal_summary",
]