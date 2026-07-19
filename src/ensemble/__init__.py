"""Phase 4: Ensemble & Weights.

Combines individual signals into a meta-predictor.

Exports
-------
ICWeightedEnsemble
    Rolling IC-weighted linear ensemble.
EnsembleConfig
    Configuration dataclass for the ensemble.
EnsembleMeta
    Metadata snapshot for auditability.
ic_to_weights
    Convert IC values to normalized weights.
detect_signal_columns
    Auto-detect signal_* columns from a DataFrame.
run_ensemble
    One-call pipeline entry point.
LightGBMEnsemble
    Gradient-boosted meta-learner (Phase 4.3, conditional).
"""

from .base import (
    EnsembleConfig,
    EnsembleMeta,
    detect_signal_columns,
    ic_to_weights,
)
from .ic_weighted import ICWeightedEnsemble
from .pipeline import run_ensemble

__all__ = [
    "EnsembleConfig",
    "EnsembleMeta",
    "ICWeightedEnsemble",
    "detect_signal_columns",
    "ic_to_weights",
    "run_ensemble",
]

# LightGBM is a conditional dependency — import lazily to avoid
# hard-requiring it during IC-ensemble-only runs.
# (Kept for backward compatibility with existing __init__.py consumers.)
try:
    from .lightgbm import LightGBMEnsemble

    __all__.append("LightGBMEnsemble")
except ImportError:
    pass