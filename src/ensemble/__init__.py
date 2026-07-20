"""Phase 4: Ensemble & Weights.

Exports
-------
EnsemblePipeline
    Unified orchestrator (IC-weighted or LightGBM mode).
EnsemblePipelineConfig
    Configuration for the unified pipeline.
ICWeightedEnsemble
    Rolling IC-weighted linear ensemble.
EnsembleConfig
    Configuration dataclass for the IC-weighted ensemble.
LightGBMEnsemble
    Gradient-boosted meta-learner (Phase 4.3).
LightGBMEnsembleConfig
    Configuration dataclass for the LightGBM meta-learner.
run_ensemble
    Legacy: compatibility wrapper around EnsemblePipeline.
"""

from .base import (
    EnsembleConfig,
    EnsembleMeta,
    detect_signal_columns,
    ic_to_weights,
)
from .ic_weighted import (
    ICWeightedEnsemble,
    compute_cross_sectional_ic,
    compute_rolling_ic,
)
from .pipeline import (
    EnsemblePipeline,
    EnsemblePipelineConfig,
    run_ensemble,
)
from .validation import (
    EnsembleMetrics,
    SignalMetrics,
    WeightStats,
    compute_cs_ic_stats,
    format_ensemble_report,
    format_metrics_table,
    signal_metrics,
    validate_ensemble,
    validate_signals,
    weight_report,
)

__all__ = [
    # Unified pipeline
    "EnsemblePipeline",
    "EnsemblePipelineConfig",
    # Base
    "EnsembleConfig",
    "EnsembleMeta",
    "detect_signal_columns",
    "ic_to_weights",
    # IC-weighted ensemble
    "ICWeightedEnsemble",
    "compute_cross_sectional_ic",
    "compute_rolling_ic",
    # Pipeline
    "run_ensemble",
    # Validation
    "SignalMetrics",
    "EnsembleMetrics",
    "WeightStats",
    "compute_cs_ic_stats",
    "signal_metrics",
    "validate_signals",
    "validate_ensemble",
    "weight_report",
    "format_metrics_table",
    "format_ensemble_report",
]

try:
    from .lightgbm import LightGBMEnsemble, LightGBMEnsembleConfig

    __all__.extend(["LightGBMEnsemble", "LightGBMEnsembleConfig"])
except ImportError:
    pass