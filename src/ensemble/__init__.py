"""Phase 4: Ensemble & Weights.

Combine weak signals into a strong meta-predictor using IC-weighted
linear combinations.

Exports:
    compute_csic: per-date Information Coefficient for a signal
    compute_ensemble_weights: IC-based rolling weights
    compute_ic_weighted_ensemble: full ensemble pipeline
"""

from .ic_weighted import (
    compute_csic,
    compute_ensemble_weights,
    compute_ic_weighted_ensemble,
)

__all__ = [
    "compute_csic",
    "compute_ensemble_weights",
    "compute_ic_weighted_ensemble",
]