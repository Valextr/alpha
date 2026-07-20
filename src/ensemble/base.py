"""Ensemble base: configuration, metadata, and weight utilities.

This module defines the shared types used by the IC-weighted ensemble
and the downstream pipeline / validation modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Sequence


@dataclass(frozen=True)
class EnsembleConfig:
    """Configuration for the IC-weighted ensemble.

    Args:
        ic_lookback: Rolling window for IC computation (trading days).
        ic_target_horizon: Forward return horizon used as IC target.
        ic_min_observations: Minimum observations for a valid IC.
        rebalance_frequency: Days between weight updates.
        weight_method: Method for converting IC → weights.
            One of ``"abs_ic"``, ``"rank_ic"``, or ``"positive_ic"``.
        signal_columns: Signal columns to include (``None`` = auto-detect).
        min_signal_ic: Minimum IC magnitude to include a signal.
        output_column: Name of the ensemble output column.
        track_weights: Whether to emit ``ensemble_weight_<name>`` columns.
    """

    ic_lookback: int = 63
    ic_target_horizon: int = 5
    ic_min_observations: int = 30
    rebalance_frequency: int = 5
    weight_method: str = "abs_ic"
    signal_columns: list[str] | None = None
    min_signal_ic: float = 0.0
    output_column: str = "signal_ensemble"
    track_weights: bool = True

    def __post_init__(self) -> None:
        if self.ic_lookback < 5:
            raise ValueError(f"ic_lookback must be >= 5, got {self.ic_lookback}")
        if self.ic_target_horizon < 1:
            raise ValueError(f"ic_target_horizon must be >= 1, got {self.ic_target_horizon}")
        if self.rebalance_frequency < 1:
            raise ValueError(
                f"rebalance_frequency must be >= 1, got {self.rebalance_frequency}"
            )
        if self.weight_method not in ("abs_ic", "rank_ic", "positive_ic"):
            raise ValueError(
                f"weight_method must be abs_ic/rank_ic/positive_ic, "
                f"got {self.weight_method!r}"
            )


@dataclass
class EnsembleMeta:
    """Metadata snapshot for the ensemble at a given rebalance date.

    Captures which signals contributed, their IC values, and their
    resulting weights. Used for auditability and attribution reporting.
    """

    date: date
    weights: dict[str, float]  # signal_name → weight
    ic_snapshot: dict[str, float]  # signal_name → rolling IC
    signal_count: int  # number of signals contributing positive weight
    effective_ic: float = 0.0  # IC of the ensemble itself (computed later)


def ic_to_weights(
    ic_values: dict[str, float],
    method: str = "abs_ic",
) -> dict[str, float]:
    """Convert IC values to normalized weights (sum to 1.0).

    Methods
    -------
    abs_ic
        Weight proportional to |IC|. A signal with strong predictive power
        (regardless of direction) receives a larger weight.
    rank_ic
        Weight proportional to IC rank (1-based). The signal with the
        highest |IC| gets the largest weight.
    positive_ic
        Only signals with positive IC contribute. Weight equals IC value.
        Signals with negative IC receive zero weight. Falls back to
        equal weight if all ICs are non-positive.

    Args:
        ic_values: Mapping from signal name to IC value.
        method: Weighting method (default ``"abs_ic"``).

    Returns:
        Mapping from signal name to weight. Values sum to 1.0.
    """
    if not ic_values:
        return {}

    if method == "abs_ic":
        abs_ics = {k: abs(v) for k, v in ic_values.items()}
        total = sum(abs_ics.values()) or 1.0
        return {k: v / total for k, v in abs_ics.items()}

    if method == "rank_ic":
        ranked = sorted(ic_values.items(), key=lambda x: abs(x[1]))
        weights = {k: float(i + 1) for i, (k, _) in enumerate(ranked)}
        total = sum(weights.values()) or 1.0
        return {k: v / total for k, v in weights.items()}

    if method == "positive_ic":
        pos_ics = {k: max(v, 0.0) for k, v in ic_values.items()}
        total = sum(pos_ics.values())
        if total > 0:
            return {k: v / total for k, v in pos_ics.items()}
        # Fallback to equal weights when all ICs are non-positive
        n = len(ic_values) or 1
        return {k: 1.0 / n for k in ic_values}

    # Default fallback: equal weight
    n = len(ic_values) or 1
    return {k: 1.0 / n for k in ic_values}


def detect_signal_columns(df_columns: Sequence[str]) -> list[str]:
    """Auto-detect signal columns from a DataFrame column list.

    Returns columns that start with ``"signal_"`` and are not
    ``"signal_ensemble"`` (i.e. individual signal outputs, not the
    ensemble output itself).
    """
    return sorted(
        col for col in df_columns
        if col.startswith("signal_") and col != "signal_ensemble"
    )