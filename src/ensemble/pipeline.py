"""Phase 4: Ensemble pipeline — unified orchestrator.

Combines weak individual signals into a single meta-predictor. Two modes:

    - ``ic_weighted`` — IC-weighted linear combination. Interpretable, no
      training loop, weights update automatically each rolling window.
    - ``lightgbm`` — Gradient-boosted meta-learner. Non-linear signal
      combination, requires explicit training with hold-back enforcement.

Typical usage (IC-weighted):

    from src.ensemble.pipeline import EnsemblePipeline, EnsemblePipelineConfig

    pipeline = EnsemblePipeline(EnsemblePipelineConfig(mode="ic_weighted"))
    result = pipeline.run(df)

    # result now has ``signal_ensemble`` + optional weight columns

Typical usage (LightGBM):

    pipeline = EnsemblePipeline(EnsemblePipelineConfig(mode="lightgbm"))
    result = pipeline.run(df, train_end="2022-12-31")

    # result now has ``ensemble_prediction`` column (probability in [0, 1])

Validation:

    report = pipeline.validate(result)

    # Returns EnsembleMetrics comparing ensemble IC to individual signal ICs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal, Sequence

import polars as pl

from .base import (
    EnsembleConfig,
    EnsembleMeta,
    detect_signal_columns,
    ic_to_weights,
)
from .ic_weighted import ICWeightedEnsemble, compute_cross_sectional_ic
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


@dataclass
class EnsemblePipelineConfig:
    """Configuration for the unified ensemble pipeline.

    Args:
        mode: Ensemble mode. ``"ic_weighted"`` for linear IC-weighting,
            ``"lightgbm"`` for the gradient-boosted meta-learner.
        ic_lookback: Rolling window for IC computation (trading days).
        ic_target_horizon: Forward return horizon used as IC target.
        ic_min_observations: Minimum observations for a valid IC.
        rebalance_frequency: Days between weight updates.
        weight_method: IC-to-weight conversion method
            (``"abs_ic"``, ``"rank_ic"``, ``"positive_ic"``).
        signal_columns: Signal columns to include (``None`` = auto-detect).
        output_column: Name of the ensemble output column.
        track_weights: Whether to emit weight columns for auditability.
        lightgbm_target_horizon: Forward return horizon for LightGBM target.
        lightgbm_hold_back_cutoff: Dates after this are excluded from training.
        lightgbm_params: Extra parameters forwarded to LightGBM.
        lightgbm_additional_features: Extra feature columns beyond signals.
    """

    mode: Literal["ic_weighted", "lightgbm"] = "ic_weighted"
    ic_lookback: int = 63
    ic_target_horizon: int = 5
    ic_min_observations: int = 30
    rebalance_frequency: int = 5
    weight_method: str = "abs_ic"
    signal_columns: list[str] | None = None
    output_column: str = "signal_ensemble"
    track_weights: bool = True

    # LightGBM-specific
    lightgbm_target_horizon: int = 1
    lightgbm_hold_back_cutoff: date = date(2023, 1, 1)
    lightgbm_params: dict[str, Any] = field(default_factory=dict)
    lightgbm_additional_features: list[str] = field(default_factory=list)

    def to_ensemble_config(self) -> EnsembleConfig:
        """Convert to an EnsembleConfig for the IC-weighted ensemble."""
        return EnsembleConfig(
            ic_lookback=self.ic_lookback,
            ic_target_horizon=self.ic_target_horizon,
            ic_min_observations=self.ic_min_observations,
            rebalance_frequency=self.rebalance_frequency,
            weight_method=self.weight_method,
            signal_columns=self.signal_columns,
            output_column=self.output_column,
            track_weights=self.track_weights,
        )


class EnsemblePipeline:
    """Unified ensemble pipeline: IC-weighted or LightGBM.

    Supports two ensemble modes, a single ``run()`` entry point, and
    optional validation of the ensemble against individual signals.

    Args:
        config: Pipeline configuration. Defaults to IC-weighted mode.
    """

    def __init__(self, config: EnsemblePipelineConfig | None = None) -> None:
        self.config = config or EnsemblePipelineConfig()
        self._ic_ensemble: ICWeightedEnsemble | None = None

    def run(
        self,
        df: pl.DataFrame,
        *,
        train_end: str | date | None = None,
        val_start: str | date | None = None,
    ) -> pl.DataFrame:
        """Run the ensemble pipeline on a signal-enriched DataFrame.

        Args:
            df: DataFrame sorted by (ticker, date) containing signal columns
                and forward return columns.
            train_end: Last training date (exclusive). Only used in LightGBM
                mode. Defaults to the hold-back cutoff.
            val_start: First validation date (inclusive). Only used in
                LightGBM mode.

        Returns:
            DataFrame with the ensemble output column appended. Column name
            depends on mode: ``signal_ensemble`` (IC-weighted) or
            ``ensemble_prediction`` (LightGBM).
        """
        if self.config.mode == "ic_weighted":
            return self._run_ic_weighted(df)
        elif self.config.mode == "lightgbm":
            return self._run_lightgbm(df, train_end=train_end, val_start=val_start)
        else:
            raise ValueError(f"Unknown ensemble mode: {self.config.mode!r}")

    def validate(
        self,
        df: pl.DataFrame,
        *,
        ensemble_col: str | None = None,
        target_col: str | None = None,
        horizons: Sequence[int] | None = None,
    ) -> EnsembleMetrics:
        """Validate the ensemble against individual signals.

        Computes IC, win rate, and CS IC for the ensemble and each
        individual signal, then checks whether the ensemble dominates
        all individual signals.

        Args:
            df: DataFrame containing the ensemble column, signal columns,
                and forward returns.
            ensemble_col: Override the ensemble column name. Defaults to
                the pipeline's output column.
            target_col: Override the target column. Defaults to the
                pipeline's target horizon.
            horizons: Forward horizons for IC decay analysis.

        Returns:
            EnsembleMetrics with per-signal comparison.
        """
        if ensemble_col is None:
            ensemble_col = self._ensemble_output_col()
        if target_col is None:
            target_col = self._target_col()

        signal_cols = self._resolve_signal_columns(df)
        return validate_ensemble(
            df,
            ensemble_col=ensemble_col,
            signal_cols=signal_cols,
            target_col=target_col,
            horizons=horizons,
        )

    def validate_signals(
        self,
        df: pl.DataFrame,
        *,
        target_col: str | None = None,
        horizons: Sequence[int] | None = None,
    ) -> list[SignalMetrics]:
        """Validate individual signals (without ensemble comparison).

        Args:
            df: DataFrame with signal columns and forward returns.
            target_col: Target column. Defaults to pipeline target horizon.
            horizons: Forward horizons for IC decay.

        Returns:
            List of SignalMetrics sorted by descending flat IC.
        """
        if target_col is None:
            target_col = self._target_col()
        signal_cols = self._resolve_signal_columns(df)
        return validate_signals(
            df,
            signal_cols=signal_cols,
            target_col=target_col,
            horizons=horizons,
        )

    def weight_stats(self) -> list[WeightStats]:
        """Return weight distribution statistics for IC-weighted mode.

        Returns:
            One WeightStats per signal.

        Raises:
            RuntimeError: If the ensemble has not been run yet.
        """
        if self.config.mode != "ic_weighted":
            raise RuntimeError("weight_stats() requires ic_weighted mode")
        if self._ic_ensemble is None:
            raise RuntimeError("Call run() first")
        return weight_report(self._ic_ensemble)

    def report(self, df: pl.DataFrame, **kwargs: Any) -> str:
        """Generate a text validation report.

        Args:
            df: Ensemble-enriched DataFrame (output from ``run()``).
            **kwargs: Passed through to ``validate()``.

        Returns:
            Formatted ensemble validation report string.
        """
        em = self.validate(df, **kwargs)
        return format_ensemble_report(em)

    # ── private ──────────────────────────────────────────────────────

    def _run_ic_weighted(self, df: pl.DataFrame) -> pl.DataFrame:
        """Run IC-weighted ensemble."""
        ec = self.config.to_ensemble_config()

        signal_cols = ec.signal_columns or detect_signal_columns(df.columns)
        if not signal_cols:
            raise ValueError("No signal columns found in DataFrame")

        missing = [c for c in signal_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Signal columns missing from DataFrame: {missing}")

        target_col = f"forward_return_{ec.ic_target_horizon}"
        if target_col not in df.columns:
            from src.signals.base import compute_forward_returns
            df = compute_forward_returns(df, horizons=[ec.ic_target_horizon])

        ensemble = ICWeightedEnsemble(
            ic_window=ec.ic_lookback,
            rebalance_freq=ec.rebalance_frequency,
            weight_method=ec.weight_method,
        )
        self._ic_ensemble = ensemble

        result = ensemble.transform(df, signal_cols, target_col)

        # Rename ensemble_score → configured output column
        if "ensemble_score" in result.columns:
            result = result.rename({"ensemble_score": ec.output_column})

        # Drop weight columns if not tracking
        if not ec.track_weights:
            w_cols = [c for c in result.columns if c.startswith("w_")]
            if w_cols:
                result = result.drop(w_cols)

        return result

    def _run_lightgbm(
        self,
        df: pl.DataFrame,
        *,
        train_end: str | date | None = None,
        val_start: str | date | None = None,
    ) -> pl.DataFrame:
        """Run LightGBM ensemble."""
        try:
            from .lightgbm import LightGBMEnsemble, LightGBMEnsembleConfig
        except ImportError:
            raise ImportError(
                "LightGBM is not installed. Install with: uv add lightgbm scikit-learn"
            )

        lgb_config = LightGBMEnsembleConfig(
            signal_columns=self.config.signal_columns or [],
            target_horizon=self.config.lightgbm_target_horizon,
            hold_back_cutoff=self.config.lightgbm_hold_back_cutoff,
            lgbm_params=self.config.lightgbm_params,
            additional_features=self.config.lightgbm_additional_features,
        )

        ensemble = LightGBMEnsemble(config=lgb_config)
        ensemble.fit(df, train_end=train_end, val_start=val_start)
        return ensemble.predict(df)

    def _ensemble_output_col(self) -> str:
        """Return the expected ensemble output column name for the current mode."""
        if self.config.mode == "lightgbm":
            return "ensemble_prediction"
        return self.config.output_column

    def _target_col(self) -> str:
        """Return the expected target forward-return column."""
        if self.config.mode == "lightgbm":
            return f"forward_return_{self.config.lightgbm_target_horizon}"
        return f"forward_return_{self.config.ic_target_horizon}"

    def _resolve_signal_columns(self, df: pl.DataFrame) -> list[str]:
        """Resolve signal columns from config or auto-detection."""
        if self.config.signal_columns:
            return list(self.config.signal_columns)
        return detect_signal_columns(df.columns)


# ── Legacy alias ─────────────────────────────────────────────────────

def run_ensemble(
    df: pl.DataFrame,
    config: EnsembleConfig | None = None,
) -> pl.DataFrame:
    """Legacy: run IC-weighted ensemble pipeline.

    This is a compatibility wrapper around ``EnsemblePipeline``.
    New code should use ``EnsemblePipeline`` directly.
    """
    if config is None:
        config = EnsembleConfig()

    pipeline_config = EnsemblePipelineConfig(
        mode="ic_weighted",
        ic_lookback=config.ic_lookback,
        ic_target_horizon=config.ic_target_horizon,
        ic_min_observations=config.ic_min_observations,
        rebalance_frequency=config.rebalance_frequency,
        weight_method=config.weight_method,
        signal_columns=config.signal_columns,
        output_column=config.output_column,
        track_weights=config.track_weights,
    )

    pipeline = EnsemblePipeline(pipeline_config)
    return pipeline.run(df)