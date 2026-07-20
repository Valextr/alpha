"""LightGBM meta-learner for combining signals.

Trains a LightGBM binary classifier on signal outputs as features,
targeting next-period return direction (positive = 1, negative = 0).

Key design decisions:
    - **Panel-aware**: data is structured as (ticker, date) rows. We train
      on historical dates and evaluate on future dates — no time leakage.
    - **Cross-sectional**: predictions are emitted per row, then the caller
      can rank cross-sectionally for portfolio construction.
    - **Walk-forward ready**: `fit()` accepts explicit train/test date
      boundaries so the same model can be retrained on rolling windows.
    - **Signal columns detected automatically**: any column matching
      `signal_<name>` is treated as a feature.
    - **Hold-back aware**: the cutoff date (2023-01-01) is a class
      constant. Fits that cross the cutoff are rejected.

Dependencies:
    lightgbm, scikit-learn, numpy, polars
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Sequence

import numpy as np
import polars as pl

import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss


# Hold-back cutoff — no training data may cross this boundary.
HOLD_BACK_CUTOFF = date(2023, 1, 1)

# Default LightGBM parameters — conservative defaults to limit overfitting.
_DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 16,           # small — prevent overfitting on few signals
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_jobs": 1,
    "random_state": 42,
}


@dataclass(frozen=True)
class LightGBMEnsembleConfig:
    """Configuration for the LightGBM meta-learner."""

    signal_columns: list[str] = field(
        default_factory=list  # auto-detected if empty
    )
    target_horizon: int = 1        # forward return horizon in days
    hold_back_cutoff: date = HOLD_BACK_CUTOFF
    lgbm_params: dict[str, Any] = field(default_factory=dict)
    additional_features: list[str] = field(
        default_factory=list  # extra columns to include beyond signals
    )


class LightGBMEnsemble:
    """Gradient-boosted meta-learner over signal features.

    Usage:
        ensemble = LightGBMEnsemble()
        ensemble.fit(df, train_end="2020-12-31")
        probs = ensemble.predict(df)
        importances = ensemble.feature_importance()
    """

    def __init__(self, config: LightGBMEnsembleConfig | None = None) -> None:
        self.config = config or LightGBMEnsembleConfig()
        self.model_: lgb.LGBMClassifier | None = None
        self._feature_cols: list[str] = []
        self._fitted: bool = False
        self._train_metrics: dict[str, float] = {}

    # ── public API ──────────────────────────────────────────────────

    def fit(
        self,
        df: pl.DataFrame,
        train_end: str | date | None = None,
        val_start: str | date | None = None,
    ) -> "LightGBMEnsemble":
        """Train the ensemble on a time-bounded training window.

        Args:
            df: DataFrame with signal columns and forward returns,
                sorted by (ticker, date).
            train_end: Last date to include in training (exclusive).
                Rows where `date < train_end` go into training.
            val_start: First date for the validation split (inclusive).
                Rows where `date >= val_start` AND `date < train_end` are
                ignored. If `val_start` is before `train_end`, it carves
                out a gap between train and val. If `val_start` is after
                `train_end`, everything after `train_end` becomes validation.

        Returns:
            Self (for chaining).

        Raises:
            ValueError: If the training window crosses the hold-back cutoff,
                or if required columns are missing.
        """
        if train_end is None:
            train_end = self.config.hold_back_cutoff

        # Normalise to Python date objects for Polars comparison
        if isinstance(train_end, str):
            train_end_date = date.fromisoformat(train_end)
        else:
            train_end_date = train_end

        # Auto-detect signal columns
        feature_cols = self._resolve_feature_columns(df)
        target_col = f"forward_return_{self.config.target_horizon}"

        # Validate required columns exist
        missing = [c for c in feature_cols + [target_col, "date"] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Parse dates for cutoff check
        dates = df["date"].to_list()
        min_date = min(dates)
        max_date = max(dates)

        # Safety: reject if training window extends past hold-back cutoff
        cutoff = self.config.hold_back_cutoff
        if train_end_date > cutoff:
            raise ValueError(
                f"Training window ends after hold-back cutoff "
                f"({train_end_date} > {cutoff}). "
                "This would contaminate the hold-back set."
            )

        # Build train mask using pl.lit() for proper date comparison
        train_mask = pl.col("date") < pl.lit(train_end_date)

        # Build val mask if val_start provided
        val_mask = None
        val_date = None
        if val_start is not None:
            val_date = date.fromisoformat(val_start) if isinstance(val_start, str) else val_start
            val_mask = pl.col("date") >= pl.lit(val_date)
            # If val_start is before train_end, train ends at val_start
            if val_date < train_end_date:
                train_mask = pl.col("date") < pl.lit(val_date)

        # Extract training data
        train_df = df.filter(train_mask)
        if train_df.is_empty():
            raise ValueError(
                f"No training data: train_end={train_end_date}, "
                f"data spans {min_date} to {max_date}"
            )

        X_train = self._extract_features(train_df, feature_cols)
        y_train = self._extract_target(train_df, target_col)

        # Extract validation data
        X_val, y_val = None, None
        if val_mask is not None:
            val_df = df.filter(val_mask)
            if not val_df.is_empty():
                X_val = self._extract_features(val_df, feature_cols)
                y_val = self._extract_target(val_df, target_col)

        # Filter out rows where target is null (end-of-period rows)
        valid = ~np.isnan(y_train)
        X_train = X_train[valid]
        y_train = y_train[valid]

        if len(X_train) == 0:
            raise ValueError("No valid training rows after filtering NaN targets")

        # Convert target: positive return → 1, negative → 0
        y_train_binary = (y_train > 0).astype(int)

        # Train
        params = {**_DEFAULT_PARAMS, **self.config.lgbm_params}
        self.model_ = lgb.LGBMClassifier(**params)
        self.model_.fit(X_train, y_train_binary)

        # Evaluate on training set
        train_probs = self.model_.predict_proba(X_train)[:, 1]
        self._train_metrics["train_auc"] = float(roc_auc_score(y_train_binary, train_probs))
        self._train_metrics["train_logloss"] = float(log_loss(y_train_binary, train_probs))

        # Evaluate on validation set if available
        if X_val is not None and y_val is not None:
            valid_val = ~np.isnan(y_val)
            if valid_val.any():
                X_val_clean = X_val[valid_val]
                y_val_clean = y_val[valid_val]
                y_val_binary = (y_val_clean > 0).astype(int)
                val_probs = self.model_.predict_proba(X_val_clean)[:, 1]
                self._train_metrics["val_auc"] = float(roc_auc_score(y_val_binary, val_probs))
                self._train_metrics["val_logloss"] = float(log_loss(y_val_binary, val_probs))

        self._feature_cols = feature_cols
        self._fitted = True
        return self

    def predict(
        self,
        df: pl.DataFrame,
        return_column: str = "ensemble_prediction",
    ) -> pl.DataFrame:
        """Predict return direction probability for each row.

        Args:
            df: DataFrame with signal columns (same schema as training).
            return_column: Name for the output probability column.

        Returns:
            DataFrame with `return_column` appended (probability in [0, 1]).
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        feature_cols = self._resolve_feature_columns(df)
        X = self._extract_features(df, feature_cols)

        probs = self.model_.predict_proba(X)[:, 1]

        # Replace NaN for rows with missing features
        # (e.g., early rows before enough lookback)
        mask = np.isnan(probs)
        if mask.any():
            probs[mask] = 0.5  # neutral

        return df.with_columns(
            pl.Series(probs, dtype=pl.Float64).alias(return_column)
        )

    def predict_direction(
        self,
        df: pl.DataFrame,
        threshold: float = 0.5,
        return_column: str = "ensemble_direction",
    ) -> pl.DataFrame:
        """Predict directional signal: +1 if P(up) > threshold, -1 otherwise.

        Args:
            df: DataFrame with signal columns.
            threshold: Probability threshold for positive classification.
            return_column: Name for the output direction column.

        Returns:
            DataFrame with `return_column` in {-1, 0, +1}.
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        feature_cols = self._resolve_feature_columns(df)
        X = self._extract_features(df, feature_cols)

        probs = self.model_.predict_proba(X)[:, 1]

        # Direction: +1 if P(up) > threshold, -1 if P(up) < (1-threshold), 0 otherwise
        direction = np.zeros(len(probs), dtype=np.float64)
        direction[probs > threshold] = 1.0
        direction[probs < (1.0 - threshold)] = -1.0

        # Handle NaN
        mask = np.isnan(direction)
        if mask.any():
            direction[mask] = 0.0

        return df.with_columns(
            pl.Series(direction, dtype=pl.Float64).alias(return_column)
        )

    def feature_importance(self) -> dict[str, float]:
        """Return feature importance scores from the trained model.

        Returns:
            Dict mapping feature column name → importance (split-based).
        """
        if not self._fitted or self.model_ is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        importances = self.model_.feature_importances_
        return dict(zip(self._feature_cols, importances.tolist()))

    def training_metrics(self) -> dict[str, float]:
        """Return training/validation metrics from the last fit."""
        return dict(self._train_metrics)

    # ── internal helpers ────────────────────────────────────────────

    def _resolve_feature_columns(self, df: pl.DataFrame) -> list[str]:
        """Determine feature columns from config or auto-detection."""
        if self.config.signal_columns:
            cols = list(self.config.signal_columns)
        else:
            # Auto-detect: any column starting with `signal_`
            cols = [c for c in df.columns if c.startswith("signal_")]

        # Append additional features
        extra = [c for c in self.config.additional_features if c in df.columns]
        cols = cols + extra

        if not cols:
            raise ValueError(
                "No signal columns found. Either set config.signal_columns "
                "or ensure the DataFrame contains columns prefixed with 'signal_'."
            )

        # Validate all exist
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns missing from DataFrame: {missing}")

        return cols

    def _extract_features(
        self, df: pl.DataFrame, feature_cols: list[str]
    ) -> np.ndarray:
        """Extract feature matrix from DataFrame as numpy array."""
        # Use struct to pack all feature columns, then to_numpy
        return df.select(feature_cols).to_numpy().astype(np.float64)

    def _extract_target(self, df: pl.DataFrame, target_col: str) -> np.ndarray:
        """Extract target array from DataFrame."""
        return df[target_col].to_numpy().astype(np.float64)