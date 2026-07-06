from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import polars as pl


class Feature(ABC):
    """Abstract base class for all features."""

    @abstractmethod
    def compute(self, df):
        """Compute feature columns and return enriched DataFrame."""
        pass


def safe_rolling(series, func, window, min_samples=None):
    """Wrapper for rolling operations that handles edge cases."""
    if min_samples is None:
        min_samples = max(1, window // 2)

    kwargs = {"window_size": window, "min_samples": min_samples}

    if func == "std":
        return series.rolling_std(**kwargs)
    elif func == "mean":
        return series.rolling_mean(**kwargs)
    elif func == "min":
        return series.rolling_min(**kwargs)
    elif func == "max":
        return series.rolling_max(**kwargs)
    elif func == "sum":
        return series.rolling_sum(**kwargs)
    else:
        raise ValueError(f"Unknown rolling function: {func}")


def compute_cross_sectional(df, value_col, method="zscore", group_by="date"):
    """Compute cross-sectional statistics per date group.

    All computations are point-in-time correct: statistics are computed
    within each date group, never across dates.

    Args:
        df: Input DataFrame
        value_col: Column to compute statistics on
        method: "zscore", "rank", or "percentile"
        group_by: Column to group by (default: "date")

    Returns:
        DataFrame with new column named "{value_col}_{method}"
    """
    out_col = f"{value_col}_{method}"

    if method == "zscore":
        # Use group-by aggregation instead of map_groups (removed in Polars 1.21+)
        stats = (
            df.group_by(group_by)
            .agg([
                pl.col(value_col).mean().alias("_mean"),
                pl.col(value_col).std().alias("_std"),
            ])
            .with_columns(pl.col("_std").replace(0, None))
        )
        return (
            df.join(stats, on=group_by, how="left")
            .with_columns(
                ((pl.col(value_col) - pl.col("_mean")) / pl.col("_std")).alias(out_col)
            )
            .drop("_mean", "_std")
        )
    elif method == "rank":
        return df.with_columns(
            (
                pl.col(value_col).rank("average").over(group_by)
                / pl.col(value_col).count().over(group_by)
            ).alias(out_col)
        )
    elif method == "percentile":
        return df.with_columns(
            (
                pl.col(value_col).rank("average").over(group_by)
                / pl.col(value_col).count().over(group_by) * 100
            ).alias(out_col)
        )
    else:
        raise ValueError(f"Unknown cross-sectional method: {method}")


def validate_features(df):
    """Validate a feature-enriched DataFrame.

    Checks for: excessive NaN rates, Inf values, constant columns,
    and extremely large values.
    """
    n_rows = len(df)
    if n_rows == 0:
        return {
            "nan_rates": {},
            "inf_columns": [],
            "constant_columns": [],
            "large_value_columns": [],
        }

    numeric_cols = [
        c for c in df.columns
        if df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    ]

    nan_rates = {}
    inf_columns = []
    constant_columns = []
    large_value_columns = []

    for col in numeric_cols:
        series = df[col]

        nan_rate = series.null_count() / n_rows
        if nan_rate > 0.01:
            nan_rates[col] = nan_rate

        if series.dtype in (pl.Float32, pl.Float64):
            if series.is_infinite().any():
                inf_columns.append(col)

        std_val = series.std()
        if std_val is not None and std_val == 0.0:
            constant_columns.append(col)

        if series.dtype in (pl.Float32, pl.Float64):
            if (series.abs() > 1e6).any():
                large_value_columns.append(col)

    return {
        "nan_rates": nan_rates,
        "inf_columns": inf_columns,
        "constant_columns": constant_columns,
        "large_value_columns": large_value_columns,
    }
