"""Drawdown circuit breaker.

Monitors portfolio equity curve and reduces leverage when drawdown
exceeds the configured threshold. Recovery requires the portfolio to
climb back a fraction of the drawdown before leverage is restored.

This prevents catastrophic losses during regime shifts and protects
capital when the strategy encounters unexpected market conditions.
"""

from __future__ import annotations

import polars as pl

from .config import RiskConfig


def drawdown_circuit_breaker(
    equity_curve: pl.DataFrame,
    config: RiskConfig,
    *,
    weight_col: str = "weight",
    equity_col: str = "equity",
) -> pl.DataFrame:
    """Apply drawdown-based leverage reduction.

    For each date:
    1. Compute peak equity and current drawdown
    2. If drawdown > max_drawdown, scale weights down proportionally
    3. Only restore when drawdown recovers by drawdown_recovery_pct

    Args:
        equity_curve: DataFrame with date, equity columns and weight
            column. One row per date (portfolio-level).
        config: RiskConfig with max_drawdown and drawdown_recovery_pct.
        weight_col: Name of the weight column to scale.
        equity_col: Name of the equity value column.

    Returns:
        DataFrame with 'drawdown', 'peak_equity', 'leverage_scale', and
        scaled weight columns.
    """
    df = equity_curve.sort("date").clone()

    # Manual peak computation (cumulative max)
    peaks = []
    running_max = float("-inf")
    for row in df.iter_rows(named=True):
        eq = row[equity_col]
        if eq > running_max:
            running_max = eq
        peaks.append(running_max)

    df = df.with_columns(
        pl.Series("peak_equity", peaks, dtype=pl.Float64),
    )

    # Compute drawdown
    df = df.with_columns(
        ((pl.col("peak_equity") - pl.col(equity_col)) / pl.col("peak_equity"))
        .alias("drawdown"),
    )

    # Determine leverage scale
    # When drawdown > threshold, scale = 1 - drawdown_excess / max_drawdown
    # Minimum scale is 0.25 (never reduce to zero completely)
    recovery_threshold = config.max_drawdown * config.drawdown_recovery_pct

    df = df.with_columns(
        pl.when(pl.col("drawdown") > config.max_drawdown)
        .then(
            # Scale down: the deeper the drawdown, the more we reduce
            (1.0 - (pl.col("drawdown") - config.max_drawdown) / config.max_drawdown)
            .clip(lower_bound=0.25)
        )
        .when(
            (pl.col("drawdown") <= config.max_drawdown)
            & (pl.col("drawdown") > recovery_threshold)
        )
        .then(
            # Recovery zone: gradually restore from reduced state
            ((pl.col("drawdown") - recovery_threshold)
             / (config.max_drawdown - recovery_threshold) + 0.25)
            .clip(lower_bound=0.25, upper_bound=1.0)
        )
        .otherwise(pl.lit(1.0))
        .alias("leverage_scale")
    )

    # Apply scaling to weight
    if weight_col in df.columns:
        df = df.with_columns(
            (pl.col(weight_col) * pl.col("leverage_scale")).alias(weight_col)
        )

    return df.select(["date", "drawdown", "peak_equity", "leverage_scale", weight_col])


def compute_drawdown_series(
    df: pl.DataFrame,
    *,
    equity_col: str = "equity",
) -> pl.DataFrame:
    """Compute drawdown series from an equity curve.

    Args:
        df: DataFrame with date and equity (or returns) columns.
        equity_col: Name of the equity value column.
            If values look like returns (< 1.0), they are cumulative-product'd
            into an equity curve first. Otherwise treated as raw equity.

    Returns:
        DataFrame with date, equity, peak_equity, drawdown columns.
    """
    df = df.sort("date").clone()

    eq_col = equity_col
    # If values look like returns (mean < 0.1), compute equity curve
    if df[equity_col].mean() < 0.1:
        df = df.with_columns(
            ((1 + pl.col(equity_col)).cum_prod()).alias("equity")
        )
        eq_col = "equity"

    # Compute peak and drawdown
    peaks = []
    running_max = float("-inf")
    for row in df.iter_rows(named=True):
        eq = row[eq_col]
        if eq > running_max:
            running_max = eq
        peaks.append(running_max)

    return (
    df.with_columns(pl.Series("peak_equity", peaks, dtype=pl.Float64))
    .with_columns(
        ((pl.col("peak_equity") - pl.col(eq_col)) / pl.col("peak_equity"))
        .alias("drawdown"),
    )
    .select(["date", eq_col, "peak_equity", "drawdown"])
    .rename({eq_col: "equity"})
)