"""Hard constraint enforcement.

Position limits, sector limits, and leverage caps. These are hard
constraints — they truncate or rescale weights, not soft penalties.

All functions operate on weight DataFrames with columns:
    ticker, date, weight

Sector-aware functions also require a 'sector' column.
"""

from __future__ import annotations

import polars as pl

from .config import RiskConfig


def enforce_position_limits(
    weights_df: pl.DataFrame,
    *,
    max_position_pct: float = 0.10,
    weight_col: str = "weight",
) -> pl.DataFrame:
    """Cap individual position weights.

    Any weight exceeding max_position_pct is truncated to that limit.

    This is a hard cap — weights are clipped, not rescaled. Downstream
    leverage and volatility controls handle the residual portfolio sizing.

    Args:
        weights_df: DataFrame with ticker, date, weight columns.
        max_position_pct: Maximum absolute weight per position.
        weight_col: Name of the weight column.

    Returns:
        DataFrame with capped weights.
    """
    return weights_df.with_columns(
        pl.col(weight_col)
        .clip(-max_position_pct, max_position_pct)
        .alias(weight_col)
    )


def enforce_sector_limits(
    weights_df: pl.DataFrame,
    *,
    max_sector_pct: float = 0.30,
    weight_col: str = "weight",
    sector_col: str = "sector",
) -> pl.DataFrame:
    """Cap sector exposure.

    If the sum of absolute weights within a sector exceeds max_sector_pct,
    weights in that sector are scaled down proportionally.

    Args:
        weights_df: DataFrame with ticker, date, weight, and sector columns.
        max_sector_pct: Maximum combined absolute weight per sector.
        weight_col: Name of the weight column.
        sector_col: Name of the sector column.

    Returns:
        DataFrame with sector-constrained weights.
    """
    # Compute sector totals per date
    sector_totals = (
        weights_df.with_columns(
            pl.col(weight_col).abs().alias("_abs_w")
        )
        .group_by([sector_col, "date"])
        .agg(pl.col("_abs_w").sum().alias("_sector_total"))
    )

    # Find sectors that exceed the limit
    violating = sector_totals.filter(pl.col("_sector_total") > max_sector_pct)

    if len(violating) == 0:
        return weights_df

    # Compute scale factor per (sector, date)
    scale_factors = sector_totals.with_columns(
        pl.when(pl.col("_sector_total") > max_sector_pct)
        .then(max_sector_pct / pl.col("_sector_total"))
        .otherwise(pl.lit(1.0))
        .alias("_scale")
    )

    # Apply scaling
    result = weights_df.join(
        scale_factors.select([sector_col, "date", "_scale"]),
        on=[sector_col, "date"],
        how="left",
    ).with_columns(
        (pl.col(weight_col) * pl.col("_scale").fill_null(1.0)).alias(weight_col)
    ).drop(["_scale"])

    return result


def enforce_leverage(
    weights_df: pl.DataFrame,
    *,
    max_leverage: float = 1.5,
    weight_col: str = "weight",
) -> pl.DataFrame:
    """Cap gross leverage.

    Gross leverage = sum of absolute weights. If it exceeds max_leverage,
    all weights are scaled down proportionally.

    Args:
        weights_df: DataFrame with ticker, date, weight columns.
        max_leverage: Maximum gross leverage.
        weight_col: Name of the weight column.

    Returns:
        DataFrame with leverage-constrained weights.
    """
    # Compute gross leverage per date
    leverage = (
        weights_df
        .with_columns(pl.col(weight_col).abs().alias("_abs_w"))
        .group_by("date")
        .agg(pl.col("_abs_w").sum().alias("_gross_leverage"))
    )

    # Compute scale factor
    scale = leverage.with_columns(
        pl.when(pl.col("_gross_leverage") > max_leverage)
        .then(max_leverage / pl.col("_gross_leverage"))
        .otherwise(pl.lit(1.0))
        .alias("_scale")
    )

    # Apply scaling
    result = weights_df.join(
        scale.select(["date", "_scale"]),
        on="date",
        how="left",
    ).with_columns(
        (pl.col(weight_col) * pl.col("_scale").fill_null(1.0)).alias(weight_col)
    ).drop(["_scale"])

    return result


def enforce_all_constraints(
    weights_df: pl.DataFrame,
    config: RiskConfig,
    *,
    weight_col: str = "weight",
    sector_col: str = "sector",
) -> pl.DataFrame:
    """Apply all constraint layers in order.

    Order matters:
    1. Position limits (hard caps on individual weights)
    2. Sector limits (scale down sectors that exceed limits)
    3. Leverage cap (global scaling if total leverage is too high)

    Args:
        weights_df: DataFrame with ticker, date, weight columns.
        config: RiskConfig with all threshold values.
        weight_col: Name of the weight column.
        sector_col: Name of the sector column.

    Returns:
        DataFrame with all constraints enforced.
    """
    result = weights_df.clone()

    # Step 1: Position limits
    result = enforce_position_limits(
        result,
        max_position_pct=config.max_position_pct,
        weight_col=weight_col,
    )

    # Step 2: Sector limits (only if sector column exists)
    if sector_col in result.columns:
        result = enforce_sector_limits(
            result,
            max_sector_pct=config.max_sector_pct,
            weight_col=weight_col,
            sector_col=sector_col,
        )

    # Step 3: Leverage cap
    result = enforce_leverage(
        result,
        max_leverage=config.max_leverage,
        weight_col=weight_col,
    )

    return result