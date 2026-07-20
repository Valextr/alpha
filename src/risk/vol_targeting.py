"""Volatility targeting.

Scales portfolio weights to achieve a target annualized volatility.

Uses rolling realized volatility per ticker, then scales weights so the
weighted portfolio volatility matches the target. This is a crucial
risk control — without it, a system trained on low-volatility regimes
can blow up when market volatility spikes.
"""

from __future__ import annotations

import polars as pl

from .config import RiskConfig


def compute_realized_volatility(
    returns_df: pl.DataFrame,
    *,
    lookback: int = 63,
    annualization_factor: float = 252.0,
) -> pl.DataFrame:
    """Compute rolling realized volatility per ticker.

    Args:
        returns_df: DataFrame with ticker, date, and 'return' columns.
        lookback: Rolling window size (trading days).
        annualization_factor: Days per year for annualization.

    Returns:
        DataFrame with ticker, date, and 'realized_vol' columns.
    """
    return (
        returns_df.sort(["ticker", "date"])
        .with_columns(
            pl.col("return")
            .rolling_std(window_size=lookback, min_samples=20)
            .over("ticker")
            .sqrt()
            .alias("_daily_vol")
        )
        .with_columns(
            (pl.col("_daily_vol") * pl.lit(annualization_factor**0.5))
            .alias("realized_vol")
        )
        .select(["ticker", "date", "realized_vol"])
    )


def volatility_target(
    weights_df: pl.DataFrame,
    returns_df: pl.DataFrame,
    config: RiskConfig,
    *,
    weight_col: str = "weight",
    return_col: str = "return",
) -> pl.DataFrame:
    """Scale weights to achieve target portfolio volatility.

    For each date:
    1. Compute realized volatility per ticker
    2. Estimate portfolio volatility as weighted sum of ticker vols
    3. Scale all weights by (target_vol / realized_portfolio_vol)

    If realized volatility is very low (< 1%), scale factor is capped at 2.0
    to avoid excessive leverage on quiet markets.

    Args:
        weights_df: DataFrame with ticker, date, weight columns.
        returns_df: DataFrame with ticker, date, and return columns.
        config: RiskConfig with target_vol and lookback_vol.
        weight_col: Name of the weight column.
        return_col: Name of the return column in returns_df.

    Returns:
        DataFrame with volatility-targeted weights.
    """
    # Compute realized volatility per ticker
    vols = compute_realized_volatility(
        returns_df.with_columns(pl.col(return_col).alias("return")),
        lookback=config.lookback_vol,
    )

    # Join with weights
    combined = weights_df.join(
        vols,
        on=["ticker", "date"],
        how="left",
    )

    # Estimate portfolio volatility per date (weighted sum of vols)
    combined = combined.with_columns(
        (pl.col(weight_col).abs() * pl.col("realized_vol").fill_null(config.target_vol))
        .alias("_weighted_vol")
    )

    port_vol = (
        combined.group_by("date")
        .agg(pl.col("_weighted_vol").sum().alias("_port_vol"))
    )

    # Compute scale factor: target / realized, capped at 2.0x
    scale = port_vol.with_columns(
        (pl.lit(config.target_vol) / pl.col("_port_vol").clip(lower_bound=0.01))
        .clip(upper_bound=2.0)
        .alias("_vol_scale")
    )

    # Apply scaling
    result = combined.join(
        scale.select(["date", "_vol_scale"]),
        on="date",
        how="left",
    ).with_columns(
        (pl.col(weight_col) * pl.col("_vol_scale").fill_null(1.0)).alias(weight_col)
    ).select(["ticker", "date", weight_col])

    return result