"""Mean Reversion signal.

Hypothesis:
    Prices that deviate from their local mean tend to revert.

    Uses z-scores on rolling price windows, inspired by López de Prado's
    "Advances in Financial Machine Learning" (Ch. 10: Fractional Differentiation).

    Signal logic:
        - When price is significantly below its rolling mean → long (+)
        - When price is significantly above its rolling mean → short (-)
        - Cross-sectional ranking ensures signals are comparable across tickers

    Signal values are bounded in [-1, +1] via tanh normalization.

Parameters:
    lookback: rolling window for mean/std (default 63 trading days ≈ 3 months)
    cross_sectional: whether to rank the signal cross-sectionally (default True)
"""

from __future__ import annotations

import math

import polars as pl

from .registry import registry


@registry.register(
    "mean_reversion_21d",
    description="21-day mean reversion z-score signal (cross-sectionally ranked)",
    category="mean_reversion",
    parameters={"lookback": 21, "cross_sectional": True},
    requires_features=["log_return_1d"],
)
def generate_mean_reversion_21d(df: pl.DataFrame) -> pl.DataFrame:
    """Generate mean reversion signal using 21-day lookback.

    Steps:
        1. Compute rolling z-score of price relative to its local mean.
        2. Negate (below mean → positive signal = long).
        3. Cross-sectionally rank to [-1, +1].

    Args:
        df: Feature-enriched DataFrame sorted by (ticker, date).

    Returns:
        DataFrame with `signal_mean_reversion_21d` column.
    """
    return _generate_mean_reversion(df, lookback=21, out_col="signal_mean_reversion_21d")


@registry.register(
    "mean_reversion_63d",
    description="63-day mean reversion z-score signal (cross-sectionally ranked)",
    category="mean_reversion",
    parameters={"lookback": 63, "cross_sectional": True},
    requires_features=["log_return_1d"],
)
def generate_mean_reversion_63d(df: pl.DataFrame) -> pl.DataFrame:
    """Generate mean reversion signal using 63-day lookback.

    A longer lookback captures slower reversion cycles.
    """
    return _generate_mean_reversion(df, lookback=63, out_col="signal_mean_reversion_63d")


def _generate_mean_reversion(
    df: pl.DataFrame,
    lookback: int,
    out_col: str,
) -> pl.DataFrame:
    """Core mean reversion signal generation."""
    min_samples = max(1, lookback // 2)

    # Rolling mean and std of close price per ticker
    mean = pl.col("close").rolling_mean(window_size=lookback, min_samples=min_samples).over("ticker")
    std = pl.col("close").rolling_std(window_size=lookback, min_samples=min_samples).over("ticker")

    # Z-score: how far current price is from its rolling mean
    df = df.with_columns(
        ((pl.col("close") - mean) / std.replace(0, None)).alias("_zscore")
    )

    # Negate z-score so that below-mean → positive (long signal)
    # Apply tanh to bound the signal roughly in [-1, +1]
    df = df.with_columns(
        (-pl.col("_zscore")).tanh().alias("_raw_signal")
    )

    # Cross-sectional ranking: normalize each date's signals to [-1, +1]
    # This ensures signals are comparable across tickers on the same date
    date_stats = (
        df.group_by("date")
        .agg([
            pl.col("_raw_signal").mean().alias("_cs_mean"),
            pl.col("_raw_signal").std().alias("_cs_std"),
        ])
        .with_columns(pl.col("_cs_std").replace(0, None))
    )

    df = (
        df.join(date_stats, on="date", how="left")
        .with_columns(
            ((pl.col("_raw_signal") - pl.col("_cs_mean")) / pl.col("_cs_std"))
            .tanh()
            .alias(out_col)
        )
        .drop(["_zscore", "_raw_signal", "_cs_mean", "_cs_std"])
    )

    return df