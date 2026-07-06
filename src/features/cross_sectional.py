from __future__ import annotations

import polars as pl

from .registry import registry


def _cs_zscore(df, value_col, out_col):
    """Compute cross-sectional z-score of a column per date.

    Uses group-by aggregation instead of map_groups (removed in Polars 1.21+).
    """
    stats = (
        df.group_by("date")
        .agg([
            pl.col(value_col).mean().alias("_mean"),
            pl.col(value_col).std().alias("_std"),
        ])
        .with_columns(pl.col("_std").replace(0, None))
    )

    return (
        df.join(stats, on="date", how="left")
        .with_columns(
            ((pl.col(value_col) - pl.col("_mean")) / pl.col("_std")).alias(out_col)
        )
        .drop("_mean", "_std")
    )


def _cs_rank(df, value_col, out_col):
    """Compute cross-sectional rank (0-1) of a column per date."""
    return df.with_columns(
        (
            pl.col(value_col).rank("average").over("date")
            / pl.col(value_col).count().over("date")
        ).alias(out_col)
    )


@registry.register(
    "cs_return_zscore_21d",
    description="Cross-sectional z-score of 21-day returns",
    category="cross_sectional",
    lookback=21,
    depends_on=["return_21d"],
)
def compute_cs_return_zscore_21d(df):
    """How did this ticker's 21d return rank vs peers on each date?"""
    return _cs_zscore(df, "return_21d", "cs_return_zscore_21d")


@registry.register(
    "cs_return_rank_21d",
    description="Cross-sectional rank (0-1) of 21-day returns",
    category="cross_sectional",
    lookback=21,
    depends_on=["return_21d"],
)
def compute_cs_return_rank_21d(df):
    """Percentile rank of 21d return within the universe per date."""
    return _cs_rank(df, "return_21d", "cs_return_rank_21d")


@registry.register(
    "cs_vol_rank_21d",
    description="Cross-sectional rank (0-1) of 21-day volatility",
    category="cross_sectional",
    lookback=21,
    depends_on=["vol_21d"],
)
def compute_cs_vol_rank_21d(df):
    """Low-vol anomaly signal: percentile rank of 21d vol within the universe."""
    return _cs_rank(df, "vol_21d", "cs_vol_rank_21d")


@registry.register(
    "cs_volume_rank_21d",
    description="Cross-sectional rank (0-1) of relative volume",
    category="cross_sectional",
    lookback=21,
    depends_on=["relative_volume_21d"],
)
def compute_cs_volume_rank_21d(df):
    """Percentile rank of relative volume within the universe per date."""
    return _cs_rank(df, "relative_volume_21d", "cs_volume_rank_21d")


@registry.register(
    "sector_relative_return_21d",
    description="21-day return minus sector median return",
    category="cross_sectional",
    lookback=21,
    depends_on=["return_21d"],
)
def compute_sector_relative_return_21d(df):
    """How did this ticker perform vs its sector peers over 21 days?

    Uses the 'sector' column from the gold layer. If sector is missing
    or 'Unknown', falls back to universe median.
    """
    sector_median = (
        df.group_by("date", "sector")
        .agg(pl.col("return_21d").median().alias("sector_return_median"))
    )

    return (
        df.join(sector_median, on=["date", "sector"], how="left")
        .with_columns(
            (pl.col("return_21d") - pl.col("sector_return_median")).alias(
                "sector_relative_return_21d"
            )
        )
        .drop("sector_return_median")
    )


CROSS_SECTIONAL_FEATURES = [
    compute_cs_return_zscore_21d,
    compute_cs_return_rank_21d,
    compute_cs_vol_rank_21d,
    compute_cs_volume_rank_21d,
    compute_sector_relative_return_21d,
]
