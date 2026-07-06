from __future__ import annotations

import polars as pl

from .registry import registry


@registry.register(
    "relative_volume_21d",
    description="Today's volume / 21-day average volume",
    category="volume",
    lookback=21,
)
def compute_relative_volume_21d(df):
    """Relative volume: today's volume divided by the 21-day rolling mean."""
    return df.with_columns(
        (
            pl.col("volume")
            / pl.col("volume").rolling_mean(window_size=21, min_samples=10)
        ).alias("relative_volume_21d")
    )


@registry.register(
    "volume_zscore_63d",
    description="Volume z-score over 63-day window",
    category="volume",
    lookback=63,
)
def compute_volume_zscore_63d(df):
    """How unusual today's volume is relative to the 63-day distribution."""
    mean_63 = pl.col("volume").rolling_mean(window_size=63, min_samples=30)
    std_63 = pl.col("volume").rolling_std(window_size=63, min_samples=30)
    return df.with_columns(
        ((pl.col("volume") - mean_63) / std_63.replace(0, None)).alias(
            "volume_zscore_63d"
        )
    )


@registry.register(
    "accumulation_distribution",
    description="Accumulation/Distribution proxy: price-volume flow indicator",
    category="volume",
    lookback=1,
)
def compute_accumulation_distribution(df):
    """Money flow proxy: where did close land within the day's range, weighted by volume.

    Positive = buying pressure (close near high).
    Negative = selling pressure (close near low).
    """
    high_low = (pl.col("high") - pl.col("low")).replace(0, None)
    mf_ratio = (
        (pl.col("close") - pl.col("low")) - (pl.col("high") - pl.col("close"))
    ) / high_low
    return df.with_columns(
        (mf_ratio * pl.col("volume")).alias("accumulation_distribution")
    )


@registry.register(
    "volume_shock",
    description="Binary flag: volume exceeds 63-day mean + 2*std",
    category="volume",
    lookback=63,
)
def compute_volume_shock(df):
    """Detect unusual volume spikes (volume > mean + 2*std over 63 days)."""
    mean_63 = pl.col("volume").rolling_mean(window_size=63, min_samples=30)
    std_63 = pl.col("volume").rolling_std(window_size=63, min_samples=30)
    threshold = mean_63 + 2 * std_63
    return df.with_columns(
        pl.when(pl.col("volume") > threshold)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("volume_shock")
    )


VOLUME_FEATURES = [
    compute_relative_volume_21d,
    compute_volume_zscore_63d,
    compute_accumulation_distribution,
    compute_volume_shock,
]
