from __future__ import annotations

import math

import polars as pl

from .registry import registry


@registry.register(
    "vol_5d",
    description="5-day realized volatility (rolling std of 1d log returns)",
    category="volatility",
    lookback=5,
    depends_on=["log_return_1d"],
)
def compute_vol_5d(df):
    return df.with_columns(
        pl.col("log_return_1d")
        .rolling_std(window_size=5, min_samples=3)
        .over("ticker")
        .alias("vol_5d")
    )


@registry.register(
    "vol_21d",
    description="21-day realized volatility",
    category="volatility",
    lookback=21,
    depends_on=["log_return_1d"],
)
def compute_vol_21d(df):
    return df.with_columns(
        pl.col("log_return_1d")
        .rolling_std(window_size=21, min_samples=10)
        .over("ticker")
        .alias("vol_21d")
    )


@registry.register(
    "vol_63d",
    description="63-day realized volatility",
    category="volatility",
    lookback=63,
    depends_on=["log_return_1d"],
)
def compute_vol_63d(df):
    return df.with_columns(
        pl.col("log_return_1d")
        .rolling_std(window_size=63, min_samples=30)
        .over("ticker")
        .alias("vol_63d")
    )


@registry.register(
    "vol_annual",
    description="Annualized volatility: vol_21d * sqrt(252)",
    category="volatility",
    lookback=21,
    depends_on=["vol_21d"],
)
def compute_vol_annual(df):
    return df.with_columns(
        (pl.col("vol_21d") * math.sqrt(252)).alias("vol_annual")
    )


@registry.register(
    "vol_of_vol_21d",
    description="Vol of vol (21d rolling std of 5d vol) - measures regime stability",
    category="volatility",
    lookback=25,
    depends_on=["vol_5d"],
)
def compute_vol_of_vol_21d(df):
    return df.with_columns(
        pl.col("vol_5d")
        .rolling_std(window_size=21, min_samples=10)
        .over("ticker")
        .alias("vol_of_vol_21d")
    )


@registry.register(
    "vol_ratio_short_long",
    description="Short-term vol / long-term vol ratio - spikes during regime changes",
    category="volatility",
    lookback=63,
    depends_on=["vol_5d", "vol_63d"],
)
def compute_vol_ratio_short_long(df):
    return df.with_columns(
        (pl.col("vol_5d") / pl.col("vol_63d")).replace(0, None).alias(
            "vol_ratio_short_long"
        )
    )


VOLATILITY_FEATURES = [
    compute_vol_5d,
    compute_vol_21d,
    compute_vol_63d,
    compute_vol_annual,
    compute_vol_of_vol_21d,
    compute_vol_ratio_short_long,
]