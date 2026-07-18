from __future__ import annotations

import polars as pl

from .registry import registry


@registry.register(
    "regime_market_trend",
    description="Universe median price trend: SMA50 / SMA200 - 1",
    category="regime",
    lookback=200,
)
def compute_regime_market_trend(df):
    """Market trend proxy using universe median close price.

    Positive = uptrend (short-term MA above long-term MA).
    Negative = downtrend.
    Computed per-date and broadcast to all tickers.
    """
    median_df = (
        df.group_by("date")
        .agg(pl.col("close").median().alias("universe_median_close"))
        .sort("date")
    )

    median_df = median_df.with_columns([
        pl.col("universe_median_close")
        .rolling_mean(window_size=50, min_samples=25)
        .alias("sma_50"),
        pl.col("universe_median_close")
        .rolling_mean(window_size=200, min_samples=100)
        .alias("sma_200"),
    ]).with_columns(
        (pl.col("sma_50") / pl.col("sma_200") - 1).alias("regime_market_trend")
    ).select(["date", "regime_market_trend"])

    return df.join(median_df, on="date", how="left")


@registry.register(
    "regime_vol_state",
    description="Current 21d vol as fraction of 252d peak vol",
    category="regime",
    lookback=252,
    depends_on=["vol_21d"],
)
def compute_regime_vol_state(df):
    """Where does current 21d vol sit relative to the past year's peak?

    0.0 = very low vol. 1.0 = vol at yearly peak.
    """
    return df.with_columns(
        (
            pl.col("vol_21d")
            / pl.col("vol_21d")
            .rolling_max(window_size=252, min_samples=126)
            .over("ticker")
        ).alias("regime_vol_state")
    )


@registry.register(
    "regime_breadth",
    description="Fraction of universe above 20-day SMA per date",
    category="regime",
    lookback=20,
)
def compute_regime_breadth(df):
    """Market participation: what % of tickers are above their 20-day SMA?

    Computed per-date and broadcast to all tickers.
    1.0 = all tickers above SMA (strong participation).
    0.0 = all tickers below SMA (weak participation).
    """
    df_with_flag = df.with_columns(
        pl.when(
            pl.col("close")
            > pl.col("close").rolling_mean(window_size=20, min_samples=10).over("ticker")
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("_above_sma20")
    )

    breadth = (
        df_with_flag.group_by("date")
        .agg(pl.col("_above_sma20").mean().alias("regime_breadth"))
    )

    return df_with_flag.join(breadth, on="date", how="left").drop("_above_sma20")


@registry.register(
    "regime_vol_regime",
    description="Binary flag: volatility regime stable vs unstable",
    category="regime",
    lookback=25,
    depends_on=["vol_of_vol_21d"],
)
def compute_regime_vol_regime(df):
    """Flag unstable volatility regimes.

    1 = unstable (vol_of_vol above its own 252d median).
    0 = stable (vol_of_vol at or below its own 252d median).
    """
    return df.with_columns(
        pl.col("vol_of_vol_21d")
        .rolling_median(window_size=252, min_samples=126)
        .over("ticker")
        .alias("_vol_of_vol_median")
    ).with_columns(
        pl.when(pl.col("vol_of_vol_21d") > pl.col("_vol_of_vol_median"))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("regime_vol_regime")
    ).drop("_vol_of_vol_median")


REGIME_FEATURES = [
    compute_regime_market_trend,
    compute_regime_vol_state,
    compute_regime_breadth,
    compute_regime_vol_regime,
]
