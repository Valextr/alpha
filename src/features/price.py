from __future__ import annotations

import polars as pl

from .registry import registry


@registry.register(
    "return_1d",
    description="1-day simple return: (close / prev_close) - 1",
    category="price",
    lookback=1,
)
def compute_return_1d(df):
    return df.with_columns(
        ((pl.col("close") / pl.col("close").shift(1)) - 1).alias("return_1d")
    )


@registry.register(
    "return_5d",
    description="5-day simple return",
    category="price",
    lookback=5,
)
def compute_return_5d(df):
    return df.with_columns(
        ((pl.col("close") / pl.col("close").shift(5)) - 1).alias("return_5d")
    )


@registry.register(
    "return_21d",
    description="21-day (approx 1-month) simple return",
    category="price",
    lookback=21,
)
def compute_return_21d(df):
    return df.with_columns(
        ((pl.col("close") / pl.col("close").shift(21)) - 1).alias("return_21d")
    )


@registry.register(
    "return_63d",
    description="63-day (approx 3-month) simple return",
    category="price",
    lookback=63,
)
def compute_return_63d(df):
    return df.with_columns(
        ((pl.col("close") / pl.col("close").shift(63)) - 1).alias("return_63d")
    )


@registry.register(
    "log_return_1d",
    description="1-day log return: ln(close / prev_close)",
    category="price",
    lookback=1,
)
def compute_log_return_1d(df):
    return df.with_columns(
        (pl.col("close") / pl.col("close").shift(1)).log().alias("log_return_1d")
    )


@registry.register(
    "log_return_5d",
    description="5-day log return",
    category="price",
    lookback=5,
)
def compute_log_return_5d(df):
    return df.with_columns(
        (pl.col("close") / pl.col("close").shift(5)).log().alias("log_return_5d")
    )


@registry.register(
    "log_return_21d",
    description="21-day log return",
    category="price",
    lookback=21,
)
def compute_log_return_21d(df):
    return df.with_columns(
        (pl.col("close") / pl.col("close").shift(21)).log().alias("log_return_21d")
    )


@registry.register(
    "log_return_63d",
    description="63-day log return",
    category="price",
    lookback=63,
)
def compute_log_return_63d(df):
    return df.with_columns(
        (pl.col("close") / pl.col("close").shift(63)).log().alias("log_return_63d")
    )


@registry.register(
    "cum_return_5d",
    description="5-day cumulative return (compounded)",
    category="price",
    lookback=5,
)
def compute_cum_return_5d(df):
    return df.with_columns(
        (pl.col("close") / pl.col("close").shift(5)).alias("cum_return_5d")
    )


@registry.register(
    "cum_return_21d",
    description="21-day cumulative return (compounded)",
    category="price",
    lookback=21,
)
def compute_cum_return_21d(df):
    return df.with_columns(
        (pl.col("close") / pl.col("close").shift(21)).alias("cum_return_21d")
    )


@registry.register(
    "cum_return_63d",
    description="63-day cumulative return (compounded)",
    category="price",
    lookback=63,
)
def compute_cum_return_63d(df):
    return df.with_columns(
        (pl.col("close") / pl.col("close").shift(63)).alias("cum_return_63d")
    )


@registry.register(
    "drawdown_from_peak",
    description="Drawdown from 252-day rolling peak: (close / rolling_max) - 1",
    category="price",
    lookback=252,
)
def compute_drawdown_from_peak(df):
    return df.with_columns(
        (
            pl.col("close")
            / pl.col("close").rolling_max(window_size=252, min_periods=1)
            - 1
        ).alias("drawdown_from_peak")
    )


@registry.register(
    "price_displacement_5d",
    description="Absolute price displacement over 5 days",
    category="price",
    lookback=5,
)
def compute_price_displacement_5d(df):
    return df.with_columns(
        (
            (pl.col("close") - pl.col("close").shift(5))
            / pl.col("close").shift(5)
        ).abs().alias("price_displacement_5d")
    )


@registry.register(
    "price_displacement_21d",
    description="Absolute price displacement over 21 days",
    category="price",
    lookback=21,
)
def compute_price_displacement_21d(df):
    return df.with_columns(
        (
            (pl.col("close") - pl.col("close").shift(21))
            / pl.col("close").shift(21)
        ).abs().alias("price_displacement_21d")
    )


PRICE_FEATURES = [
    compute_return_1d,
    compute_return_5d,
    compute_return_21d,
    compute_return_63d,
    compute_log_return_1d,
    compute_log_return_5d,
    compute_log_return_21d,
    compute_log_return_63d,
    compute_cum_return_5d,
    compute_cum_return_21d,
    compute_cum_return_63d,
    compute_drawdown_from_peak,
    compute_price_displacement_5d,
    compute_price_displacement_21d,
]
