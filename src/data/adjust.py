"""Silver layer: adjust raw prices for splits and dividends.

All adjustments are backward adjustments (prices expressed in current shares).
"""

import logging

import polars as pl

logger = logging.getLogger(__name__)


def adjust_for_splits(
    bars: pl.DataFrame,
    splits: pl.DataFrame,
) -> pl.DataFrame:
    """Adjust OHLCV for stock splits.

    Args:
        bars: Daily bars with columns [ticker, date, open, high, low, close, volume]
        splits: Split events with columns [ticker, date, action_type, factor]

    Returns:
        Bars adjusted backward for all splits.
    """
    if splits.is_empty():
        return bars

    adjusted = bars.clone()

    for ticker in splits["ticker"].unique():
        ticker_splits = (
            splits.filter(pl.col("ticker") == ticker)
            .sort("date", descending=True)
        )

        for split_row in ticker_splits.iter_rows(named=True):
            factor = float(split_row["factor"])
            if factor <= 1.0:
                continue

            split_date = split_row["date"]
            price_cols = ["open", "high", "low", "close"]

            # Adjust all bars on or before the split date for this ticker
            adjusted = adjusted.with_columns(
                [
                    pl.when(
                        (pl.col("ticker") == ticker) & (pl.col("date") <= split_date)
                    )
                    .then(pl.col(c) / factor)
                    .otherwise(pl.col(c))
                    .alias(c)
                    for c in price_cols
                ]
                + [
                    pl.when(
                        (pl.col("ticker") == ticker) & (pl.col("date") <= split_date)
                    )
                    .then(pl.col("volume") * factor)
                    .otherwise(pl.col("volume"))
                    .alias("volume")
                ]
            )

            logger.debug("Adjusted %s for split %.1fx on %s", ticker, factor, split_date)

    return adjusted


def adjust_for_dividends(
    bars: pl.DataFrame,
    dividends: pl.DataFrame,
) -> pl.DataFrame:
    """Adjust closing prices for dividends (price-relative method).

    Note: This is a simplified adjustment. For production, consider using
    the official adjusted close from the data provider.

    Args:
        bars: Daily bars
        dividends: Dividend events with columns [ticker, ex_date, amount]

    Returns:
        Bars with dividend-adjusted prices.
    """
    if dividends.is_empty():
        return bars

    adjusted = bars.clone()

    for ticker in dividends["ticker"].unique():
        ticker_divs = (
            dividends.filter(pl.col("ticker") == ticker)
            .sort("ex_date", descending=True)
        )

        for div_row in ticker_divs.iter_rows(named=True):
            ex_date = div_row["ex_date"]
            amount = float(div_row["amount"])

            if amount <= 0:
                continue

            # Get the close price on the day before ex-date
            prev_close = (
                adjusted
                .filter((pl.col("ticker") == ticker) & (pl.col("date") < ex_date))
                .sort("date", descending=True)
                .select(pl.col("close")).head(1)
            )

            if prev_close.is_empty():
                continue

            prev_close_val = float(prev_close["close"][0])
            if prev_close_val <= 0:
                continue

            adjustment_factor = (prev_close_val - amount) / prev_close_val

            # Adjust all bars on or before ex-date
            price_cols = ["open", "high", "low", "close"]
            mask = (pl.col("ticker") == ticker) & (pl.col("date") <= ex_date)

            adjusted = adjusted.with_columns(
                [
                    pl.when(mask).then(pl.col(c) * adjustment_factor).otherwise(pl.col(c)).alias(c)
                    for c in price_cols
                ]
            )

    return adjusted


def build_silver_layer(
    bronze_bars: pl.DataFrame,
    bronze_dividends: pl.DataFrame,
    bronze_actions: pl.DataFrame,
) -> pl.DataFrame:
    """Build the silver layer from bronze data.

    Pipeline:
    1. Start with raw bronze bars
    2. Adjust for splits
    3. Adjust for dividends
    4. Add quality flags

    Args:
        bronze_bars: Raw daily bars
        bronze_dividends: Raw dividend data
        bronze_actions: Raw corporate action data

    Returns:
        Silver layer DataFrame with adjusted prices and quality flags.
    """
    logger.info("Building silver layer: %d bars, %d dividends, %d actions",
                len(bronze_bars), len(bronze_dividends), len(bronze_actions))

    # Filter actions to only splits
    splits = pl.DataFrame()
    if not bronze_actions.is_empty():
        splits = bronze_actions.filter(pl.col("action_type") == "split")

    # Step 1: Adjust for splits
    adjusted = adjust_for_splits(bronze_bars, splits)

    # Step 2: Adjust for dividends
    adjusted = adjust_for_dividends(adjusted, bronze_dividends)

    # Step 3: Add quality flags
    adjusted = adjusted.with_columns([
        # Forward-adjusted close (same as close after adjustments)
        pl.col("close").alias("adj_close"),

        # Cumulative split factor (placeholder — would need per-ticker tracking)
        pl.lit(1.0, dtype=pl.Float64).alias("split_factor"),

        # Trailing 12-month dividend yield (placeholder — compute in gold layer)
        pl.lit(0.0, dtype=pl.Float64).alias("dividend_yield"),

        # Market date flag
        pl.lit(True).alias("is_market_date"),

        # Data quality flags
        pl.when(pl.col("volume") == 0)
        .then(pl.lit("thin"))
        .otherwise(pl.lit("good"))
        .alias("data_quality"),
    ])

    # Flag suspicious price gaps (>20% day-over-day)
    adjusted = (
        adjusted.sort(["ticker", "date"])
        .with_columns(
            pl.col("adj_close")
            .shift(1)
            .over("ticker")
            .alias("prev_adj_close")
        )
        .with_columns(
            pl.when(
                pl.col("prev_adj_close").is_not_null()
                & (((pl.col("adj_close") / pl.col("prev_adj_close")) - 1).abs() > 0.20)
            )
            .then(pl.lit("suspicious"))
            .otherwise(pl.col("data_quality"))
            .alias("data_quality")
        )
        .drop("prev_adj_close")
    )

    logger.info("Silver layer built: %d rows", len(adjusted))
    return adjusted
