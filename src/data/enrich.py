"""Gold layer: enrich silver data with cross-sectional features."""

import logging

import polars as pl

logger = logging.getLogger(__name__)

# Simplified sector mapping for prototyping
# In production, this would come from a reference data source
SECTOR_MAP = {
    # Tech
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "GOOG": "Technology", "META": "Technology", "NVDA": "Technology",
    "AMD": "Technology", "INTC": "Technology", "CRM": "Technology",
    "ADBE": "Technology", "NFLX": "Communication Services",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "C": "Financials",
    "BRK-B": "Financials", "V": "Financials", "MA": "Financials",
    # Healthcare
    "JNJ": "Health Care", "UNH": "Health Care", "PFE": "Health Care",
    "ABBV": "Health Care", "MRK": "Health Care", "LLY": "Health Care",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    # Consumer
    "PG": "Consumer Defensive", "KO": "Consumer Defensive",
    "PEP": "Consumer Defensive", "WMT": "Consumer Defensive",
    "COST": "Consumer Defensive", "NKE": "Consumer Cyclical",
    # Industrials
    "CAT": "Industrials", "HON": "Industrials", "UNP": "Industrials",
    "BA": "Industrials", "GE": "Industrials",
    # Other
    "DIS": "Communication Services", "TSLA": "Consumer Cyclical",
    "AMZN": "Consumer Cyclical",
    # ETFs — assign to their primary sector
    "SPY": "Broad Market", "QQQ": "Broad Market", "IWM": "Broad Market",
    "DIA": "Broad Market",
    "XLF": "Financials", "XLE": "Energy", "XLK": "Technology",
    "XLV": "Health Care", "XLI": "Industrials", "XLP": "Consumer Defensive",
    "XLU": "Utilities", "XLB": "Materials", "XLRE": "Real Estate",
    "XLY": "Consumer Cyclical",
}


def enrich_with_sector(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Add sector classification."""
    sector_df = pl.DataFrame({
        "ticker": list(SECTOR_MAP.keys()),
        "sector": list(SECTOR_MAP.values()),
    })

    return df.join(sector_df, on="ticker", how="left").with_columns(
        pl.col("sector").fill_null("Unknown")
    )


def enrich_with_volume_features(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Add rolling volume features."""
    return (
        df.sort(["ticker", "date"])
        .with_columns([
            pl.col("volume")
            .rolling_mean(window_size=20)
            .over("ticker")
            .alias("avg_volume_20d"),
            pl.col("volume")
            .rolling_mean(window_size=60)
            .over("ticker")
            .alias("avg_volume_60d"),
        ])
        .with_columns(
            pl.when(pl.col("avg_volume_60d") > 0)
            .then(pl.col("volume") / pl.col("avg_volume_60d"))
            .otherwise(pl.lit(1.0))
            .alias("volume_ratio")
        )
    )


def enrich_with_market_cap_bucket(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Add market cap bucket.

    Note: This is a simplified placeholder. In production, use actual
    market cap data from the provider. For now, use sector-based heuristics.

    Uses vectorized when/otherwise + is_in — no row-wise Python iteration.
    """
    mega_caps = ["AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "BRK.B", "JPM"]
    large_caps = ["TSLA", "UNH", "V", "MA", "JNJ", "XOM", "CVX", "PG", "HD", "DIS", "NFLX"]
    etfs = ["SPY", "QQQ", "IWM", "DIA"]

    return df.with_columns(
        pl.when(pl.col("ticker").is_in(mega_caps))
        .then(pl.lit("mega"))
        .when(pl.col("ticker").is_in(large_caps))
        .then(pl.lit("large"))
        .when(pl.col("ticker").is_in(etfs) | pl.col("ticker").str.starts_with("X"))
        .then(pl.lit("etf"))
        .otherwise(pl.lit("mid"))
        .alias("market_cap_bucket")
    )


def enrich_with_universe_flag(
    df: pl.DataFrame,
    universe_date: pl.Date = None,
) -> pl.DataFrame:
    """Flag whether each ticker is in the investable universe on that date.

    For prototyping, assume all tickers are always in the universe.
    """
    return df.with_columns(pl.lit(True).alias("universe_date"))


def build_gold_layer(
    silver_bars: pl.DataFrame,
) -> pl.DataFrame:
    """Build the gold layer from silver data.

    Pipeline:
    1. Start with silver (adjusted) bars
    2. Add sector classification
    3. Add volume features
    4. Add market cap bucket
    5. Add universe flag

    Args:
        silver_bars: Silver layer DataFrame

    Returns:
        Gold layer DataFrame ready for signal generation.
    """
    logger.info("Building gold layer: %d rows", len(silver_bars))

    gold = silver_bars.clone()

    # Step 1: Sector
    gold = enrich_with_sector(gold)

    # Step 2: Volume features
    gold = enrich_with_volume_features(gold)

    # Step 3: Market cap bucket
    gold = enrich_with_market_cap_bucket(gold)

    # Step 4: Universe flag
    gold = enrich_with_universe_flag(gold)

    # Final sort
    gold = gold.sort(["ticker", "date"])

    logger.info("Gold layer built: %d rows, %d tickers",
                len(gold), gold["ticker"].n_unique())
    return gold
