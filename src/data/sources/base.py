"""Abstract base class for data sources."""

from abc import ABC, abstractmethod
from datetime import date
from typing import Optional

import polars as pl


class DataSource(ABC):
    """Abstract interface for market data sources.

    All data sources (Polygon.io, yfinance, IB) must implement this interface
    to ensure consistent ingestion behavior.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Source identifier (e.g., 'polygon', 'yfinance', 'ib')."""

    @abstractmethod
    async def fetch_daily_bars(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> pl.DataFrame:
        """Fetch daily OHLCV bars for a ticker.

        Returns a DataFrame with columns:
            ticker, date, open, high, low, close, volume, vwap

        All prices are raw (unadjusted). Adjustments happen in the silver layer.
        """

    @abstractmethod
    async def fetch_dividends(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> pl.DataFrame:
        """Fetch dividend history for a ticker.

        Returns a DataFrame with columns:
            ticker, ex_date, amount
        """

    @abstractmethod
    async def fetch_corporate_actions(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> Optional[pl.DataFrame]:
        """Fetch corporate actions (splits, mergers) for a ticker.

        Returns a DataFrame with columns:
            ticker, date, action_type, factor

        May return None if the source doesn't support corporate actions.
        """

    @abstractmethod
    async def fetch_universe(self) -> list[str]:
        """Fetch the investable universe of tickers.

        Returns a list of ticker symbols.
        """

    def validate_bars(self, df: pl.DataFrame) -> pl.DataFrame:
        """Basic validation of bar data.

        Default: remove negative prices, future dates, add source column.
        """
        if df.is_empty():
            return df

        today = date.today()
        validated = df.filter(
            (pl.col("close") > 0)
            & (pl.col("date") <= today)
        )

        # Add source column
        if "source" not in validated.columns:
            validated = validated.with_columns(
                pl.lit(self.name).alias("source")
            )

        # Add vwap if missing
        if "vwap" not in validated.columns:
            validated = validated.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("vwap")
            )

        return validated
