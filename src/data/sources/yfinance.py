"""yfinance data source — free, good for prototyping.

WARNING: yfinance has survivorship bias. Dead tickers disappear.
Only use for pipeline development, never for final backtests.
"""

import asyncio
import logging
from datetime import date
from typing import Optional

import polars as pl
import yfinance as yf

from .base import DataSource

logger = logging.getLogger(__name__)


class YFinanceSource(DataSource):
    """yfinance data source for prototyping."""

    @property
    def name(self) -> str:
        return "yfinance"

    async def _run_in_executor(self, func, *args, **kwargs):
        """Run blocking yfinance call in a thread executor."""
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: func(*args, **kwargs)
        )

    async def fetch_daily_bars(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> pl.DataFrame:
        """Fetch daily bars via yfinance."""
        logger.debug("Fetching %s daily bars [%s, %s]", ticker, start, end)

        empty = pl.DataFrame(
            schema={
                "ticker": pl.Utf8,
                "date": pl.Date,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
                "vwap": pl.Float64,
            }
        )

        try:
            stock = yf.Ticker(ticker)
            hist = await self._run_in_executor(
                stock.history,
                start=start.isoformat(),
                end=end.isoformat(),
                auto_adjust=False,
            )
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", ticker, e)
            return empty

        if hist.empty:
            logger.warning("No data for %s", ticker)
            return empty

        # Convert to Polars
        df = pl.from_pandas(hist.reset_index())

        # Find and rename date column from index
        date_col = None
        for candidate in ["Date", "Dates", "timestamp", "Datetime"]:
            if candidate in df.columns:
                date_col = candidate
                break

        if date_col:
            df = df.rename({date_col: "date"})

        # Standardize column names
        rename_map = {}
        for src, dst in [
            ("Open", "open"), ("High", "high"), ("Low", "low"),
            ("Close", "close"), ("Volume", "volume"),
        ]:
            if src in df.columns:
                rename_map[src] = dst
        if rename_map:
            df = df.rename(rename_map)

        # Ensure ticker column
        if "ticker" not in df.columns:
            df = df.with_columns(pl.lit(ticker).alias("ticker"))

        # Ensure date is Date type
        if "date" in df.columns and df["date"].dtype != pl.Date:
            df = df.with_columns(
                pl.col("date").cast(pl.Datetime).dt.date().alias("date")
            )

        # Select only the columns we want
        cols = ["ticker", "date", "open", "high", "low", "close", "volume"]
        existing = [c for c in cols if c in df.columns]
        df = df.select(existing)

        # Add vwap (yfinance doesn't provide it)
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("vwap"))

        return self.validate_bars(df)

    async def fetch_dividends(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> pl.DataFrame:
        """Fetch dividends via yfinance.

        Note: yfinance.dividends is a property, not a method.
        We fetch all and filter by date range.
        """
        logger.debug("Fetching %s dividends [%s, %s]", ticker, start, end)

        empty = pl.DataFrame(
            schema={"ticker": pl.Utf8, "ex_date": pl.Date, "amount": pl.Float64}
        )

        try:
            stock = yf.Ticker(ticker)
            # dividends is a property that returns a Series
            div_series = await self._run_in_executor(lambda: stock.dividends)
        except Exception as e:
            logger.warning("Failed to fetch dividends for %s: %s", ticker, e)
            return empty

        if div_series is None or div_series.empty:
            return empty

        df = pl.from_pandas(div_series.reset_index())

        # Find date column
        date_col = None
        for candidate in ["Date", "Dates", "timestamp", "index"]:
            if candidate in df.columns:
                date_col = candidate
                break

        if date_col:
            df = df.rename({date_col: "ex_date"})

        # Find amount column
        for candidate in ["Dividends", "amount", ticker]:
            if candidate in df.columns and candidate != "amount":
                df = df.rename({candidate: "amount"})
                break

        # Filter by date range
        if "ex_date" in df.columns:
            df = df.filter(
                (pl.col("ex_date") >= start) & (pl.col("ex_date") <= end)
            )

        df = df.with_columns(pl.lit(ticker).alias("ticker"))

        if "ex_date" in df.columns and df["ex_date"].dtype != pl.Date:
            df = df.with_columns(
                pl.col("ex_date").cast(pl.Datetime).dt.date().alias("ex_date")
            )

        if "ex_date" in df.columns and "amount" in df.columns:
            return df.select(["ticker", "ex_date", "amount"])
        return empty

    async def fetch_corporate_actions(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> Optional[pl.DataFrame]:
        """Fetch splits via yfinance.

        Note: yfinance.splits is a property, not a method.
        """
        logger.debug("Fetching %s corporate actions [%s, %s]", ticker, start, end)

        try:
            stock = yf.Ticker(ticker)
            splits_series = await self._run_in_executor(lambda: stock.splits)
        except Exception as e:
            logger.warning("Failed to fetch splits for %s: %s", ticker, e)
            return None

        if splits_series is None or splits_series.empty:
            return None

        df = pl.from_pandas(splits_series.reset_index())

        # Find date column
        date_col = None
        for candidate in ["Date", "Dates", "timestamp", "index"]:
            if candidate in df.columns:
                date_col = candidate
                break

        if date_col:
            df = df.rename({date_col: "date"})

        # Find factor column
        for candidate in ["Splits", "factor", ticker]:
            if candidate in df.columns and candidate != "factor":
                df = df.rename({candidate: "factor"})
                break

        # Filter by date range
        if "date" in df.columns:
            df = df.filter(
                (pl.col("date") >= start) & (pl.col("date") <= end)
            )

        df = df.with_columns(
            pl.lit(ticker).alias("ticker"),
            pl.lit("split").alias("action_type"),
        )

        if "date" in df.columns and df["date"].dtype != pl.Date:
            df = df.with_columns(
                pl.col("date").cast(pl.Datetime).dt.date().alias("date")
            )

        if "date" in df.columns and "factor" in df.columns:
            return df.select(["ticker", "date", "action_type", "factor"])
        return None

    async def fetch_universe(self) -> list[str]:
        """yfinance doesn't provide a universe endpoint."""
        logger.info("yfinance has no universe endpoint; use config default_universe")
        return []
