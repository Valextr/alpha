"""Alpaca intraday data ingestion.

Downloads 1-minute bars from Alpaca and saves to Parquet in the project's
data directory, following the same layout as the daily data layers
(data/intraday/{resolution}/ticker={TICKER}/year={YYYY}/part-0.parquet).

Usage:
    from src.data.alpaca import download_intraday
    download_intraday(tickers=['AAPL'], start='2025-01-01', end='2025-12-31')
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

load_dotenv()

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

log = logging.getLogger(__name__)


def _get_client() -> StockHistoricalDataClient:
    import os
    return StockHistoricalDataClient(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_API_SECRET"],
    )


def download_intraday(
    tickers: list[str],
    start: str | date,
    end: str | date,
    resolution: str = "1min",
    data_dir: Path | None = None,
) -> list[Path]:
    """Download 1-minute bars for the given tickers and date range.

    Args:
        tickers: List of ticker symbols.
        start: Start date (ISO string or date).
        end: End date (ISO string or date).
        resolution: Bar resolution (currently only "1min" is supported).
        data_dir: Override data directory (defaults to project root / data).

    Returns:
        List of parquet files written.
    """
    if isinstance(start, str):
        start = date.fromisoformat(start)
    if isinstance(end, str):
        end = date.fromisoformat(end)

    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / "data"

    out_root = data_dir / "intraday" / resolution
    client = _get_client()
    written: list[Path] = []

    # Alpaca free tier has rate limits; download one ticker at a time
    for ticker in tickers:
        log.info("Downloading %s (%s to %s)", ticker, start, end)

        # Download in monthly chunks to stay within API limits
        cursor = start
        while cursor <= end:
            # Calculate end of current month
            if cursor.month == 12:
                month_end = cursor.replace(year=cursor.year + 1, month=1, day=1) - timedelta(days=1)
            else:
                month_end = cursor.replace(month=cursor.month + 1, day=1) - timedelta(days=1)
            
            # Don't exceed the requested end date
            if month_end > end:
                month_end = end

            req = StockBarsRequest(
                symbol_or_symbols=[ticker],
                timeframe=TimeFrame.Minute,
                start=datetime(cursor.year, cursor.month, cursor.day),
                end=datetime(month_end.year, month_end.month, month_end.day, 23, 59, 59),
            )
            bars = client.get_stock_bars(req)
            df = bars.df

            if df.empty:
                log.warning("No bars for %s %s-%s", ticker, cursor, month_end)
                cursor = month_end + timedelta(days=1)
                continue

            # Flatten multi-index
            df = df.reset_index()
            df = df.rename(columns={"timestamp": "datetime"})

            # Convert to polars
            df = pl.from_pandas(df)

            # Extract date for partitioning
            df = df.with_columns(
                pl.col("datetime").dt.date().alias("date"),
                pl.lit(ticker).alias("ticker"),
            )

            # Write per-month partitions (avoid overwriting)
            month = df["date"].dt.month()
            for m in month.unique():
                y = df["date"].dt.year().unique()[0]  # All data is from same year
                mdf = df.filter(pl.col("date").dt.month() == m)
                out = out_root / f"ticker={ticker}" / f"year={y}" / f"month={m:02d}"
                out.mkdir(parents=True, exist_ok=True)
                path = out / "part-0.parquet"
                mdf.write_parquet(str(path))
                written.append(path)
                log.info("Wrote %s (%d bars)", path, len(mdf))

            cursor = month_end + timedelta(days=1)

    log.info("Done. Wrote %d files for %d tickers.", len(written), len(tickers))
    return written