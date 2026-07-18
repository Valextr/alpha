"""Main ingestion pipeline orchestrator.

Orchestrates the full pipeline:
1. Fetch bronze data (raw) from sources
2. Build silver layer (adjusted)
3. Build gold layer (enriched)
4. Create DuckDB catalog
"""

import argparse
import asyncio
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import polars as pl
from tqdm import tqdm

from .config import DataConfig, get_config
from .sources.yfinance import YFinanceSource
from .adjust import build_silver_layer
from .enrich import build_gold_layer
from .catalog import create_catalog
from .validate import run_all_checks

# Staleness threshold — data older than this is re-fetched
STALENESS_THRESHOLD_DAYS = 1

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_dirs(config: DataConfig) -> None:
    """Ensure data directories exist."""
    dirs = [
        config.data_dir / "bronze" / "daily",
        config.data_dir / "bronze" / "dividends",
        config.data_dir / "bronze" / "corporate_actions",
        config.data_dir / "silver" / "daily",
        config.data_dir / "gold" / "daily",
        config.data_dir / "_catalog",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def save_parquet(
    df: pl.DataFrame,
    output_dir: Path,
    ticker: str = None,
    year: str = None,
) -> None:
    """Save DataFrame as Parquet file, partitioned by year and ticker.

    Args:
        df: DataFrame to save
        output_dir: Output directory
        ticker: Ticker symbol for partitioning
        year: Year for partitioning
    """
    if df.is_empty():
        return

    # Create partition directories
    if year:
        output_dir = output_dir / f"year={year}"
    if ticker:
        output_dir = output_dir / f"ticker={ticker}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing data (idempotent — only write if newer)
    out_file = output_dir / "part-0.parquet"

    if out_file.exists():
        existing = pl.read_parquet(str(out_file))
        if existing.height >= df.height:
            return  # Skip — already have this data

    df.write_parquet(
        str(out_file),
        compression="zstd",
    )


def save_parquet_partitioned_by_year(
    df: pl.DataFrame,
    output_dir: Path,
    ticker: str = None,
) -> None:
    """Save DataFrame as Parquet, partitioning rows into year=YYYY folders.

    Unlike save_parquet which takes a single year argument, this function
    groups the DataFrame by the actual year of each row's date column so
    data spanning multiple years lands in the correct partitions.

    Args:
        df: DataFrame with a 'date' column
        output_dir: Base output directory
        ticker: Ticker symbol for partitioning
    """
    if df.is_empty() or "date" not in df.columns:
        return

    df = df.with_columns(
        pl.col("date").dt.year().cast(pl.Utf8).alias("_year")
    )
    for year, year_df in df.group_by("_year", maintain_order=True):
        year_str = str(year[0])
        year_df = year_df.drop("_year")
        save_parquet(year_df, output_dir, ticker=ticker, year=year_str)


async def fetch_bronze_for_ticker(
    source,
    ticker: str,
    start: date,
    end: date,
    config: DataConfig,
) -> dict:
    """Fetch all bronze data for a single ticker."""
    result = {
        "bars": await source.fetch_daily_bars(ticker, start, end),
        "dividends": await source.fetch_dividends(ticker, start, end),
        "actions": await source.fetch_corporate_actions(ticker, start, end),
    }

    # Save bronze data — partitioned by year and ticker
    if not result["bars"].is_empty():
        save_parquet_partitioned_by_year(
            result["bars"],
            config.data_dir / "bronze" / "daily",
            ticker=ticker,
        )

    if not result["dividends"].is_empty():
        save_parquet(
            result["dividends"],
            config.data_dir / "bronze" / "dividends",
            ticker=ticker,
        )

    if result["actions"] and not result["actions"].is_empty():
        save_parquet(
            result["actions"],
            config.data_dir / "bronze" / "corporate_actions",
            ticker=ticker,
        )

    return result


async def run_pipeline(
    config: DataConfig,
    tickers: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    update_only: bool = False,
    verbose: bool = False,
) -> None:
    """Run the full ingestion pipeline."""
    setup_logging(verbose)

    start_date = date.fromisoformat(start) if start else date.fromisoformat(config.default_start_date)
    end_date = date.fromisoformat(end) if end else (date.today() - timedelta(days=1))

    if tickers is None:
        tickers = config.default_universe

    logger.info(
        "Pipeline start: %d tickers, %s to %s",
        len(tickers), start_date, end_date,
    )

    ensure_dirs(config)

    # Initialize source
    source = YFinanceSource()

    # Step 1: Fetch bronze data
    logger.info("Step 1: Fetching bronze data...")
    all_bars = []
    all_dividends = []
    all_actions = []

    for ticker in tqdm(tickers, desc="Fetching tickers"):
        try:
            result = await fetch_bronze_for_ticker(
                source, ticker, start_date, end_date, config
            )
            all_bars.append(result["bars"])
            all_dividends.append(result["dividends"])
            if result["actions"]:
                all_actions.append(result["actions"])
        except Exception as e:
            logger.error("Failed to fetch %s: %s", ticker, e)

    # Combine — ensure consistent schemas
    valid_bars = [df for df in all_bars if not df.is_empty()]
    if valid_bars:
        # Union all columns, then fill missing with null
        all_cols = set()
        for df in valid_bars:
            all_cols.update(df.columns)
        all_cols = sorted(all_cols)
        normalized = []
        for df in valid_bars:
            for col in all_cols:
                if col not in df.columns:
                    df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias(col))
            df = df.select(all_cols)
            normalized.append(df)
        bronze_bars = pl.concat(normalized, how="vertical_relaxed")
    else:
        bronze_bars = pl.DataFrame()

    valid_divs = [df for df in all_dividends if not df.is_empty()]
    bronze_divs = pl.concat(valid_divs, how="vertical_relaxed") if valid_divs else pl.DataFrame()

    valid_actions = [df for df in all_actions if df and not df.is_empty()]
    bronze_actions = pl.concat(valid_actions, how="vertical_relaxed") if valid_actions else pl.DataFrame()

    logger.info(
        "Bronze: %d bars, %d dividends, %d actions",
        len(bronze_bars), len(bronze_divs), len(bronze_actions),
    )

    if bronze_bars.is_empty():
        logger.error("No data fetched — aborting pipeline")
        return

    # Step 2: Build silver layer
    logger.info("Step 2: Building silver layer...")
    silver = build_silver_layer(bronze_bars, bronze_divs, bronze_actions)

    # Save silver — partitioned by year and ticker
    for ticker in silver["ticker"].unique():
        ticker_data = silver.filter(pl.col("ticker") == ticker)
        save_parquet_partitioned_by_year(
            ticker_data,
            config.data_dir / "silver" / "daily",
            ticker=ticker,
        )

    # Step 3: Build gold layer
    logger.info("Step 3: Building gold layer...")
    gold = build_gold_layer(silver)

    # Save gold — partitioned by year and ticker
    for ticker in gold["ticker"].unique():
        ticker_data = gold.filter(pl.col("ticker") == ticker)
        save_parquet_partitioned_by_year(
            ticker_data,
            config.data_dir / "gold" / "daily",
            ticker=ticker,
        )

    # Step 4: Create DuckDB catalog
    logger.info("Step 4: Creating DuckDB catalog...")
    create_catalog(config.data_dir, config.duckdb_file)

    # Step 5: Validate
    logger.info("Step 5: Running validation...")
    result = run_all_checks(gold, "gold")
    print("\n" + result.summary())

    logger.info("Pipeline complete!")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Alpha data ingestion pipeline")
    sub = parser.add_subparsers(dest="command")

    # Fetch command
    fetch_parser = sub.add_parser("fetch", help="Fetch data")
    fetch_parser.add_argument("--all", action="store_true", help="Fetch all configured tickers")
    fetch_parser.add_argument("--ticker", type=str, help="Fetch specific ticker")
    fetch_parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    fetch_parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    fetch_parser.add_argument("-v", "--verbose", action="store_true")

    # Update command
    sub.add_parser("update", help="Update with latest data")

    # Full pipeline
    full_parser = sub.add_parser("run", help="Run full pipeline")
    full_parser.add_argument("--tickers", nargs="+", help="Tickers to fetch")
    full_parser.add_argument("--start", type=str, help="Start date")
    full_parser.add_argument("--end", type=str, help="End date")
    full_parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    config = get_config()

    if args.command == "fetch":
        tickers = config.default_universe if args.all else ([args.ticker] if args.ticker else None)
        if not tickers:
            parser.print_help()
            return
        asyncio.run(run_pipeline(config, tickers, args.start, args.end, verbose=args.verbose))
    elif args.command == "update":
        asyncio.run(run_pipeline(config, update_only=True))
    elif args.command == "run":
        tickers = args.tickers or config.default_universe
        asyncio.run(run_pipeline(config, tickers, args.start, args.end, verbose=args.verbose))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
