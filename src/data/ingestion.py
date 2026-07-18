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


def find_ticker_max_date(data_dir: Path, layer: str, ticker: str) -> Optional[date]:
    """Find the maximum date in parquet files for a ticker in a given layer.

    Scans all year partitions under data_dir/{layer}/daily/ticker={T}/year={YYYY}/.

    Returns None if no data exists for this ticker in this layer.
    """
    base = data_dir / layer / "daily" / f"ticker={ticker}"
    if not base.exists():
        return None

    max_date = None
    for year_dir in sorted(base.glob("year=*")):
        parquet_file = year_dir / "part-0.parquet"
        if parquet_file.exists():
            try:
                df = pl.read_parquet(str(parquet_file), columns=["date"])
                if not df.is_empty():
                    d = df["date"].max().to_python()
                    if d is not None:
                        if isinstance(d, datetime):
                            d = d.date()
                        if max_date is None or d > max_date:
                            max_date = d
            except Exception:
                # Corrupt or unreadable file — treat as missing
                continue

    return max_date


def should_fetch_ticker(
    config: DataConfig,
    ticker: str,
    start_date: date,
    end_date: date,
    force: bool = False,
) -> tuple[bool, date]:
    """Determine whether a ticker needs fetching and compute the effective start date.

    Returns (should_fetch, effective_start_date).

    Logic:
    - If no bronze data exists: fetch full range
    - If data exists but is stale (max_date < end_date - threshold): incremental from existing
    - If data is fresh (max_date >= end_date - threshold): skip
    - force=True bypasses staleness checks
    """
    existing_max = find_ticker_max_date(config.data_dir, "bronze", ticker)

    if existing_max is None:
        # No data — fetch full range
        return True, start_date

    if force:
        # Force refresh — fetch full range
        return True, start_date

    # Check staleness: is the latest data point within threshold of end_date?
    cutoff = end_date - timedelta(days=STALENESS_THRESHOLD_DAYS)
    if existing_max >= cutoff:
        # Data is fresh — nothing to fetch
        return False, end_date

    # Data exists but is stale/incomplete — fetch incremental
    effective_start = existing_max + timedelta(days=1)
    return True, effective_start


def load_bronze_from_disk(
    config: DataConfig,
    tickers: list[str],
    start: date,
    end: date,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load existing bronze data from parquet on disk.

    Reads all year/ticker partitions for bronze/daily, bronze/dividends,
    bronze/corporate_actions, filtering to the requested tickers and date range.

    Returns (bars, dividends, actions) — may be empty if no data exists.
    """
    bars_list = []
    divs_list = []
    actions_list = []

    for ticker in tickers:
        # Load daily bars
        bar_base = config.data_dir / "bronze" / "daily" / f"ticker={ticker}"
        if bar_base.exists():
            for year_dir in sorted(bar_base.glob("year=*")):
                parquet_file = year_dir / "part-0.parquet"
                if parquet_file.exists():
                    try:
                        df = pl.read_parquet(str(parquet_file))
                        if not df.is_empty():
                            df = df.filter(
                                (pl.col("date") >= start) & (pl.col("date") <= end)
                            )
                            if not df.is_empty():
                                bars_list.append(df)
                    except Exception:
                        continue

        # Load dividends
        div_base = config.data_dir / "bronze" / "dividends" / f"ticker={ticker}"
        if div_base.exists():
            for parquet_file in sorted(div_base.glob("*.parquet")):
                try:
                    df = pl.read_parquet(str(parquet_file))
                    if not df.is_empty():
                        df = df.filter(
                            (pl.col("ex_date") >= start) & (pl.col("ex_date") <= end)
                        )
                        if not df.is_empty():
                            divs_list.append(df)
                except Exception:
                    continue

        # Load corporate actions
        act_base = config.data_dir / "bronze" / "corporate_actions" / f"ticker={ticker}"
        if act_base.exists():
            for parquet_file in sorted(act_base.glob("*.parquet")):
                try:
                    df = pl.read_parquet(str(parquet_file))
                    if not df.is_empty():
                        df = df.filter(
                            (pl.col("date") >= start) & (pl.col("date") <= end)
                        )
                        if not df.is_empty():
                            actions_list.append(df)
                except Exception:
                    continue

    # Combine with consistent schemas
    bars = (pl.concat(bars_list, how="vertical_relaxed")
            if bars_list else pl.DataFrame())
    divs = (pl.concat(divs_list, how="vertical_relaxed")
            if divs_list else pl.DataFrame())
    actions = (pl.concat(actions_list, how="vertical_relaxed")
               if actions_list else pl.DataFrame())

    return bars, divs, actions


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
    """Run the full ingestion pipeline.

    Supports resumption: re-running with the same tickers skips already-ingested
    data and only fetches missing or stale data (older than STALENESS_THRESHOLD_DAYS).
    Partial failures don't require full re-fetch — only the failed tickers retry.
    """
    setup_logging(verbose)

    start_date = date.fromisoformat(start) if start else date.fromisoformat(config.default_start_date)
    end_date = date.fromisoformat(end) if end else (date.today() - timedelta(days=1))

    if tickers is None:
        tickers = config.default_universe

    logger.info(
        "Pipeline start: %d tickers, %s to %s (update_only=%s)",
        len(tickers), start_date, end_date, update_only,
    )

    ensure_dirs(config)

    # Initialize source
    source = YFinanceSource()

    # ── Step 1a: Determine which tickers need fetching ──────────────────
    logger.info("Step 1: Checking existing bronze data...")
    fetch_plan = {}  # ticker -> (effective_start, reason)
    skipped = 0
    to_fetch = []

    for ticker in tickers:
        force = not update_only  # first-run or explicit --tickers means fetch
        needs_fetch, effective_start = should_fetch_ticker(
            config, ticker, start_date, end_date, force=force
        )
        if needs_fetch:
            fetch_plan[ticker] = (effective_start, "incremental" if effective_start > start_date else "full")
            to_fetch.append(ticker)
        else:
            skipped += 1
            fetch_plan[ticker] = (None, "fresh")

    logger.info(
        "Resumption: %d to fetch, %d already fresh (skipped)",
        len(to_fetch), skipped,
    )

    # ── Step 1b: Fetch bronze data for tickers that need it ─────────────
    if to_fetch:
        logger.info("Fetching bronze data for %d tickers...", len(to_fetch))
    all_bars = []
    all_dividends = []
    all_actions = []

    for ticker in tqdm(to_fetch, desc="Fetching tickers"):
        eff_start, reason = fetch_plan[ticker]
        try:
            result = await fetch_bronze_for_ticker(
                source, ticker, eff_start, end_date, config
            )
            all_bars.append(result["bars"])
            all_dividends.append(result["dividends"])
            if result["actions"]:
                all_actions.append(result["actions"])
        except Exception as e:
            logger.error("Failed to fetch %s: %s", ticker, e)

    # ── Step 1c: Load existing bronze + merge with newly fetched ────────
    logger.info("Loading existing bronze data...")
    existing_bars, existing_divs, existing_actions = load_bronze_from_disk(
        config, tickers, start_date, end_date
    )

    # Append newly fetched bars/divs/actions to existing
    new_bars_list = [df for df in all_bars if not df.is_empty()]
    if existing_bars.is_empty():
        final_bars = new_bars_list
    else:
        final_bars = [existing_bars] + new_bars_list

    new_divs_list = [df for df in all_dividends if not df.is_empty()]
    if existing_divs.is_empty():
        final_divs = new_divs_list
    else:
        final_divs = [existing_divs] + new_divs_list

    new_actions_list = [df for df in all_actions if df and not df.is_empty()]
    if existing_actions.is_empty():
        final_actions = new_actions_list
    else:
        final_actions = [existing_actions] + new_actions_list

    # Combine — ensure consistent schemas
    valid_bars = [df for df in final_bars if not df.is_empty()]
    if valid_bars:
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

    valid_divs = [df for df in final_divs if not df.is_empty()]
    bronze_divs = pl.concat(valid_divs, how="vertical_relaxed") if valid_divs else pl.DataFrame()

    valid_actions = [df for df in final_actions if df and not df.is_empty()]
    bronze_actions = pl.concat(valid_actions, how="vertical_relaxed") if valid_actions else pl.DataFrame()

    logger.info(
        "Bronze: %d bars, %d dividends, %d actions (from disk + network)",
        len(bronze_bars), len(bronze_divs), len(bronze_actions),
    )

    if bronze_bars.is_empty():
        logger.error("No data available — aborting pipeline")
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
    fetch_parser.add_argument("--force", action="store_true", help="Force re-fetch even if data is fresh")
    fetch_parser.add_argument("-v", "--verbose", action="store_true")

    # Update command — incremental only (no full re-fetch)
    update_parser = sub.add_parser("update", help="Update with latest data")
    update_parser.add_argument("--tickers", nargs="+", help="Tickers to update (default: all)")
    update_parser.add_argument("--end", type=str, help="End date")
    update_parser.add_argument("--force", action="store_true", help="Force re-fetch even if data is fresh")
    update_parser.add_argument("-v", "--verbose", action="store_true")

    # Full pipeline
    full_parser = sub.add_parser("run", help="Run full pipeline")
    full_parser.add_argument("--tickers", nargs="+", help="Tickers to fetch")
    full_parser.add_argument("--start", type=str, help="Start date")
    full_parser.add_argument("--end", type=str, help="End date")
    full_parser.add_argument("--update-only", action="store_true",
                             help="Skip tickers with fresh data (incremental)")
    full_parser.add_argument("--force", action="store_true",
                             help="Force re-fetch even if data is fresh")
    full_parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    config = get_config()

    if args.command == "fetch":
        tickers = config.default_universe if args.all else ([args.ticker] if args.ticker else None)
        if not tickers:
            parser.print_help()
            return
        # fetch: fetches everything (update_only=False = force=True in should_fetch)
        asyncio.run(run_pipeline(config, tickers, args.start, args.end,
                                 update_only=False, verbose=args.verbose))
    elif args.command == "update":
        tickers = getattr(args, "tickers", None)
        force = getattr(args, "force", False)
        # update: incremental by default (update_only=True), --force overrides
        asyncio.run(run_pipeline(config, tickers, end=args.end,
                                 update_only=not force, verbose=args.verbose))
    elif args.command == "run":
        tickers = args.tickers or config.default_universe
        update_only = getattr(args, "update_only", False)
        force = getattr(args, "force", False)
        if force:
            update_only = False
        asyncio.run(run_pipeline(config, tickers, args.start, args.end,
                                 update_only=update_only, verbose=args.verbose))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
