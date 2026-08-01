"""DuckDB catalog management.

Creates views that abstract Parquet files, so downstream code
queries DuckDB views instead of touching Parquet directly.
"""

import logging
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)


def _glob_pattern(data_dir: Path, layer: str, subdir: str = "daily") -> str:
    """Build a glob pattern for Parquet files."""
    return str(data_dir / layer / subdir / "**" / "*.parquet")


def _has_files(data_dir: Path, layer: str, subdir: str = "daily") -> bool:
    """Check if a directory has any Parquet files."""
    dir_path = data_dir / layer / subdir
    if not dir_path.exists():
        return False
    # Check direct files and recursive
    return any(dir_path.glob("*.parquet")) or any(dir_path.rglob("*.parquet"))


def create_catalog(
    data_dir: Path,
    duckdb_file: Path,
) -> None:
    """Create or rebuild the DuckDB catalog.

    Args:
        data_dir: Root data directory
        duckdb_file: Path to DuckDB database file
    """
    logger.info("Creating DuckDB catalog at %s", duckdb_file)

    duckdb_file.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(duckdb_file))

    # Helper: create view only if files exist
    def create_view(name: str, layer: str, subdir: str = "daily", columns: str = "*"):
        pattern = _glob_pattern(data_dir, layer, subdir)
        if _has_files(data_dir, layer, subdir):
            conn.execute(f"""
                CREATE OR REPLACE VIEW {name} AS
                SELECT {columns} FROM read_parquet('{pattern}')
            """)
            logger.debug("Created view %s from %s", name, pattern)
        else:
            logger.warning("No files for %s (%s) — skipping view", name, pattern)

    # Bronze views
    create_view("bronze_daily", "bronze", "daily")
    create_view("bronze_dividends", "bronze", "dividends")
    create_view("bronze_corporate_actions", "bronze", "corporate_actions")

    # Silver views
    create_view("silver_daily", "silver", "daily")

    # Gold views
    create_view("gold_daily", "gold", "daily")

    # Convenience: universe view (only investable dates with good data)
    # Only create if gold_daily exists
    try:
        conn.execute("SELECT 1 FROM gold_daily LIMIT 0")
        conn.execute("""
            CREATE OR REPLACE VIEW universe_daily AS
            SELECT * FROM gold_daily
            WHERE universe_date = true
              AND data_quality = 'good'
        """)
    except duckdb.CatalogException:
        logger.info("gold_daily view not available — skipping universe_daily")

    conn.close()
    logger.info("DuckDB catalog created successfully")


def query_catalog(
    duckdb_file: Path,
    query: str,
    params: tuple = (),
) -> "polars.DataFrame":
    """Query the DuckDB catalog."""
    import polars as pl

    conn = duckdb.connect(str(duckdb_file))
    result = conn.execute(query, params).fetchdf()
    conn.close()

    return pl.from_pandas(result)


def catalog_stats(
    duckdb_file: Path,
) -> dict:
    """Get catalog statistics."""
    conn = duckdb.connect(str(duckdb_file))

    stats = {}
    for view in ["bronze_daily", "silver_daily", "gold_daily", "universe_daily"]:
        try:
            result = conn.execute(f"SELECT COUNT(*) FROM {view}").fetchone()
            stats[view] = result[0] if result else 0
        except Exception as e:
            stats[view] = f"Not available: {e}"

    conn.close()
    return stats


def main():
    """CLI entry point for catalog management."""
    import argparse
    import sys

    from .config import get_config

    parser = argparse.ArgumentParser(description="Manage DuckDB catalog")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("rebuild", help="Rebuild the DuckDB catalog")
    sub.add_parser("stats", help="Show catalog statistics")

    args = parser.parse_args()
    config = get_config()

    if args.command == "rebuild":
        create_catalog(config.data_dir, config.duckdb_file)
        print("✅ Catalog rebuilt")
    elif args.command == "stats":
        stats = catalog_stats(config.duckdb_file)
        for view, count in stats.items():
            print(f"  {view}: {count}")
    else:
        parser.print_help()
        sys.exit(1)
