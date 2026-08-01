"""Data pipeline configuration."""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Central configuration for the data pipeline."""

    # Directories
    data_dir: Path = Field(default=Path("data"), description="Root data directory")
    universe_file: Path = Field(
        default=Path("data/universe.csv"), description="Ticker universe file"
    )
    duckdb_file: Path = Field(
        default=Path("data/alpha.duckdb"), description="DuckDB database file"
    )

    # Polygon.io
    polygon_api_key: Optional[str] = Field(
        default=None, description="Polygon.io API key (env: POLYGON_API_KEY)"
    )
    polygon_base_url: str = Field(
        default="https://api.polygon.io", description="Polygon.io base URL"
    )

    # Ingestion defaults
    default_start_date: str = Field(
        default="2014-01-01", description="Default data start date (10+ years back)"
    )
    default_end_date: Optional[str] = Field(
        default=None,
        description="Default data end date (None = yesterday)",
    )
    batch_size: int = Field(default=100, description="Symbols per API batch")
    retry_attempts: int = Field(default=3, description="API retry attempts")
    retry_delay: int = Field(default=5, description="Seconds between retries")

    # Quality thresholds
    max_price_gap: float = Field(
        default=0.20, description="Max price gap threshold (20%)"
    )
    min_volume: int = Field(
        default=0, description="Volume threshold for 'thin' flag"
    )

    # Universe defaults
    default_universe: list[str] = Field(
        default_factory=lambda: [
            # Broad market ETFs
            "SPY", "QQQ", "IWM", "DIA",
            # Sector ETFs
            "XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLU", "XLB", "XLRE", "XLY",
            # Mega-cap tech
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
            # Financials
            "JPM", "BAC", "WFC", "GS", "MS", "C", "BRK-B",
            # Healthcare
            "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY",
            # Energy
            "XOM", "CVX", "COP", "SLB",
            # Consumer
            "PG", "KO", "PEP", "WMT", "COST", "NKE",
            # Industrials
            "CAT", "HON", "UNP", "BA", "GE",
            # Other large cap
            "V", "MA", "DIS", "NFLX", "AMD", "INTC", "CRM", "ADBE",
        ],
        description="Default ticker universe for prototyping",
    )

    model_config = {"env_prefix": "ALPHA_"}

    def model_post_init(self, __context) -> None:
        if self.polygon_api_key is None:
            self.polygon_api_key = os.getenv("POLYGON_API_KEY")


def get_config(**overrides) -> DataConfig:
    """Get configuration, applying overrides."""
    return DataConfig(**overrides)
