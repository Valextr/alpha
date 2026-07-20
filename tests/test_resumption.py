"""Tests for pipeline resumption logic (skip stale/fresh tickers)."""

import pytest
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from src.data.config import DataConfig, get_config
from src.data.ingestion import (
    find_ticker_max_date,
    should_fetch_ticker,
    load_bronze_from_disk,
    save_parquet,
    STALENESS_THRESHOLD_DAYS,
)


@pytest.fixture()
def tmp_data_dir(tmp_path):
    """Create a temporary data directory with bronze structure."""
    data_dir = tmp_path / "data"
    bronze_daily = data_dir / "bronze" / "daily"
    bronze_divs = data_dir / "bronze" / "dividends"
    bronze_actions = data_dir / "bronze" / "corporate_actions"
    silver = data_dir / "silver" / "daily"
    gold = data_dir / "gold" / "daily"
    catalog = data_dir / "_catalog"
    for d in [bronze_daily, bronze_divs, bronze_actions, silver, gold, catalog]:
        d.mkdir(parents=True)
    return data_dir


@pytest.fixture()
def config(tmp_data_dir):
    return DataConfig(data_dir=tmp_data_dir)


class TestFindTickerMaxDate:
    def test_no_data_returns_none(self, config):
        assert find_ticker_max_date(config.data_dir, "bronze", "AAPL") is None

    def test_single_year(self, config):
        # Write bronze bars for SPY in year 2024
        out_dir = config.data_dir / "bronze" / "daily" / "ticker=SPY" / "year=2024"
        out_dir.mkdir(parents=True)
        df = pl.DataFrame({
            "date": [date(2024, 1, 1), date(2024, 6, 15), date(2024, 12, 31)],
            "close": [100.0, 120.0, 130.0],
        })
        save_parquet(df, config.data_dir / "bronze" / "daily", ticker="SPY", year="2024")
        max_date = find_ticker_max_date(config.data_dir, "bronze", "SPY")
        assert max_date == date(2024, 12, 31)

    def test_multiple_years(self, config):
        # Write year 2023
        df23 = pl.DataFrame({
            "date": [date(2023, 6, 1), date(2023, 12, 31)],
            "close": [90.0, 100.0],
        })
        save_parquet(df23, config.data_dir / "bronze" / "daily", ticker="MSFT", year="2023")
        # Write year 2024
        df24 = pl.DataFrame({
            "date": [date(2024, 1, 2), date(2024, 3, 10)],
            "close": [101.0, 110.0],
        })
        save_parquet(df24, config.data_dir / "bronze" / "daily", ticker="MSFT", year="2024")
        max_date = find_ticker_max_date(config.data_dir, "bronze", "MSFT")
        assert max_date == date(2024, 3, 10)

    def test_nonexistent_ticker(self, config):
        # Write data for AAPL but query for GOOGL
        df = pl.DataFrame({
            "date": [date(2024, 1, 1)],
            "close": [100.0],
        })
        save_parquet(df, config.data_dir / "bronze" / "daily", ticker="AAPL", year="2024")
        assert find_ticker_max_date(config.data_dir, "bronze", "GOOGL") is None


class TestShouldFetchTicker:
    def test_no_existing_data_fetches_full(self, config):
        should, eff_start = should_fetch_ticker(
            config, "AAPL", date(2020, 1, 1), date(2024, 12, 31)
        )
        assert should is True
        assert eff_start == date(2020, 1, 1)

    def test_fresh_data_skips(self, config):
        # Data within staleness threshold
        yesterday = date.today() - timedelta(days=1)
        df = pl.DataFrame({
            "date": [yesterday],
            "close": [100.0],
        })
        save_parquet(df, config.data_dir / "bronze" / "daily", ticker="SPY",
                     year=str(yesterday.year))
        should, eff_start = should_fetch_ticker(
            config, "SPY", date(2020, 1, 1), date.today()
        )
        assert should is False
        assert eff_start is None

    def test_stale_data_fetches_incremental(self, config):
        # Data is 10 days old
        old_date = date.today() - timedelta(days=10)
        df = pl.DataFrame({
            "date": [old_date],
            "close": [100.0],
        })
        save_parquet(df, config.data_dir / "bronze" / "daily", ticker="QQQ",
                     year=str(old_date.year))
        should, eff_start = should_fetch_ticker(
            config, "QQQ", date(2020, 1, 1), date.today()
        )
        assert should is True
        assert eff_start == old_date + timedelta(days=1)

    def test_force_override(self, config):
        # Fresh data but force=True should still fetch
        yesterday = date.today() - timedelta(days=1)
        df = pl.DataFrame({
            "date": [yesterday],
            "close": [100.0],
        })
        save_parquet(df, config.data_dir / "bronze" / "daily", ticker="DIA",
                     year=str(yesterday.year))
        should, eff_start = should_fetch_ticker(
            config, "DIA", date(2020, 1, 1), date.today(), force=True
        )
        assert should is True
        assert eff_start == date(2020, 1, 1)


class TestLoadBronzeFromDisk:
    def test_empty_dir(self, config):
        bars, divs, actions = load_bronze_from_disk(
            config, ["AAPL"], date(2024, 1, 1), date(2024, 12, 31)
        )
        assert bars.is_empty()
        assert divs.is_empty()
        assert actions.is_empty()

    def test_loads_bars(self, config):
        df = pl.DataFrame({
            "date": [date(2024, 1, 15), date(2024, 6, 15)],
            "close": [100.0, 120.0],
            "ticker": ["SPY", "SPY"],
        })
        save_parquet(df, config.data_dir / "bronze" / "daily", ticker="SPY", year="2024")
        bars, divs, actions = load_bronze_from_disk(
            config, ["SPY"], date(2024, 1, 1), date(2024, 12, 31)
        )
        assert not bars.is_empty()
        assert bars.height == 2

    def test_filters_by_date_range(self, config):
        df = pl.DataFrame({
            "date": [date(2023, 1, 1), date(2024, 1, 1), date(2024, 6, 15)],
            "close": [90.0, 100.0, 120.0],
            "ticker": ["SPY", "SPY", "SPY"],
        })
        save_parquet(df, config.data_dir / "bronze" / "daily", ticker="SPY", year="2023")
        save_parquet(df, config.data_dir / "bronze" / "daily", ticker="SPY", year="2024")
        # Only load from mid-2024 onward
        bars, _, _ = load_bronze_from_disk(
            config, ["SPY"], date(2024, 6, 1), date(2024, 12, 31)
        )
        assert not bars.is_empty()
        # Should only have June 15 data (date 2024-06-15 is >= 2024-06-01)
        min_date = bars["date"].min()
        if hasattr(min_date, 'to_python'):
            min_date = min_date.to_python()
        assert min_date >= date(2024, 6, 1)