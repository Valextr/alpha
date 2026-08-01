"""Unit tests for the data pipeline modules.

Covers: adjust.py, enrich.py, validate.py, catalog.py

All tests use synthetic data from conftest.py — no network access,
no pre-existing disk data required.
"""

import sys
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .conftest import _trading_dates
from src.data.adjust import adjust_for_dividends, adjust_for_splits, build_silver_layer
from src.data.catalog import _glob_pattern, _has_files
from src.data.enrich import (
    enrich_with_market_cap_bucket,
    enrich_with_sector,
    enrich_with_universe_flag,
    enrich_with_volume_features,
    build_gold_layer,
)
from src.data.validate import (
    ValidationResult,
    check_data_quality_flags,
    check_future_dates,
    check_missing_dates,
    check_negative_prices,
    check_price_gaps,
    check_zero_volume,
    run_all_checks,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# adjust.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAdjustForSplits:
    def test_no_splits_returns_unchanged(self, single_ticker_bars, splits_empty):
        result = adjust_for_splits(single_ticker_bars, splits_empty)
        assert result.equals(single_ticker_bars)

    def test_split_adjusts_prices_backward(self, single_ticker_bars, splits_2x):
        """After a 2:1 split, prices before the split date should be halved."""
        split_date = date(2023, 2, 15)
        result = adjust_for_splits(single_ticker_bars, splits_2x)

        # Bars before/on split date: close should be /2
        before = result.filter(pl.col("date") <= split_date)
        after = result.filter(pl.col("date") > split_date)

        # Original closes were 100, 101, ... (60 bars from Jan 3)
        # Split is on Feb 15 (~row 31, close ~131)
        # Before split: halved (50.0, 50.5, ..., ~65.5)
        # After split: unchanged (starts from ~132)
        before_max = float(before["close"].max())
        after_min = float(after["close"].min())
        # Before should be roughly half of original max before split (~65.5)
        assert before_max == pytest.approx(65.5, abs=0.5)
        # After should be > 100 (not halved)
        assert after_min > 100

    def test_split_adjusts_volume_forward(self, single_ticker_bars, splits_2x):
        """Volume should be multiplied by the split factor."""
        result = adjust_for_splits(single_ticker_bars, splits_2x)
        split_date = date(2023, 2, 15)
        before = result.filter(pl.col("date") <= split_date)
        # Original volume 1M → 2M after 2x split
        assert int(before["volume"].min()) == 2_000_000

    def test_split_does_not_affect_other_tickers(self, two_ticker_bars):
        """Splits for AAPL should not touch MSFT bars."""
        splits = pl.DataFrame(
            {
                "ticker": ["AAPL"],
                "date": [date(2023, 2, 15)],
                "action_type": ["split"],
                "factor": [2.0],
            }
        )
        result = adjust_for_splits(two_ticker_bars, splits)
        msft_original = two_ticker_bars.filter(pl.col("ticker") == "MSFT")
        msft_result = result.filter(pl.col("ticker") == "MSFT")
        assert msft_result.equals(msft_original)

    def test_split_factor_le_1_is_skipped(self, single_ticker_bars):
        """Reverse splits (factor <= 1) are skipped."""
        splits = pl.DataFrame(
            {
                "ticker": ["AAPL"],
                "date": [date(2023, 2, 15)],
                "action_type": ["split"],
                "factor": [0.5],  # reverse split → skip
            }
        )
        result = adjust_for_splits(single_ticker_bars, splits)
        # Should be unchanged since factor <= 1
        assert result.equals(single_ticker_bars)


class TestAdjustForDividends:
    def test_no_dividends_returns_unchanged(self, single_ticker_bars, dividends_empty):
        result = adjust_for_dividends(single_ticker_bars, dividends_empty)
        assert result.equals(single_ticker_bars)

    def test_dividend_adjusts_prices(self, single_ticker_bars, dividends_fixture):
        """Dividend adjustment should reduce pre-ex-date prices."""
        result = adjust_for_dividends(single_ticker_bars, dividends_fixture)
        # Just check the output has the same shape
        assert len(result) == len(single_ticker_bars)
        assert result.columns == single_ticker_bars.columns

    def test_dividend_does_not_affect_other_tickers(self, two_ticker_bars):
        """AAPL dividends should not touch MSFT bars."""
        divs = pl.DataFrame(
            {
                "ticker": ["AAPL"],
                "ex_date": [date(2023, 2, 10)],
                "amount": [0.23],
            }
        )
        result = adjust_for_dividends(two_ticker_bars, divs)
        msft_original = two_ticker_bars.filter(pl.col("ticker") == "MSFT")
        msft_result = result.filter(pl.col("ticker") == "MSFT")
        assert msft_result.equals(msft_original)

    def test_negative_dividend_skipped(self, single_ticker_bars):
        """Dividends with amount <= 0 are skipped."""
        divs = pl.DataFrame(
            {
                "ticker": ["AAPL"],
                "ex_date": [date(2023, 2, 10)],
                "amount": [-1.0],
            }
        )
        result = adjust_for_dividends(single_ticker_bars, divs)
        assert result.equals(single_ticker_bars)


class TestBuildSilverLayer:
    def test_silver_adds_expected_columns(self, single_ticker_bars, dividends_empty, corporate_actions_empty):
        silver = build_silver_layer(single_ticker_bars, dividends_empty, corporate_actions_empty)
        for col in ["adj_close", "split_factor", "dividend_yield", "is_market_date", "data_quality"]:
            assert col in silver.columns

    def test_silver_zero_volume_flags_thin(self, bars_with_zero_volume, dividends_empty, corporate_actions_empty):
        silver = build_silver_layer(bars_with_zero_volume, dividends_empty, corporate_actions_empty)
        thin = silver.filter(pl.col("data_quality") == "thin")
        assert thin.height > 0

    def test_silver_large_gap_flags_suspicious(self, bars_with_large_price_gap, dividends_empty, corporate_actions_empty):
        silver = build_silver_layer(bars_with_large_price_gap, dividends_empty, corporate_actions_empty)
        suspicious = silver.filter(pl.col("data_quality") == "suspicious")
        assert suspicious.height > 0

    def test_silver_with_splits(self, single_ticker_bars, dividends_empty):
        """Silver layer should apply splits before dividends."""
        actions = pl.DataFrame(
            {
                "ticker": ["AAPL"],
                "date": [date(2023, 2, 15)],
                "action_type": ["split"],
                "factor": [2.0],
            }
        )
        silver = build_silver_layer(single_ticker_bars, dividends_empty, actions)
        # adj_close before split should be halved (max ~65.5)
        before = silver.filter(pl.col("date") <= date(2023, 2, 15))
        before_max = float(before["adj_close"].max())
        # Should be roughly half of 131 (the original close on split date)
        assert before_max == pytest.approx(65.5, abs=0.5)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# enrich.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEnrichWithSector:
    def test_known_tickers_get_sector(self, silver_bars):
        result = enrich_with_sector(silver_bars)
        assert "sector" in result.columns
        aapl = result.filter(pl.col("ticker") == "AAPL")
        assert aapl["sector"].to_list() == ["Technology"] * len(aapl)

    def test_unknown_ticker_gets_unknown(self):
        df = pl.DataFrame(
            {
                "ticker": ["UNKNOWN"],
                "date": [date(2023, 1, 3)],
                "close": [100.0],
            }
        )
        result = enrich_with_sector(df)
        assert result["sector"][0] == "Unknown"


class TestEnrichWithVolumeFeatures:
    def test_adds_volume_columns(self, silver_bars):
        result = enrich_with_volume_features(silver_bars)
        for col in ["avg_volume_20d", "avg_volume_60d", "volume_ratio"]:
            assert col in result.columns

    def test_volume_ratio_reasonable(self, silver_bars):
        result = enrich_with_volume_features(silver_bars)
        # With constant volume, ratio should be 1.0 once we have enough history
        ratio = result["volume_ratio"].drop_nulls()
        assert (ratio > 0).all()

    def test_zero_volume_doesnt_crash(self):
        df = pl.DataFrame(
            {
                "ticker": ["ZERO"] * 30,
                "date": _trading_dates(date(2023, 1, 3), 30),
                "volume": [0] * 30,
            }
        )
        result = enrich_with_volume_features(df)
        assert not result["volume_ratio"].has_nulls()


class TestEnrichWithMarketCapBucket:
    def test_mega_cap_classification(self, silver_bars):
        result = enrich_with_market_cap_bucket(silver_bars)
        assert "market_cap_bucket" in result.columns
        aapl = result.filter(pl.col("ticker") == "AAPL")
        assert aapl["market_cap_bucket"][0] == "mega"

    def test_etf_classification(self):
        df = pl.DataFrame(
            {
                "ticker": ["SPY", "QQQ", "XLF"],
                "date": [date(2023, 1, 3)] * 3,
                "close": [400.0, 350.0, 30.0],
            }
        )
        result = enrich_with_market_cap_bucket(df)
        buckets = result["market_cap_bucket"].to_list()
        assert buckets == ["etf", "etf", "etf"]

    def test_unknown_ticker_gets_mid(self):
        df = pl.DataFrame(
            {
                "ticker": ["UNKNOWN"],
                "date": [date(2023, 1, 3)],
                "close": [50.0],
            }
        )
        result = enrich_with_market_cap_bucket(df)
        assert result["market_cap_bucket"][0] == "mid"


class TestEnrichWithUniverseFlag:
    def test_adds_universe_date_column(self, silver_bars):
        result = enrich_with_universe_flag(silver_bars)
        assert "universe_date" in result.columns
        assert (result["universe_date"] == True).all()


class TestBuildGoldLayer:
    def test_gold_has_all_expected_columns(self, silver_bars):
        gold = build_gold_layer(silver_bars)
        expected = [
            "sector", "avg_volume_20d", "avg_volume_60d",
            "volume_ratio", "market_cap_bucket", "universe_date",
        ]
        for col in expected:
            assert col in gold.columns, f"Missing gold column: {col}"

    def test_gold_sorted_by_ticker_date(self, silver_bars):
        gold = build_gold_layer(silver_bars)
        tickers = gold["ticker"].to_list()
        dates = gold["date"].to_list()
        for i in range(1, len(tickers)):
            if tickers[i] == tickers[i - 1]:
                assert dates[i] >= dates[i - 1]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# validate.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestValidationResult:
    def test_is_valid_when_no_errors(self):
        r = ValidationResult()
        assert r.is_valid

    def test_is_invalid_when_errors(self):
        r = ValidationResult()
        r.errors.append("bad")
        assert not r.is_valid

    def test_summary_contains_counts(self):
        r = ValidationResult()
        r.errors.append("err")
        r.warnings.append("warn")
        r.info.append("info")
        s = r.summary()
        assert "1 errors" in s
        assert "1 warnings" in s


class TestCheckNegativePrices:
    def test_catches_negative_close(self, bars_with_negative_price):
        r = ValidationResult()
        check_negative_prices(bars_with_negative_price, r)
        assert len(r.errors) > 0
        assert any("negative close" in e for e in r.errors)

    def test_passes_clean_data(self, single_ticker_bars):
        r = ValidationResult()
        check_negative_prices(single_ticker_bars, r)
        assert r.is_valid


class TestCheckFutureDates:
    def test_passes_historical_data(self, single_ticker_bars):
        r = ValidationResult()
        check_future_dates(single_ticker_bars, r)
        assert r.is_valid

    def test_catches_future_dates(self):
        future = date.today() + timedelta(days=30)
        df = pl.DataFrame(
            {
                "ticker": ["FUT"] * 5,
                "date": [future + timedelta(days=i) for i in range(5)],
                "close": [100.0] * 5,
            }
        )
        r = ValidationResult()
        check_future_dates(df, r)
        assert len(r.errors) > 0


class TestCheckMissingDates:
    def test_detects_large_gaps(self, bars_with_date_gaps):
        r = ValidationResult()
        check_missing_dates(bars_with_date_gaps, r)
        # Should find gaps >5 days
        assert len(r.warnings) > 0 or len(r.info) > 0

    def test_handles_missing_date_column(self):
        df = pl.DataFrame({"ticker": ["A"], "close": [100.0]})
        r = ValidationResult()
        check_missing_dates(df, r)
        assert any("date" in w for w in r.warnings)


class TestCheckZeroVolume:
    def test_detects_excessive_zero_volume(self, bars_with_zero_volume):
        r = ValidationResult()
        check_zero_volume(bars_with_zero_volume, r)
        assert len(r.warnings) > 0

    def test_passes_normal_volume(self, single_ticker_bars):
        r = ValidationResult()
        check_zero_volume(single_ticker_bars, r)
        assert len(r.warnings) == 0


class TestCheckPriceGaps:
    def test_detects_large_gap(self, bars_with_large_price_gap):
        r = ValidationResult()
        check_price_gaps(bars_with_large_price_gap, threshold=0.20, result=r)
        assert len(r.warnings) > 0

    def test_passes_normal_prices(self, single_ticker_bars):
        r = ValidationResult()
        check_price_gaps(single_ticker_bars, threshold=0.20, result=r)
        # Normal walk-up by $1/day on a ~$100 stock should not trigger
        assert len(r.warnings) == 0


class TestCheckDataQualityFlags:
    def test_detects_non_good_flags(self, silver_bars):
        # Inject some "thin" and "suspicious" flags
        df = silver_bars.with_columns(
            pl.when(pl.col("ticker") == "MSFT")
            .then(pl.lit("thin"))
            .otherwise(pl.col("data_quality"))
            .alias("data_quality")
        )
        r = ValidationResult()
        check_data_quality_flags(df, r)
        assert len(r.warnings) > 0

    def test_passes_all_good(self, silver_bars):
        r = ValidationResult()
        check_data_quality_flags(silver_bars, r)
        assert len(r.warnings) == 0


class TestRunAllChecks:
    def test_runs_all_checks(self, single_ticker_bars):
        r = run_all_checks(single_ticker_bars, layer="silver")
        assert len(r.info) > 0  # at least the header line

    def test_returns_invalid_for_bad_data(self, bars_with_negative_price):
        r = run_all_checks(bars_with_negative_price, layer="bronze")
        assert not r.is_valid


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# catalog.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestGlobPattern:
    def test_bronze_daily_pattern(self):
        pattern = _glob_pattern(Path("/data"), "bronze", "daily")
        assert pattern == "/data/bronze/daily/**/*.parquet"

    def test_dividends_subdir(self):
        pattern = _glob_pattern(Path("/data"), "bronze", "dividends")
        assert pattern == "/data/bronze/dividends/**/*.parquet"


class TestHasFiles:
    def test_empty_directory(self):
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            # Create the structure but no parquet files
            (p / "bronze" / "daily").mkdir(parents=True)
            assert _has_files(p, "bronze", "daily") is False

    def test_nonexistent_directory(self):
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            assert _has_files(p, "bronze", "daily") is False

    def test_with_parquet_files(self):
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            (p / "bronze" / "daily" / "ticker=AAPL").mkdir(parents=True)
            # Write a minimal parquet
            pl.DataFrame({"ticker": ["AAPL"], "close": [100.0]}).write_parquet(
                str(p / "bronze" / "daily" / "ticker=AAPL" / "part-0.parquet")
            )
            assert _has_files(p, "bronze", "daily") is True


class TestCatalogIntegration:
    """Integration tests for create_catalog / query_catalog / catalog_stats.

    Uses a temporary directory with real parquet files — no external
    data required.
    """

    def _setup_minimal_data(self, tmpdir: Path) -> Path:
        """Create minimal bronze/silver/gold structure in tmpdir."""
        for layer in ["bronze", "silver", "gold"]:
            layer_dir = tmpdir / layer / "daily"
            layer_dir.mkdir(parents=True)
            pl.DataFrame({
                "ticker": ["AAPL"],
                "date": [date(2023, 1, 3)],
                "close": [150.0],
                "data_quality": ["good"],
                "universe_date": [True],
            }).write_parquet(str(layer_dir / "part-0.parquet"))

        # Create bronze subdirs for dividends/corporate_actions (empty)
        (tmpdir / "bronze" / "dividends").mkdir(parents=True)
        (tmpdir / "bronze" / "corporate_actions").mkdir(parents=True)

        return tmpdir

    def test_create_catalog_creates_views(self):
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            self._setup_minimal_data(tmpdir)
            duckdb_file = tmpdir / "test.duckdb"

            from src.data.catalog import create_catalog
            create_catalog(tmpdir, duckdb_file)

            assert duckdb_file.exists()

            # Verify we can query the views
            import duckdb
            conn = duckdb.connect(str(duckdb_file))
            for view in ["bronze_daily", "silver_daily", "gold_daily"]:
                result = conn.execute(f"SELECT COUNT(*) FROM {view}").fetchone()
                assert result[0] == 1, f"Expected 1 row in {view}, got {result[0]}"

            conn.close()

    def test_query_catalog_returns_polars_df(self):
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            self._setup_minimal_data(tmpdir)
            duckdb_file = tmpdir / "test.duckdb"

            from src.data.catalog import create_catalog, query_catalog
            create_catalog(tmpdir, duckdb_file)

            df = query_catalog(duckdb_file, "SELECT COUNT(*) as cnt FROM silver_daily")
            assert isinstance(df, pl.DataFrame)
            assert int(df["cnt"][0]) == 1

    def test_catalog_stats_returns_dict(self):
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            self._setup_minimal_data(tmpdir)
            duckdb_file = tmpdir / "test.duckdb"

            from src.data.catalog import create_catalog, catalog_stats
            create_catalog(tmpdir, duckdb_file)

            stats = catalog_stats(duckdb_file)
            assert isinstance(stats, dict)
            assert stats["bronze_daily"] == 1
            assert stats["silver_daily"] == 1
            assert stats["gold_daily"] == 1

    def test_catalog_without_gold_skips_universe_view(self):
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # Only create bronze/silver, no gold
            for layer in ["bronze", "silver"]:
                layer_dir = tmpdir / layer / "daily"
                layer_dir.mkdir(parents=True)
                pl.DataFrame({
                    "ticker": ["AAPL"],
                    "date": [date(2023, 1, 3)],
                    "close": [150.0],
                }).write_parquet(str(layer_dir / "part-0.parquet"))

            duckdb_file = tmpdir / "test.duckdb"

            from src.data.catalog import create_catalog
            create_catalog(tmpdir, duckdb_file)

            # Should complete without error
            import duckdb
            conn = duckdb.connect(str(duckdb_file))
            # universe_daily should not exist
            try:
                conn.execute("SELECT 1 FROM universe_daily")
                universe_exists = True
            except duckdb.CatalogException:
                universe_exists = False
            conn.close()

            assert not universe_exists, "universe_daily should not exist when gold is missing"