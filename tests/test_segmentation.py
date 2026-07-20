"""Tests for validation segmentation module."""

from datetime import date, timedelta

import polars as pl
import pytest

from src.validation.segmentation import (
    DataSegment,
    SegmentationConfig,
    get_default_segmentation,
    get_equal_segments,
    segment_dataframe,
    segment_query,
    add_segment_column,
)


# ── helpers ──────────────────────────────────────────────────────────


def _trading_dates(start: date, n: int) -> list[date]:
    """Return *n* weekday dates starting from *start*."""
    dates: list[date] = []
    d = start
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    return dates


def _make_segmented_bars(
    num_train: int = 200,
    num_val: int = 100,
    num_hold: int = 100,
) -> pl.DataFrame:
    """Create bars spanning train/validation/hold-back periods.

    Train: 2014-01-02 through ~2019
    Validation: 2020 through ~2022
    Hold-back: 2023 through ~2024
    """
    all_dates = (
        _trading_dates(date(2014, 1, 2), num_train)
        + _trading_dates(date(2020, 1, 2), num_val)
        + _trading_dates(date(2023, 1, 3), num_hold)
    )
    closes = [100.0 + i * 0.1 for i in range(len(all_dates))]
    return pl.DataFrame(
        {
            "ticker": "TEST",
            "date": all_dates,
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": 1_000_000,
        }
    )


# ── SegmentationConfig ──────────────────────────────────────────────


class TestSegmentationConfig:
    def test_default_config_is_valid(self):
        cfg = get_default_segmentation()
        assert cfg.train_start < cfg.train_end
        assert cfg.train_end == cfg.validation_start
        assert cfg.validation_start < cfg.validation_end
        assert cfg.validation_end == cfg.hold_back_start

    def test_bad_boundaries_raise(self):
        with pytest.raises(ValueError, match="ordered and contiguous"):
            SegmentationConfig(
                train_start=date(2014, 1, 1),
                train_end=date(2020, 1, 1),
                validation_start=date(2019, 1, 1),  # before train_end
                validation_end=date(2023, 1, 1),
                hold_back_start=date(2023, 1, 1),
            )

    def test_non_contiguous_boundaries_raise(self):
        # Gap between train_end and validation_start
        with pytest.raises(ValueError, match="ordered and contiguous"):
            SegmentationConfig(
                train_start=date(2014, 1, 2),
                train_end=date(2020, 1, 2),
                validation_start=date(2020, 6, 1),  # gap
                validation_end=date(2023, 1, 3),
                hold_back_start=date(2023, 1, 3),
            )


# ── segment_dataframe ───────────────────────────────────────────────


class TestSegmentDataFrame:
    @pytest.fixture()
    def bars(self):
        return _make_segmented_bars(num_train=200, num_val=100, num_hold=100)

    def test_default_split_counts(self, bars):
        result = segment_dataframe(bars)
        assert len(result.train) == 200
        assert len(result.validation) == 100
        assert len(result.hold_back) == 100

    def test_no_overlap_between_segments(self, bars):
        result = segment_dataframe(bars)
        train_max = result.train["date"].max()
        val_min = result.validation["date"].min()
        val_max = result.validation["date"].max()
        hb_min = result.hold_back["date"].min()
        assert train_max < val_min, "train bleeds into validation"
        assert val_max < hb_min, "validation bleeds into hold-back"

    def test_combined_dataframe(self, bars):
        result = segment_dataframe(bars)
        combined = result.combined
        assert combined.shape[0] == bars.shape[0]
        assert "segment" in combined.columns
        assert set(combined["segment"].unique().to_list()) == {
            DataSegment.TRAIN.value,
            DataSegment.VALIDATION.value,
            DataSegment.HOLD_BACK.value,
        }

    def test_empty_dataframe(self):
        empty_df = pl.DataFrame({"date": [], "ticker": []})
        result = segment_dataframe(empty_df)
        assert len(result.train) == 0
        assert len(result.validation) == 0
        assert len(result.hold_back) == 0

    def test_missing_date_column_raises(self):
        bad_df = pl.DataFrame({"ticker": ["A"], "close": [100.0]})
        with pytest.raises(ValueError, match="must have a 'date' column"):
            segment_dataframe(bad_df)

    def test_custom_config(self, bars):
        custom = SegmentationConfig(
            train_start=date(2014, 1, 2),
            train_end=date(2018, 1, 2),
            validation_start=date(2018, 1, 2),
            validation_end=date(2023, 1, 3),
            hold_back_start=date(2023, 1, 3),
        )
        result = segment_dataframe(bars, config=custom)
        # Everything from 2014-2017 goes to train, rest of pre-2023 to validation
        assert len(result.hold_back) == 100  # 2023+ bars unchanged
        assert len(result.train) + len(result.validation) + len(result.hold_back) == bars.shape[0]

    def test_preserves_columns(self, bars):
        result = segment_dataframe(bars)
        for seg_df in [result.train, result.validation, result.hold_back]:
            assert "ticker" in seg_df.columns
            assert "close" in seg_df.columns

    def test_sorted_output(self, bars):
        result = segment_dataframe(bars)
        for seg_df in [result.train, result.validation, result.hold_back]:
            dates = seg_df["date"].to_list()
            assert dates == sorted(dates)


# ── equal segments ───────────────────────────────────────────────────


class TestEqualSegments:
    def test_approximate_ratios(self):
        df = _make_segmented_bars(num_train=300, num_val=100, num_hold=100)
        cfg = get_equal_segments(df)
        result = segment_dataframe(df, cfg)
        total = len(df)
        train_pct = len(result.train) / total * 100
        val_pct = len(result.validation) / total * 100
        hold_pct = len(result.hold_back) / total * 100
        assert abs(train_pct - 60.0) < 5.0
        assert abs(val_pct - 20.0) < 5.0
        assert abs(hold_pct - 20.0) < 5.0


# ── add_segment_column ──────────────────────────────────────────────


class TestAddSegmentColumn:
    def test_tags_correctly(self):
        df = _make_segmented_bars(num_train=100, num_val=50, num_hold=50)
        tagged = add_segment_column(df)
        assert "segment" in tagged.columns
        train_count = tagged.filter(pl.col("segment") == "train").height
        val_count = tagged.filter(pl.col("segment") == "validation").height
        hold_count = tagged.filter(pl.col("segment") == "hold_back").height
        assert train_count == 100
        assert val_count == 50
        assert hold_count == 50

    def test_preserves_row_count(self):
        df = _make_segmented_bars()
        tagged = add_segment_column(df)
        assert tagged.shape[0] == df.shape[0]


# ── segment_query ────────────────────────────────────────────────────


class TestSegmentQuery:
    def test_from_duckdb(self):
        result = segment_query("data/alpha.duckdb", "gold_daily")
        assert len(result.train) > 0
        assert len(result.validation) > 0
        assert len(result.hold_back) > 0
        assert set(result.train.get_column("ticker").unique().to_list()) == {
            "AAPL", "AMZN", "GOOGL", "META", "MSFT",
        }

    def test_invalid_table_raises(self):
        import pytest
        with pytest.raises(ValueError, match="not found"):
            segment_query("data/alpha.duckdb", "nonexistent_table")


# ── stats ───────────────────────────────────────────────────────────


class TestStats:
    def test_stats_dict(self):
        df = _make_segmented_bars(num_train=200, num_val=100, num_hold=100)
        result = segment_dataframe(df)
        stats = result.stats()
        assert "train" in stats
        assert "validation" in stats
        assert "hold_back" in stats
        assert stats["total_rows"] == 400
        assert stats["tickers"] == ["TEST"]
        assert "pct" in stats["train"]
        assert "rows" in stats["train"]