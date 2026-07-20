"""Validation engine tests — segmentation and walk-forward analysis."""

import sys
from datetime import date, timedelta

import polars as pl
import pytest

sys.path.insert(0, str(__file__.rsplit("/", 2)[0]))

from src.validation.segmentation import (
    DataSegment,
    SegmentationConfig,
    SegmentationResult,
    get_default_segmentation,
    get_equal_segments,
    segment_dataframe,
    add_segment_column,
)
from src.validation.walkforward import (
    WalkForwardConfig,
    WalkForwardResult,
    FoldResult,
    walk_forward,
    walk_forward_on_holdback,
    walk_forward_summary,
    _detect_signal_cols,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _trading_dates(start: date, n: int) -> list[date]:
    """Return *n* weekday dates starting from *start*."""
    dates: list[date] = []
    d = start
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    return dates


def _make_long_series(
    ticker: str,
    num_days: int = 3400,
    base_close: float = 100.0,
    start_date: date = date(2014, 1, 2),
    volume: int = 1_000_000,
) -> pl.DataFrame:
    """Create a multi-year OHLCV series spanning ~2014-2027."""
    import math

    dates = _trading_dates(start_date, num_days)
    closes = []
    c = base_close
    for i in range(num_days):
        c = c * (1.0 + 0.01 * math.sin(i * 0.05))
        closes.append(round(c, 2))

    return pl.DataFrame(
        {
            "ticker": [ticker] * num_days,
            "date": dates,
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [max(c - 1, 1.0) for c in closes],
            "close": closes,
            "volume": [volume + (i % 10) * 100_000 for i in range(num_days)],
        }
    )


def _multi_ticker_long(
    tickers: list[str] | None = None,
    num_days: int = 3400,
) -> pl.DataFrame:
    """Create multi-ticker data spanning ~2014-2027."""
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL"]
    frames = [_make_long_series(t, num_days=num_days) for t in tickers]
    return pl.concat(frames, how="vertical_relaxed").sort(["ticker", "date"])


def _add_dummy_signal(df: pl.DataFrame) -> pl.DataFrame:
    """Add a simple signal column for testing walk-forward."""
    return df.with_columns(
        (pl.col("close").pct_change().over("ticker") * 3)
        .clip(-1, 1)
        .alias("signal_test_momentum")
    )


# ── Segmentation tests ──────────────────────────────────────────────

class TestSegmentation:
    """Data segmentation — train/validation/hold-back splits."""

    def _setup(self) -> pl.DataFrame:
        return _multi_ticker_long(num_days=3400)

    def test_default_segment_produces_three_segments(self):
        df = self._setup()
        result = segment_dataframe(df)
        assert isinstance(result, SegmentationResult)
        assert len(result.train) > 0
        assert len(result.validation) > 0
        assert len(result.hold_back) > 0

    def test_segments_are_temporally_ordered(self):
        df = self._setup()
        result = segment_dataframe(df)

        train_max = result.train["date"].max()
        val_min = result.validation["date"].min()
        val_max = result.validation["date"].max()
        hold_min = result.hold_back["date"].min()

        assert train_max < val_min
        assert val_max < hold_min

    def test_default_boundaries(self):
        config = get_default_segmentation()
        assert config.train_start == date(2014, 1, 2)
        assert config.train_end == date(2020, 1, 2)
        assert config.validation_end == date(2023, 1, 3)
        assert config.hold_back_start == date(2023, 1, 3)

    def test_segments_partition_data(self):
        df = self._setup()
        result = segment_dataframe(df)
        total = len(result.train) + len(result.validation) + len(result.hold_back)
        assert total == len(df)

    def test_segments_are_disjoint(self):
        df = self._setup()
        result = segment_dataframe(df)

        train_dates = set(result.train["date"].to_list())
        val_dates = set(result.validation["date"].to_list())
        hold_dates = set(result.hold_back["date"].to_list())

        assert len(train_dates & val_dates) == 0
        assert len(val_dates & hold_dates) == 0
        assert len(train_dates & hold_dates) == 0

    def test_custom_segmentation_dates(self):
        df = self._setup()
        config = SegmentationConfig(
            train_start=date(2014, 1, 2),
            train_end=date(2018, 1, 1),
            validation_start=date(2018, 1, 1),
            validation_end=date(2021, 1, 1),
            hold_back_start=date(2021, 1, 1),
        )
        result = segment_dataframe(df, config)
        assert result.train["date"].max() < date(2018, 1, 1)
        assert result.validation["date"].max() < date(2021, 1, 1)

    def test_segmentation_config_enforces_ordering(self):
        with pytest.raises(ValueError, match="ordered"):
            SegmentationConfig(
                train_start=date(2020, 1, 1),
                train_end=date(2018, 1, 1),
                validation_start=date(2018, 1, 1),
                validation_end=date(2023, 1, 1),
                hold_back_start=date(2023, 1, 1),
            )

    def test_segmentation_config_requires_contiguous(self):
        with pytest.raises(ValueError, match="contiguous"):
            SegmentationConfig(
                train_start=date(2014, 1, 1),
                train_end=date(2018, 1, 1),
                validation_start=date(2019, 1, 1),
                validation_end=date(2023, 1, 1),
                hold_back_start=date(2023, 1, 1),
            )

    def test_empty_dataframe(self):
        df = pl.DataFrame({"ticker": [], "date": []}).with_columns(
            pl.col("ticker").cast(pl.Utf8),
            pl.col("date").cast(pl.Date),
        )
        result = segment_dataframe(df)
        assert len(result.train) == 0
        assert len(result.validation) == 0
        assert len(result.hold_back) == 0

    def test_missing_date_column_raises(self):
        df = pl.DataFrame({"ticker": ["A"], "value": [1]})
        with pytest.raises(ValueError, match="date"):
            segment_dataframe(df)

    def test_equal_segments(self):
        df = self._setup()
        config = get_equal_segments(df)
        result = segment_dataframe(df, config)
        total = len(result.train) + len(result.validation) + len(result.hold_back)
        assert len(result.train) / total > 0.5
        assert len(result.hold_back) / total > 0.1

    def test_segment_result_stats(self):
        df = self._setup()
        result = segment_dataframe(df)
        stats = result.stats()
        assert "train" in stats
        assert "validation" in stats
        assert "hold_back" in stats
        assert stats["total_rows"] == len(df)
        assert len(stats["tickers"]) == 3

    def test_segment_result_repr(self):
        df = self._setup()
        result = segment_dataframe(df)
        repr_str = repr(result)
        assert "train=" in repr_str
        assert "val=" in repr_str
        assert "hold_back=" in repr_str

    def test_add_segment_column(self):
        df = self._setup()
        tagged = add_segment_column(df)
        assert "segment" in tagged.columns
        values = set(tagged["segment"].to_list())
        assert DataSegment.TRAIN.value in values
        assert DataSegment.VALIDATION.value in values
        assert DataSegment.HOLD_BACK.value in values

    def test_combined_dataframe(self):
        df = self._setup()
        result = segment_dataframe(df)
        assert result.combined is not None
        assert len(result.combined) == len(df)
        assert "segment" in result.combined.columns


# ── Walk-forward tests ──────────────────────────────────────────────

class TestWalkForward:
    """Walk-forward analysis — rolling train/test windows."""

    def _setup(self) -> pl.DataFrame:
        df = _multi_ticker_long(num_days=3400)
        return _add_dummy_signal(df)

    def test_produces_folds(self):
        df = self._setup()
        result = walk_forward(df)
        assert len(result.folds) > 0

    def test_folds_are_temporally_ordered(self):
        df = self._setup()
        result = walk_forward(df)
        for i in range(1, len(result.folds)):
            prev = result.folds[i - 1]
            curr = result.folds[i]
            assert curr.eval_start >= prev.eval_start

    def test_eval_after_train(self):
        df = self._setup()
        result = walk_forward(df)
        for fold in result.folds:
            assert fold.train_end < fold.eval_start
            assert fold.eval_start < fold.eval_end

    def test_aggregated_metrics_exist(self):
        df = self._setup()
        result = walk_forward(df)
        sig = "signal_test_momentum"
        assert sig in result.aggregated
        assert result.aggregated[sig][1]["fold_count"] > 0

    def test_mean_ic_reasonable(self):
        df = self._setup()
        result = walk_forward(df)
        sig = "signal_test_momentum"
        for h in (1, 5, 21):
            agg = result.aggregated[sig][h]
            if agg["fold_count"] > 0:
                assert abs(agg["mean_ic"]) <= 1.0 + 1e-6

    def test_ic_stability_computed(self):
        df = self._setup()
        result = walk_forward(df)
        sig = "signal_test_momentum"
        agg = result.aggregated[sig][1]
        if agg["fold_count"] > 1:
            assert agg["std_ic"] >= 0.0
            if agg["ic_cv"] is not None:
                assert agg["ic_cv"] >= 0.0

    def test_fold_metrics_dataframe_has_rows(self):
        df = self._setup()
        result = walk_forward(df)
        assert result.fold_metrics is not None
        assert result.fold_metrics.height > 0
        assert "ic" in result.fold_metrics.columns
        assert "signal" in result.fold_metrics.columns
        assert "horizon" in result.fold_metrics.columns

    def test_custom_config(self):
        df = self._setup()
        config = WalkForwardConfig(
            train_window_days=250,
            eval_window_days=30,
            step_days=15,
        )
        result = walk_forward(df, config)
        assert len(result.folds) > 0

    def test_small_dataframe_produces_no_folds(self):
        """30 days is too short for the default 504-day train window."""
        dates = _trading_dates(date(2020, 1, 1), 30)
        df = pl.DataFrame({
            "ticker": ["X"] * 30,
            "date": dates,
            "close": [100.0 + i for i in range(30)],
            "open": [100.0 + i for i in range(30)],
            "high": [101.0 + i for i in range(30)],
            "low": [99.0 + i for i in range(30)],
            "volume": [1_000_000] * 30,
        }).with_columns(
            pl.lit(0.5).alias("signal_test")
        )
        result = walk_forward(df)
        assert len(result.folds) == 0

    def test_no_signal_columns_raises(self):
        df = pl.DataFrame({
            "ticker": ["X"],
            "date": [date(2020, 1, 1)],
            "close": [100.0],
        })
        with pytest.raises(ValueError, match="signal"):
            walk_forward(df)

    def test_fold_results_have_correct_fields(self):
        df = self._setup()
        result = walk_forward(df)
        fold = result.folds[0]
        assert isinstance(fold.fold_index, int)
        assert isinstance(fold.train_start, date)
        assert isinstance(fold.train_end, date)
        assert isinstance(fold.eval_start, date)
        assert isinstance(fold.eval_end, date)
        assert isinstance(fold.signal_columns, list)
        assert isinstance(fold.metrics, dict)

    def test_fold_results_have_nonempty_signal_list(self):
        df = self._setup()
        result = walk_forward(df)
        assert "signal_test_momentum" in result.folds[0].signal_columns


class TestWalkForwardOnHoldback:
    """Walk-forward restricted to hold-back segment."""

    def test_holdback_walk_forward(self):
        df = _multi_ticker_long(num_days=3400)
        df = _add_dummy_signal(df)
        result = walk_forward_on_holdback(df)
        assert len(result.folds) > 0
        # All eval dates should be in the hold-back period (2023+)
        for fold in result.folds:
            assert fold.eval_start >= date(2023, 1, 3)


class TestWalkForwardSummary:
    """Human-readable summary output."""

    def test_summary_is_nonempty(self):
        df = _multi_ticker_long(num_days=3400)
        df = _add_dummy_signal(df)
        result = walk_forward(df)
        summary = walk_forward_summary(result)
        assert len(summary) > 30
        assert "WALK-FORWARD" in summary
        assert "signal_test_momentum" in summary

    def test_summary_includes_fold_count(self):
        df = _multi_ticker_long(num_days=3400)
        df = _add_dummy_signal(df)
        result = walk_forward(df)
        summary = walk_forward_summary(result)
        assert "Folds:" in summary

    def test_empty_result_summary(self):
        result = WalkForwardResult(
            config=WalkForwardConfig(),
            folds=[],
            aggregated={},
            fold_metrics=pl.DataFrame(),
        )
        summary = walk_forward_summary(result)
        assert "WALK-FORWARD" in summary


class TestDetectSignalCols:
    """Auto-detection of signal columns."""

    def test_detects_signal_prefix(self):
        df = pl.DataFrame({
            "signal_alpha": [0.1],
            "signal_beta": [-0.5],
            "return_1d": [0.01],
            "close": [100.0],
        })
        cols = _detect_signal_cols(df)
        assert "signal_alpha" in cols
        assert "signal_beta" in cols
        assert "return_1d" not in cols
        assert "close" not in cols

    def test_no_signals(self):
        df = pl.DataFrame({"close": [100.0]})
        cols = _detect_signal_cols(df)
        assert cols == []