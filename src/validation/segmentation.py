"""Data segmentation — train/validation/hold-back splits.

Splitting rules:
  - Chronological only (no random shuffling — time series cannot be reshuffled).
  - No overlap between segments. A date belongs to exactly one segment.
  - Boundaries are exclusive on the right: [start, end).
  - Hold-back cutoff is the most critical boundary. Once set, Phase 4-5
    training must never peek past it.

The default segmentation follows the triage plan:
  - Train:       2014-01-02 .. 2020-01-02   (~48%, pre-COVID)
  - Validation:  2020-01-02 .. 2023-01-03   (~24%, COVID through inflation)
  - Hold-back:   2023-01-03 .. present       (~28%, AI boom, current regime)

These boundaries prioritize market regimes over exact percentages.
"60/20/20" is the aspirational target; regime-aware boundaries are
the actual constraint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Optional

import duckdb
import polars as pl


class DataSegment(Enum):
    """Which segment a row belongs to."""

    TRAIN = "train"
    VALIDATION = "validation"
    HOLD_BACK = "hold_back"


@dataclass(frozen=True)
class SegmentationConfig:
    """Immutable configuration for a train/validation/hold-back split.

    Attributes:
        train_start:     First date included in training (inclusive).
                         Defaults to the earliest date in the data.
        train_end:       Last training date (exclusive).
        validation_start: First validation date (inclusive).
                         Defaults to train_end.
        validation_end:   Last validation date (exclusive).
        hold_back_start:  First hold-back date (inclusive).
                         Defaults to validation_end.
        description:      Human-readable label for this segmentation.
    """

    train_start: date | None = None
    train_end: date = date(2020, 1, 2)
    validation_start: date | None = None
    validation_end: date = date(2023, 1, 3)
    hold_back_start: date | None = None
    description: str = ""

    def __post_init__(self) -> None:
        # Fill in defaults for contiguity: start of each segment =
        # end of previous segment
        ts = self.train_start
        te = self.train_end
        vs = self.validation_start if self.validation_start else te
        ve = self.validation_end
        hs = self.hold_back_start if self.hold_back_start else ve

        # Validate ordering (train_start may be None — resolved at split time)
        if ts is not None:
            if not (ts < te == vs < ve == hs):
                raise ValueError(
                    f"Segment boundaries must be ordered and contiguous. "
                    f"Got: train=[{ts}, {te}), val=[{vs}, {ve}), "
                    f"hold_back=[{hs}, +inf)"
                )
        else:
            if not (te == vs < ve == hs):
                raise ValueError(
                    f"Segment boundaries must be ordered and contiguous. "
                    f"Got: train=[None, {te}), val=[{vs}, {ve}), "
                    f"hold_back=[{hs}, +inf)"
                )


@dataclass(frozen=True)
class SegmentationResult:
    """Output of a segmentation run — the three DataFrames and metadata."""

    config: SegmentationConfig
    train: pl.DataFrame
    validation: pl.DataFrame
    hold_back: pl.DataFrame

    # Convenience: combined DataFrame with a `segment` column
    combined: Optional[pl.DataFrame] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.combined is None:
            _tag = lambda df, seg: df.with_columns(pl.lit(seg).alias("segment"))
            combined = pl.concat([
                _tag(self.train, DataSegment.TRAIN.value),
                _tag(self.validation, DataSegment.VALIDATION.value),
                _tag(self.hold_back, DataSegment.HOLD_BACK.value),
            ])
            object.__setattr__(self, "combined", combined)

    @property
    def train_dates(self) -> list[date]:
        return sorted(self.train.get_column("date").to_list())

    @property
    def validation_dates(self) -> list[date]:
        return sorted(self.validation.get_column("date").to_list())

    @property
    def hold_back_dates(self) -> list[date]:
        return sorted(self.hold_back.get_column("date").to_list())

    def stats(self) -> dict:
        """Return a summary dict of the segmentation."""
        total = len(self.train) + len(self.validation) + len(self.hold_back)
        ts = self.config.train_start or (self.train.get_column("date").min() if self.train.height else None)
        vs = self.config.validation_start or self.config.train_end
        hs = self.config.hold_back_start or self.config.validation_end
        return {
            "description": self.config.description,
            "train": {
                "rows": len(self.train),
                "pct": len(self.train) / total * 100 if total else 0,
                "start": ts.isoformat() if ts else "N/A",
                "end": self.config.train_end.isoformat(),
            },
            "validation": {
                "rows": len(self.validation),
                "pct": len(self.validation) / total * 100 if total else 0,
                "start": vs.isoformat(),
                "end": self.config.validation_end.isoformat(),
            },
            "hold_back": {
                "rows": len(self.hold_back),
                "pct": len(self.hold_back) / total * 100 if total else 0,
                "start": hs.isoformat(),
                "end": self._hold_back_end(),
            },
            "total_rows": total,
            "tickers": sorted(self.train.get_column("ticker").unique().to_list()) if self.train.height and "ticker" in self.train.columns else [],
        }

    def _hold_back_end(self) -> str:
        if self.hold_back.height == 0:
            return "N/A"
        return self.hold_back.get_column("date").max().isoformat()

    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"SegmentationResult("
            f"train={stats['train']['rows']} ({stats['train']['pct']:.1f}%), "
            f"val={stats['validation']['rows']} ({stats['validation']['pct']:.1f}%), "
            f"hold_back={stats['hold_back']['rows']} ({stats['hold_back']['pct']:.1f}%))"
        )


def get_default_segmentation() -> SegmentationConfig:
    """Return the default segmentation from the triage plan.

    Boundaries:
        Train:       2014-01-02 .. 2020-01-02  (pre-COVID, steady markets)
        Validation:  2020-01-02 .. 2023-01-03  (COVID, recovery, inflation)
        Hold-back:   2023-01-03 .. present      (AI boom, rate cuts, current)

    Note: boundaries align with actual trading days. 2020-01-01 and
    2023-01-01 fall on holidays/weekends, so the exclusive boundaries
    land on the first trading day of each year.
    """
    return SegmentationConfig(
        train_start=date(2014, 1, 2),
        train_end=date(2020, 1, 2),
        validation_start=date(2020, 1, 2),
        validation_end=date(2023, 1, 3),
        hold_back_start=date(2023, 1, 3),
        description="Triage plan: regime-aware split (2020 / 2023 boundaries)",
    )


def get_equal_segments(df: pl.DataFrame) -> SegmentationConfig:
    """Compute approximately equal 60/20/20 segments from actual data.

    Unlike the default regime-aware split, this divides the available
    dates purely by count. Useful for prototyping when regime boundaries
    matter less.

    Args:
        df: DataFrame with a 'date' column.

    Returns:
        SegmentationConfig for a ~60/20/20 split.
    """
    dates = sorted(df.get_column("date").unique().to_list())
    n = len(dates)
    train_end_idx = int(n * 0.60)
    val_end_idx = int(n * 0.80)

    return SegmentationConfig(
        train_start=dates[0],
        train_end=dates[train_end_idx],
        validation_start=dates[train_end_idx],
        validation_end=dates[val_end_idx],
        hold_back_start=dates[val_end_idx],
        description=f"Equal-count split: ~60/20/20 ({n} total dates)",
    )


def _resolve_config(
    df: pl.DataFrame,
    config: Optional[SegmentationConfig],
) -> SegmentationConfig:
    """Resolve a SegmentationConfig against actual data bounds.

    - If config is None, uses the triage-plan defaults.
    - If train_start is None, resolves to the earliest date in the data.
    - If the data range is shorter than the calendar boundaries, falls
      back to proportional splits so tests with synthetic data still work.
    """
    if config is None:
        config = SegmentationConfig()

    if "date" not in df.columns or df.is_empty():
        return config

    data_min = df.get_column("date").min()
    data_max = df.get_column("date").max()

    # Normalize to date (DuckDB fetch_df returns datetime.datetime)
    if hasattr(data_min, 'date'):
        data_min = data_min.date()
    if hasattr(data_max, 'date'):
        data_max = data_max.date()

    # If train_start is None, resolve to data min
    if config.train_start is None:
        config = SegmentationConfig(
            train_start=data_min,
            train_end=config.train_end,
            validation_start=config.train_end,
            validation_end=config.validation_end,
            hold_back_start=config.validation_end,
            description=config.description,
        )

    # If the data doesn't span past the boundaries, fall back to
    # proportional splits so that synthetic/test data still segments
    if data_max < config.hold_back_start:
        dates = sorted(df.get_column("date").unique().to_list())
        n = len(dates)
        train_end_idx = int(n * 0.60)
        val_end_idx = int(n * 0.80)
        return SegmentationConfig(
            train_start=dates[0],
            train_end=dates[train_end_idx],
            validation_start=dates[train_end_idx],
            validation_end=dates[val_end_idx],
            hold_back_start=dates[val_end_idx],
            description=f"Proportional fallback (data={data_min}:{data_max}): ~60/20/20",
        )

    return config


def segment_dataframe(
    df: pl.DataFrame,
    config: Optional[SegmentationConfig] = None,
) -> SegmentationResult:
    """Split a DataFrame into train/validation/hold-back segments.

    Args:
        df: DataFrame with at least 'date' and 'ticker' columns,
            sorted by date.
        config: Segmentation boundaries. Defaults to the triage plan
            configuration (falls back to proportional if data is short).

    Returns:
        SegmentationResult with train, validation, and hold_back DataFrames.

    Raises:
        ValueError: If the DataFrame lacks a 'date' column.
    """
    if "date" not in df.columns:
        raise ValueError("DataFrame must have a 'date' column for segmentation")

    if df.is_empty():
        resolved = config if config is not None else SegmentationConfig()
        empty_df = pl.DataFrame(schema=df.schema)
        return SegmentationResult(
            config=resolved,
            train=empty_df,
            validation=empty_df,
            hold_back=empty_df,
        )

    resolved = _resolve_config(df, config)

    train_start = resolved.train_start or df.get_column("date").min()
    validation_start = resolved.validation_start or resolved.train_end

    mask_train = (pl.col("date") >= pl.lit(train_start)) & (
        pl.col("date") < pl.lit(resolved.train_end)
    )
    mask_val = (pl.col("date") >= pl.lit(validation_start)) & (
        pl.col("date") < pl.lit(resolved.validation_end)
    )
    mask_hold = pl.col("date") >= pl.lit(resolved.hold_back_start)

    train = df.filter(mask_train).sort("date")
    validation = df.filter(mask_val).sort("date")
    hold_back = df.filter(mask_hold).sort("date")

    return SegmentationResult(
        config=resolved,
        train=train,
        validation=validation,
        hold_back=hold_back,
    )


def segment_query(
    duckdb_file: Path | str,
    table: str = "gold_daily",
    config: Optional[SegmentationConfig] = None,
) -> SegmentationResult:
    """Segment data directly from DuckDB.

    Args:
        duckdb_file: Path to the DuckDB database.
        table: Table or view name to query (default: "gold_daily").
        config: Segmentation boundaries. Defaults to the triage plan.

    Returns:
        SegmentationResult with train, validation, and hold-back DataFrames.
    """
    if config is None:
        config = get_default_segmentation()

    conn = duckdb.connect(str(duckdb_file))
    try:
        df = conn.execute(f"SELECT * FROM {table} ORDER BY date, ticker").fetch_df()
    except Exception:
        raise ValueError(f"Table '{table}' not found in {duckdb_file}")
    finally:
        conn.close()

    return segment_dataframe(pl.from_pandas(df), config=config)


def add_segment_column(
    df: pl.DataFrame,
    config: Optional[SegmentationConfig] = None,
) -> pl.DataFrame:
    """Add a 'segment' column to an existing DataFrame.

    Useful for downstream code that needs to filter by segment
    without creating three separate DataFrames.

    Args:
        df: DataFrame with a 'date' column.
        config: Segmentation boundaries. Defaults to the triage plan.

    Returns:
        DataFrame with an appended 'segment' column (str).
    """
    if df.is_empty():
        return df.with_columns(pl.lit(DataSegment.TRAIN.value).alias("segment"))

    resolved = _resolve_config(df, config)
    train_start = resolved.train_start or df.get_column("date").min()
    validation_start = resolved.validation_start or resolved.train_end

    return df.with_columns(
        pl.when(pl.col("date") < pl.lit(resolved.train_end))
        .then(pl.lit(DataSegment.TRAIN.value))
        .when(pl.col("date") < pl.lit(resolved.validation_end))
        .then(pl.lit(DataSegment.VALIDATION.value))
        .otherwise(pl.lit(DataSegment.HOLD_BACK.value))
        .alias("segment"),
    )


# ── Convenience aliases / helpers (legacy API compatibility) ─────────


def split_train_validation_holdback(
    df: pl.DataFrame,
    config: SegmentationConfig | None = None,
) -> dict[str, pl.DataFrame]:
    """Convenience: split into a dict of DataFrames.

    Equivalent to calling ``segment_dataframe()`` but returns a flat
    dict mapping ``{"train", "validation", "holdback"}`` to DataFrames.

    Args:
        df: DataFrame with 'date' and 'ticker' columns.
        config: Segmentation boundaries (defaults to triage plan).

    Returns:
        Dict with ``train``, ``validation``, and ``holdback`` keys.
    """
    seg = segment_dataframe(df, config)
    return {
        "train": seg.train,
        "validation": seg.validation,
        "holdback": seg.hold_back,
    }


def get_date_range(df: pl.DataFrame) -> tuple[date, date]:
    """Return (min_date, max_date) from a DataFrame's 'date' column."""
    if "date" not in df.columns:
        raise ValueError("DataFrame must have a 'date' column")
    return (df["date"].min(), df["date"].max())


def assert_segment_coverage(
    segments: dict[str, pl.DataFrame],
    min_fraction: float = 0.05,
) -> dict[str, float]:
    """Validate that each segment has a reasonable share of total data.

    Args:
        segments: Dict with ``train``, ``validation``, ``holdback`` keys.
        min_fraction: Minimum fraction of total rows each segment should have.

    Returns:
        Dict mapping segment name to its fraction of total rows.

    Raises:
        ValueError: If any segment has fewer rows than ``min_fraction`` of total.
    """
    total = sum(len(seg) for seg in segments.values())
    if total == 0:
        raise ValueError("Total segment size is 0")

    fractions: dict[str, float] = {}
    for name, seg in segments.items():
        frac = len(seg) / total
        fractions[name] = frac
        if frac < min_fraction:
            raise ValueError(
                f"Segment '{name}' is too small: {frac:.1%} of total "
                f"({len(seg)}/{total} rows), minimum is {min_fraction:.0%}"
            )
    return fractions


def get_ticker_date_ranges(
    df: pl.DataFrame,
) -> dict[str, tuple[date, date]]:
    """Return per-ticker date ranges.

    Returns:
        Dict mapping ticker -> (min_date, max_date).
    """
    if "date" not in df.columns or "ticker" not in df.columns:
        raise ValueError("DataFrame must have 'date' and 'ticker' columns")

    ranges: dict[str, tuple[date, date]] = {}
    for ticker in df["ticker"].unique().to_list():
        ticker_df = df.filter(pl.col("ticker") == ticker)
        ranges[ticker] = (ticker_df["date"].min(), ticker_df["date"].max())
    return ranges