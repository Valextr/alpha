"""Validation engine — data segmentation, walk-forward analysis, perturbation tests.

Phase 6 of the Alpha project. The core invariant: once a segment boundary
is defined, no Phase 4-5 code may look past the validation boundary into
hold-back data. The hold-back is the final arbiter of strategy viability.
"""

from .segmentation import (
    DataSegment,
    SegmentationConfig,
    SegmentationResult,
    get_default_segmentation,
    get_equal_segments,
    segment_dataframe,
    segment_query,
    add_segment_column,
    # Legacy / convenience functions
    split_train_validation_holdback,
    get_date_range,
    get_ticker_date_ranges,
    assert_segment_coverage,
)

from .walkforward import (
    WalkForwardConfig,
    WalkForwardResult,
    FoldResult,
    walk_forward,
    walk_forward_on_holdback,
    walk_forward_summary,
    _detect_signal_cols,
)

from .perturbation import (
    PerturbationResult,
    SignalPerturbationReport,
    PerturbationSummary,
    run_perturbation_test,
    run_full_perturbation_sweep,
    format_perturbation_report,
)

__all__ = [
    # Segmentation
    "DataSegment",
    "SegmentationConfig",
    "SegmentationResult",
    "get_default_segmentation",
    "get_equal_segments",
    "segment_dataframe",
    "segment_query",
    "add_segment_column",
    "split_train_validation_holdback",
    "get_date_range",
    "get_ticker_date_ranges",
    "assert_segment_coverage",
    # Walk-forward
    "WalkForwardConfig",
    "WalkForwardResult",
    "FoldResult",
    "walk_forward",
    "walk_forward_on_holdback",
    "walk_forward_summary",
    "_detect_signal_cols",
    # Perturbation tests
    "PerturbationResult",
    "SignalPerturbationReport",
    "PerturbationSummary",
    "run_perturbation_test",
    "run_full_perturbation_sweep",
    "format_perturbation_report",
]