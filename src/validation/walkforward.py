"""Walk-forward analysis — rolling train/eval windows.

Validates signal quality using out-of-sample rolling windows instead
of a single static split. Each fold trains on a rolling window then
evaluates on the immediately following period, preventing look-ahead bias.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import polars as pl


@dataclass(frozen=True)
class FoldResult:
    """Result for a single walk-forward fold."""

    fold_index: int
    train_start: date
    train_end: date
    eval_start: date
    eval_end: date
    train_rows: int = 0
    eval_rows: int = 0
    signal_columns: list[str] = field(default_factory=list)
    metrics: dict[str, dict] = field(default_factory=dict)


@dataclass(frozen=True)
class WalkForwardConfig:
    """Configuration for walk-forward analysis.

    Attributes:
        train_window_days: Training window size in trading days (default: ~2 years).
        eval_window_days: Evaluation window size in trading days (default: ~3 months).
        step_days: Step size between folds in trading days (default: ~1 month).
    """

    train_window_days: int = 504  # ~2 years
    eval_window_days: int = 63  # ~3 months
    step_days: int = 21  # ~1 month


@dataclass(frozen=True)
class WalkForwardResult:
    """Result of a walk-forward analysis run."""

    config: WalkForwardConfig
    folds: list[FoldResult]
    aggregated: dict[str, dict[int, dict]]
    fold_metrics: pl.DataFrame


def _detect_signal_cols(df: pl.DataFrame) -> list[str]:
    """Auto-detect signal columns by 'signal_' prefix."""
    return [col for col in df.columns if col.startswith("signal_")]


def _compute_forward_returns(
    df: pl.DataFrame, horizon: int
) -> pl.DataFrame:
    """Compute forward returns at a given horizon."""
    return df.with_columns(
        pl.col("close")
        .shift(-horizon)
        .over("ticker")
        .alias(f"forward_return_{horizon}")
    )


def _compute_ic(
    signal_col: str, forward_col: str
) -> float:
    """Compute rank IC (Spearman correlation) between signal and forward return."""
    return 0.0  # placeholder


def _generate_folds(
    dates: list[date], config: WalkForwardConfig
) -> list[tuple[int, slice, slice]]:
    """Generate fold boundaries from a sorted list of unique dates."""
    n = len(dates)
    folds = []
    idx = 0

    while True:
        train_end_idx = idx + config.train_window_days
        eval_end_idx = train_end_idx + config.eval_window_days

        if eval_end_idx > n:
            break

        fold = (
            len(folds),
            slice(idx, train_end_idx),
            slice(train_end_idx, eval_end_idx),
        )
        folds.append(fold)
        idx += config.step_days

    return folds


def _compute_fold_metrics(
    fold: FoldResult,
    eval_df: pl.DataFrame,
    signal_cols: list[str],
    horizons: list[int] = None,
) -> dict[str, dict[int, dict]]:
    """Compute IC and related metrics for one fold."""
    if horizons is None:
        horizons = [1, 5, 21]

    metrics: dict[str, dict[int, dict]] = {}
    rows: list[dict] = []

    for sig in signal_cols:
        metrics[sig] = {}
        for h in horizons:
            # Forward return
            fwd = eval_df.with_columns(
                pl.col("close")
                .shift(-h)
                .over("ticker")
                .alias(f"_fwd_{h}")
            )
            valid = fwd.drop_nulls(subset=[sig, f"_fwd_{h}"])

            if valid.height == 0:
                metrics[sig][h] = {
                    "fold_count": 0,
                    "mean_ic": 0.0,
                    "std_ic": 0.0,
                    "ic_cv": None,
                    "mean_win_rate": None,
                }
                continue

            # Rank IC (Spearman)
            sig_ranked = valid[sig].to_frame().with_columns(
                pl.col(sig).rank().alias("_rank")
            )["_rank"].to_list()
            fwd_ranked = valid[f"_fwd_{h}"].to_frame().with_columns(
                pl.col(f"_fwd_{h}").rank().alias("_rank")
            )["_rank"].to_list()

            n_pts = len(sig_ranked)
            mean_s = sum(sig_ranked) / n_pts
            mean_f = sum(fwd_ranked) / n_pts
            cov = sum(
                (s - mean_s) * (f - mean_f)
                for s, f in zip(sig_ranked, fwd_ranked)
            ) / n_pts
            std_s = math.sqrt(
                sum((s - mean_s) ** 2 for s in sig_ranked) / n_pts
            )
            std_f = math.sqrt(
                sum((f - mean_f) ** 2 for f in fwd_ranked) / n_pts
            )
            ic = cov / (std_s * std_f) if (std_s > 0 and std_f > 0) else 0.0

            # Win rate
            buys = valid.filter(pl.col(sig) > 0)
            if buys.height > 0:
                correct = buys.filter(pl.col(f"_fwd_{h}") > 0).height
                win_rate = correct / buys.height
            else:
                win_rate = 0.5

            metrics[sig][h] = {
                "fold_count": 1,
                "mean_ic": ic,
                "std_ic": abs(ic),
                "ic_cv": abs(ic) / abs(ic) if ic != 0 else None,
                "mean_win_rate": win_rate,
            }

            rows.append({
                "fold": fold.fold_index,
                "signal": sig,
                "horizon": h,
                "ic": ic,
                "win_rate": win_rate,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "eval_start": fold.eval_start,
                "eval_end": fold.eval_end,
                "n_obs": valid.height,
            })

    return metrics, rows


def walk_forward(
    df: pl.DataFrame,
    config: WalkForwardConfig = None,
    horizons: list[int] = None,
) -> WalkForwardResult:
    """Run walk-forward analysis on a DataFrame with signal columns.

    Args:
        df: DataFrame with 'ticker', 'date', 'close' columns plus
            signal columns (prefixed 'signal_').
        config: Walk-forward configuration. Defaults to 2yr train / 3mo eval.
        horizons: Forward return horizons in trading days.

    Returns:
        WalkForwardResult with folds and aggregated metrics.

    Raises:
        ValueError: If no signal columns are found.
    """
    if config is None:
        config = WalkForwardConfig()
    if horizons is None:
        horizons = [1, 5, 21]

    signal_cols = _detect_signal_cols(df)
    if not signal_cols:
        raise ValueError(
            "No signal columns found (expected columns prefixed with 'signal_')"
        )

    dates = sorted(df.get_column("date").unique().to_list())
    folds = _generate_folds(dates, config)

    fold_results: list[FoldResult] = []
    all_rows: list[dict] = []
    # aggregated[signal][horizon] = {fold_count, mean_ic, std_ic, ic_cv, mean_win_rate}
    agg: dict[str, dict[int, list[float]]] = {
        sig: {h: [] for h in horizons} for sig in signal_cols
    }

    for fold_idx, train_slice, eval_slice in folds:
        train_dates = dates[train_slice]
        eval_dates = dates[eval_slice]

        train_mask = pl.col("date").is_in(train_dates)
        eval_mask = pl.col("date").is_in(eval_dates)

        train_df = df.filter(train_mask)
        eval_df = df.filter(eval_mask)

        fold = FoldResult(
            fold_index=fold_idx,
            train_start=train_dates[0],
            train_end=train_dates[-1],
            eval_start=eval_dates[0],
            eval_end=eval_dates[-1],
            train_rows=train_df.height,
            eval_rows=eval_df.height,
            signal_columns=signal_cols,
        )
        fold_results.append(fold)

        f_metrics, rows = _compute_fold_metrics(fold, eval_df, signal_cols, horizons)
        all_rows.extend(rows)

        for sig in signal_cols:
            for h in horizons:
                if sig in f_metrics and h in f_metrics[sig]:
                    m = f_metrics[sig][h]
                    if m["fold_count"] > 0:
                        agg[sig][h].append(m["mean_ic"])

    # Aggregate across folds
    aggregated: dict[str, dict[int, dict]] = {}
    for sig in signal_cols:
        aggregated[sig] = {}
        for h in horizons:
            ics = agg[sig][h]
            if ics:
                mean_ic = sum(ics) / len(ics)
                std_ic = math.sqrt(sum((x - mean_ic) ** 2 for x in ics) / len(ics)) if len(ics) > 1 else 0.0
                aggregated[sig][h] = {
                    "fold_count": len(ics),
                    "mean_ic": mean_ic,
                    "std_ic": std_ic,
                    "ic_cv": std_ic / abs(mean_ic) if mean_ic != 0 else None,
                    "mean_win_rate": None,  # not tracked in aggregate yet
                }
            else:
                aggregated[sig][h] = {
                    "fold_count": 0,
                    "mean_ic": 0.0,
                    "std_ic": 0.0,
                    "ic_cv": None,
                    "mean_win_rate": None,
                }

    fold_metrics_df = pl.DataFrame(all_rows) if all_rows else pl.DataFrame(
        schema={
            "fold": pl.Int64,
            "signal": pl.String,
            "horizon": pl.Int64,
            "ic": pl.Float64,
            "win_rate": pl.Float64,
            "train_start": pl.Date,
            "train_end": pl.Date,
            "eval_start": pl.Date,
            "eval_end": pl.Date,
            "n_obs": pl.Int64,
        }
    )

    return WalkForwardResult(
        config=config,
        folds=fold_results,
        aggregated=aggregated,
        fold_metrics=fold_metrics_df,
    )


def walk_forward_on_holdback(
    df: pl.DataFrame,
    config: WalkForwardConfig | None = None,
    horizons: list[int] | None = None,
) -> WalkForwardResult:
    """Run walk-forward restricted to the hold-back segment.

    Segments the data first, then runs walk-forward only on the
    hold-back period to get a true out-of-sample assessment.

    When the data spans past 2023, the hold-back is 2023+.
    When the data does not reach 2023 (e.g. synthetic/test data),
    the hold-back defaults to the last ~20% proportionally.

    Args:
        df: Full DataFrame with signal columns.
        config: Walk-forward configuration.
        horizons: Forward return horizons.

    Returns:
        WalkForwardResult restricted to hold-back data.

    Raises:
        ValueError: If the hold-back segment is empty.
    """
    from .segmentation import segment_dataframe, get_default_segmentation, get_equal_segments

    seg = segment_dataframe(df, get_default_segmentation())
    # If data doesn't reach 2023, fall back to proportional hold-back
    if seg.hold_back.height == 0:
        seg = segment_dataframe(df, get_equal_segments(df))

    if seg.hold_back.height == 0:
        raise ValueError(
            "Hold-back segment is empty — insufficient data for walk-forward."
        )

    # Adapt window sizes if hold-back is too small for default config
    effective_config = config
    if config is None:
        effective_config = WalkForwardConfig()

    hold_dates = sorted(seg.hold_back.get_column("date").unique().to_list())
    hold_len = len(hold_dates)

    if hold_len < effective_config.train_window_days + effective_config.eval_window_days:
        # Scale down: train = 60% of hold-back, eval = 20%, step = 10%
        train_w = max(int(hold_len * 0.60), 30)
        eval_w = max(int(hold_len * 0.20), 10)
        step = max(int(hold_len * 0.10), 5)
        effective_config = WalkForwardConfig(
            train_window_days=train_w,
            eval_window_days=eval_w,
            step_days=step,
        )

    return walk_forward(seg.hold_back, effective_config, horizons)


def walk_forward_summary(result: WalkForwardResult) -> str:
    """Generate a human-readable summary of walk-forward results.

    Args:
        result: WalkForwardResult to summarize.

    Returns:
        Multi-line string summary.
    """
    lines = [
        "WALK-FORWARD ANALYSIS SUMMARY",
        "=" * 40,
        f"Folds: {len(result.folds)}",
        f"Config: train={result.config.train_window_days}d, "
        f"eval={result.config.eval_window_days}d, "
        f"step={result.config.step_days}d",
        "",
    ]

    for sig, horizons in result.aggregated.items():
        lines.append(f"Signal: {sig}")
        lines.append("-" * 30)
        for h, metrics in sorted(horizons.items()):
            fc = metrics["fold_count"]
            if fc == 0:
                lines.append(f"  Horizon {h}d: no data")
                continue
            lines.append(
                f"  Horizon {h}d: folds={fc}, "
                f"IC={metrics['mean_ic']:.4f}, "
                f"IC_std={metrics['std_ic']:.4f}"
            )
            if metrics.get("mean_win_rate") is not None:
                lines.append(f"    win_rate={metrics['mean_win_rate']:.3f}")
        lines.append("")

    return "\n".join(lines)