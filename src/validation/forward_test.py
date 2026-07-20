"""Phase 8: Forward test runner.

Runs the signal pipeline on the hold-back segment (2023-01-01 onwards)
and measures predictive performance in a strict out-of-sample setting.

No parameter tuning on hold-back data — just validation that signals
retain predictive power outside the training/validation windows.

Segments (from triage plan):
    Train:       2014-01-01 to 2020-01-01  (60%)
    Validation:  2020-01-01 to 2023-01-01  (20%)
    Hold-back:   2023-01-01 onwards         (20%)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from src.data.config import get_config
from src.features.pipeline import compute_features
from src.signals.pipeline import generate_all_with_forward_returns
from src.signals.base import (
    compute_forward_returns,
    rank_ic,
    ic_decay,
    win_rate,
    signal_summary,
)


# ── Hold-back cutoff from triage plan ────────────────────────────────
HOLD_BACK_START = date(2023, 1, 1)
TRAIN_END = date(2020, 1, 1)
VALIDATION_END = date(2023, 1, 1)

# Horizons to evaluate
EVALUATION_HORIZONS = [1, 5, 21]


@dataclass
class SignalForwardResult:
    """Forward test result for one signal."""
    signal_name: str
    # Basic stats
    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    # IC metrics
    ic_by_horizon: dict[int, float] = field(default_factory=dict)
    # Win rate
    win_rate_by_horizon: dict[int, float] = field(default_factory=dict)
    # Per-ticker breakdown
    per_ticker_ic: dict[str, dict[int, float]] = field(default_factory=dict)


@dataclass
class ForwardTestResult:
    """Full forward test report."""
    hold_back_start: str = ""
    actual_date_range: str = ""
    tickers: list[str] = field(default_factory=list)
    total_rows: int = 0
    signals_evaluated: list[str] = field(default_factory=list)
    signal_results: list[SignalForwardResult] = field(default_factory=list)
    # Comparison with training period
    comparison: dict[str, Any] = field(default_factory=dict)
    summary: str = ""


def load_gold_data(
    data_dir: Path,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pl.DataFrame:
    """Load gold layer data, optionally filtered by date range.

    Args:
        data_dir: Root data directory (contains gold/daily).
        start_date: Inclusive start date filter (None = no lower bound).
        end_date: Exclusive end date filter (None = no upper bound).

    Returns:
        DataFrame sorted by (ticker, date) with gold data.
    """
    gold_dir = data_dir / "gold" / "daily"
    if not gold_dir.exists():
        raise FileNotFoundError(f"No gold data directory at {gold_dir}")

    # Find all parquet files
    files = list(gold_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {gold_dir}")

    frames = [pl.read_parquet(str(f)) for f in sorted(files)]
    df = pl.concat(frames, how="vertical_relaxed").sort(["ticker", "date"])

    if start_date:
        df = df.filter(pl.col("date") >= start_date)
    if end_date:
        df = df.filter(pl.col("date") < end_date)

    return df


def run_forward_test(
    df: pl.DataFrame,
    horizons: list[int] | None = None,
    categories: list[str] | None = None,
) -> list[SignalForwardResult]:
    """Run the signal pipeline on a DataFrame and evaluate predictive power.

    Args:
        df: Gold data sorted by (ticker, date).
        horizons: Forward horizons in trading days.
        categories: Signal categories to evaluate (None = all).

    Returns:
        List of SignalForwardResult, one per signal.
    """
    if horizons is None:
        horizons = EVALUATION_HORIZONS

    # Compute features
    enriched = compute_features(df)

    # Generate signals + forward returns
    signals = generate_all_with_forward_returns(
        enriched,
        categories=categories,
        horizons=horizons,
    )

    # Identify signal columns
    signal_cols = [c for c in signals.columns if c.startswith("signal_")]

    results = []
    for sig_col in signal_cols:
        sig_name = sig_col  # e.g. signal_mean_reversion_21d
        result = SignalForwardResult(signal_name=sig_name)

        # Basic stats
        col = signals[sig_col].drop_nulls()
        result.count = len(col)
        result.mean = float(col.mean()) if len(col) > 0 else 0.0
        result.std = float(col.std()) if len(col) > 0 else 0.0

        # IC by horizon
        for h in horizons:
            target = f"forward_return_{h}"
            if target in signals.columns:
                result.ic_by_horizon[h] = rank_ic(sig_col, target, signals)
                result.win_rate_by_horizon[h] = win_rate(sig_col, target, signals)

        # Per-ticker IC
        tickers = signals["ticker"].unique().to_list()
        for ticker in tickers:
            ticker_df = signals.filter(pl.col("ticker") == ticker)
            ticker_ics = {}
            for h in horizons:
                target = f"forward_return_{h}"
                if target in ticker_df.columns:
                    ticker_ics[h] = rank_ic(sig_col, target, ticker_df)
            result.per_ticker_ic[ticker] = ticker_ics

        results.append(result)

    return results


def run_comparison_test(
    data_dir: Path,
    train_start: date | None = None,
    train_end: date = TRAIN_END,
    hold_back_start: date = HOLD_BACK_START,
    horizons: list[int] | None = None,
) -> dict[str, Any]:
    """Compare signal performance between training and hold-back periods.

    The key question: does the forward-test IC drop significantly below
    the training-period IC? A large drop suggests overfitting.

    Args:
        data_dir: Root data directory.
        train_start: Start of training period (None = earliest data).
        train_end: End of training period (default: 2020-01-01).
        hold_back_start: Start of hold-back period (default: 2023-01-01).
        horizons: Forward horizons.

    Returns:
        Dict mapping signal_name -> {train_ic, holdback_ic, ic_change, ic_retention_pct}.
    """
    if horizons is None:
        horizons = EVALUATION_HORIZONS

    # Load training data
    train_df = load_gold_data(data_dir, start_date=train_start, end_date=train_end)
    train_results = run_forward_test(train_df, horizons=horizons)

    # Load hold-back data
    hb_df = load_gold_data(data_dir, start_date=hold_back_start)
    hb_results = run_forward_test(hb_df, horizons=horizons)

    comparison = {}
    for tr, hr in zip(train_results, hb_results):
        sig = tr.signal_name
        comp = {"signal": sig, "training_period": {}, "holdback_period": {}}
        for h in horizons:
            train_ic = tr.ic_by_horizon.get(h, 0.0)
            hb_ic = hr.ic_by_horizon.get(h, 0.0)
            ic_change = hb_ic - train_ic
            retention = (abs(hb_ic) / abs(train_ic) * 100) if train_ic != 0 else 0.0
            comp["training_period"][h] = {
                "ic": round(train_ic, 4),
                "win_rate": round(tr.win_rate_by_horizon.get(h, 0.0), 4),
            }
            comp["holdback_period"][h] = {
                "ic": round(hb_ic, 4),
                "win_rate": round(hr.win_rate_by_horizon.get(h, 0.0), 4),
            }
            comp[f"ic_change_{h}d"] = round(ic_change, 4)
            comp[f"ic_retention_{h}d_pct"] = round(retention, 1)
        comparison[sig] = comp

    return comparison


def generate_report(
    results: list[SignalForwardResult],
    tickers: list[str],
    date_range: str,
    comparison: dict[str, Any] | None = None,
) -> ForwardTestResult:
    """Generate a structured forward test report.

    Args:
        results: List of SignalForwardResult from run_forward_test.
        tickers: List of tickers in the test.
        date_range: Human-readable date range string.
        comparison: Optional comparison dict from run_comparison_test.

    Returns:
        ForwardTestResult with full report.
    """
    report = ForwardTestResult(
        hold_back_start=str(HOLD_BACK_START),
        actual_date_range=date_range,
        tickers=tickers,
        total_rows=results[0].count if results else 0,
        signals_evaluated=[r.signal_name for r in results],
        signal_results=results,
        comparison=comparison or {},
    )

    # Generate summary
    lines = []
    lines.append(f"Forward Test Report — {date_range}")
    lines.append(f"Hold-back start: {HOLD_BACK_START}")
    lines.append(f"Tickers: {', '.join(tickers)}")
    lines.append("")

    for r in results:
        lines.append(f"--- {r.signal_name} ---")
        lines.append(f"  Observations: {r.count}")
        lines.append(f"  Signal mean: {r.mean:.4f}, std: {r.std:.4f}")
        lines.append(f"  IC by horizon:")
        for h in sorted(r.ic_by_horizon.keys()):
            ic = r.ic_by_horizon[h]
            wr = r.win_rate_by_horizon.get(h, 0.0)
            lines.append(f"    {h}d: IC={ic:+.4f}, Win Rate={wr:.1%}")
        lines.append(f"  Per-ticker IC (1d):")
        for ticker in sorted(r.per_ticker_ic.keys()):
            ticker_ics = r.per_ticker_ic[ticker]
            ic_1d = ticker_ics.get(1, 0.0)
            lines.append(f"    {ticker}: {ic_1d:+.4f}")
        lines.append("")

    report.summary = "\n".join(lines)
    return report


def print_report(report: ForwardTestResult) -> None:
    """Print the forward test report to stdout."""
    print(report.summary)
    print("=" * 70)

    if report.comparison:
        print("\nCOMPARISON: Training vs Hold-back IC")
        print("-" * 70)
        for sig, comp in report.comparison.items():
            print(f"\n{sig}:")
            for h in [1, 5, 21]:
                train_ic = comp.get("training_period", {}).get(h, {}).get("ic", 0.0)
                hb_ic = comp.get("holdback_period", {}).get(h, {}).get("ic", 0.0)
                retention = comp.get(f"ic_retention_{h}d_pct", 0.0)
                print(f"  {h}d: Train IC={train_ic:+.4f} → Hold-back IC={hb_ic:+.4f} (retention: {retention:.0f}%)")


def export_report(
    report: ForwardTestResult,
    output_path: Path,
) -> None:
    """Export the report as JSON.

    Args:
        report: ForwardTestResult to export.
        output_path: Path to write JSON report.
    """
    # Convert to serializable dict
    data = {
        "hold_back_start": report.hold_back_start,
        "actual_date_range": report.actual_date_range,
        "tickers": report.tickers,
        "total_rows": report.total_rows,
        "signals_evaluated": report.signals_evaluated,
        "signal_results": [asdict(r) for r in report.signal_results],
        "comparison": report.comparison,
        "summary_text": report.summary,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def main() -> None:
    """CLI entry point: run the forward test on hold-back data."""
    import argparse

    parser = argparse.ArgumentParser(description="Phase 8: Forward test runner")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Root data directory (default: from config)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/forward_test_holdback.json"),
        help="Output JSON report path",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare training vs hold-back IC",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=None,
        help="Forward horizons in trading days (default: 1 5 21)",
    )
    args = parser.parse_args()

    config = get_config()
    data_dir = args.data_dir or config.data_dir

    # Load hold-back data
    print(f"Loading hold-back data from {HOLD_BACK_START}...")
    df = load_gold_data(data_dir, start_date=HOLD_BACK_START)
    tickers = df["ticker"].unique().to_list()
    min_date = df["date"].min()
    max_date = df["date"].max()
    date_range = f"{min_date} to {max_date} ({len(df)} rows, {len(tickers)} tickers)"
    print(f"  Date range: {date_range}")

    # Run forward test
    print("Computing features...")
    print("Generating signals + forward returns...")
    horizons = args.horizons or EVALUATION_HORIZONS
    results = run_forward_test(df, horizons=horizons)

    # Generate report
    comparison = None
    if args.compare:
        print("Running comparison test (training vs hold-back)...")
        comparison = run_comparison_test(data_dir, horizons=horizons)

    report = generate_report(results, tickers, date_range, comparison)
    print_report(report)

    # Export
    export_report(report, args.output)
    print(f"\nReport exported to {args.output}")


if __name__ == "__main__":
    main()