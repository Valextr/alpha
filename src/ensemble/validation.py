"""Ensemble validation and IC analysis.

Validates that the IC-weighted ensemble outperforms individual signals
using multiple complementary metrics:

    * **Flat IC** — rank correlation across all (ticker, date) observations.
    * **Cross-sectional IC (CS IC)** — per-date rank correlation across
      tickers, aggregated as mean, std, and capture ratios.
    * **IC Decay** — CS IC at each forward horizon (1d, 5d, 21d, 63d).
    * **Win Rate** — fraction of observations where signal and return
      agree in sign.
    * **Weight Distribution** — how ensemble weights distribute across
      signals (detects single-signal dominance).
    * **Subsample Analysis** — IC in bull vs bear regimes.

Public API
----------
    validate_signals(df, signal_cols, target_col, horizons)
        Compute IC, win rate, and summary stats for each signal.

    validate_ensemble(df, ensemble_col, signal_cols, target_col, horizons)
        Compare ensemble IC to each individual signal IC.

    signal_summary(df, signal_col, target_col, horizons)
        Produce a per-signal validation report.

    weight_report(ensemble)
        Inspect the ICWeightedEnsemble weight schedule for dominance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import polars as pl

from src.signals.base import rank_ic, ic_decay, win_rate, signal_summary as _base_signal_summary
from src.ensemble.ic_weighted import (
    ICWeightedEnsemble,
    compute_cross_sectional_ic,
    compute_rolling_ic,
)


# ── Data classes for structured reports ─────────────────────────────


@dataclass
class SignalMetrics:
    """IC, win rate, and CS IC stats for one signal."""

    signal: str
    flat_ic: float
    win_rate: float
    cs_ic_mean: float
    cs_ic_std: float
    cs_ic_positive_pct: float  # fraction of dates with positive CS IC
    ic_decay: dict[int, float] = field(default_factory=dict)


@dataclass
class EnsembleMetrics:
    """Comparison metrics: ensemble vs individual signals."""

    ensemble: str
    ensemble_flat_ic: float
    ensemble_win_rate: float
    ensemble_cs_ic_mean: float
    ensemble_cs_ic_std: float
    ensemble_cs_ic_positive_pct: float
    ensemble_ic_decay: dict[int, float] = field(default_factory=dict)
    # Per-signal metrics for comparison
    signals: dict[str, SignalMetrics] = field(default_factory=dict)
    # Ensemble IC must exceed every individual signal's IC
    dominates_all: bool = False


# ── Core validation functions ────────────────────────────────────────


def compute_cs_ic_stats(
    df: pl.DataFrame,
    signal_col: str,
    target_col: str,
) -> tuple[float, float, float]:
    """Compute aggregated CS IC statistics.

    Returns
    -------
    (mean, std, positive_pct)
        *mean* — mean of daily CS IC values.
        *std* — standard deviation.
        *positive_pct* — fraction of dates where CS IC > 0.
    """
    if signal_col not in df.columns or target_col not in df.columns:
        return (0.0, 0.0, 0.0)

    cs_ic = compute_cross_sectional_ic(df, signal_col, target_col)
    if cs_ic.is_empty():
        return (0.0, 0.0, 0.0)

    mean_val = float(cs_ic["cs_ic"].mean())
    std_val = float(cs_ic["cs_ic"].std())
    pos = float((cs_ic["cs_ic"] > 0).sum()) / len(cs_ic)

    return (mean_val, std_val, pos)


def signal_metrics(
    df: pl.DataFrame,
    signal_col: str,
    target_col: str,
    horizons: Sequence[int] | None = None,
) -> SignalMetrics:
    """Compute all validation metrics for a single signal.

    Parameters
    ----------
    df : pl.DataFrame
        Must contain ``date``, *signal_col*, *target_col*, and forward
        return columns if *horizons* are specified.
    signal_col : str
        Signal column name.
    target_col : str
        Primary forward return column (e.g., ``"forward_return_1"``).
    horizons : Sequence[int] | None
        Forward horizons for IC decay analysis. Default ``[1, 5, 21]``.

    Returns
    -------
    SignalMetrics
        Structured metrics for this signal.
    """
    if horizons is None:
        horizons = [1, 5, 21]

    flat = rank_ic(signal_col, target_col, df)
    wr = win_rate(signal_col, target_col, df)
    cs_mean, cs_std, cs_pos = compute_cs_ic_stats(df, signal_col, target_col)

    decay = ic_decay(signal_col, df, horizons=horizons)

    return SignalMetrics(
        signal=signal_col,
        flat_ic=flat,
        win_rate=wr,
        cs_ic_mean=cs_mean,
        cs_ic_std=cs_std,
        cs_ic_positive_pct=cs_pos,
        ic_decay=decay,
    )


def validate_signals(
    df: pl.DataFrame,
    signal_cols: Sequence[str],
    target_col: str = "forward_return_1",
    horizons: Sequence[int] | None = None,
) -> list[SignalMetrics]:
    """Validate multiple signals against the same target.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing signal columns and forward returns.
    signal_cols : Sequence[str]
        Signal column names.
    target_col : str
        Primary target column.
    horizons : Sequence[int] | None
        Forward horizons for IC decay.

    Returns
    -------
    list[SignalMetrics]
        One metrics object per signal, sorted by descending flat IC.
    """
    metrics = [
        signal_metrics(df, sig, target_col, horizons)
        for sig in signal_cols
        if sig in df.columns
    ]
    return sorted(metrics, key=lambda m: abs(m.flat_ic), reverse=True)


def validate_ensemble(
    df: pl.DataFrame,
    ensemble_col: str,
    signal_cols: Sequence[str],
    target_col: str = "forward_return_1",
    horizons: Sequence[int] | None = None,
) -> EnsembleMetrics:
    """Validate the ensemble against individual signals.

    Computes metrics for the ensemble and each signal, then checks
    whether the ensemble's flat IC exceeds every individual signal's
    flat IC.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing the ensemble column, signal columns, and
        forward returns.
    ensemble_col : str
        Ensemble score column name (e.g., ``"ensemble_score"``).
    signal_cols : Sequence[str]
        Individual signal column names.
    target_col : str
        Primary target column.
    horizons : Sequence[int] | None
        Forward horizons for IC decay.

    Returns
    -------
    EnsembleMetrics
        Ensemble metrics with per-signal comparisons.
    """
    if horizons is None:
        horizons = [1, 5, 21]

    em = signal_metrics(df, ensemble_col, target_col, horizons)

    sig_metrics = {
        sig: signal_metrics(df, sig, target_col, horizons)
        for sig in signal_cols
        if sig in df.columns
    }

    # Check dominance: ensemble IC must exceed every individual signal
    dominates = all(
        abs(em.flat_ic) >= abs(sm.flat_ic)
        for sm in sig_metrics.values()
    )

    return EnsembleMetrics(
        ensemble=ensemble_col,
        ensemble_flat_ic=em.flat_ic,
        ensemble_win_rate=em.win_rate,
        ensemble_cs_ic_mean=em.cs_ic_mean,
        ensemble_cs_ic_std=em.cs_ic_std,
        ensemble_cs_ic_positive_pct=em.cs_ic_positive_pct,
        ensemble_ic_decay=em.ic_decay,
        signals=sig_metrics,
        dominates_all=dominates,
    )


# ── Weight analysis ─────────────────────────────────────────────────


@dataclass
class WeightStats:
    """Distribution statistics for one signal's weight."""
    signal: str
    mean: float
    std: float
    min: float
    max: float
    pct_zero: float  # fraction of dates with zero weight


def weight_report(
    ensemble: ICWeightedEnsemble,
) -> list[WeightStats]:
    """Inspect the weight schedule from an ICWeightedEnsemble.

    Returns one WeightStats per signal. High pct_zero (>0.5) or
    max close to 1.0 indicates potential single-signal dominance.

    Parameters
    ----------
    ensemble : ICWeightedEnsemble
        Must have called ``transform()`` first.

    Returns
    -------
    list[WeightStats]
        One stats object per signal.
    """
    if ensemble._weight_schedule is None:
        raise RuntimeError("Call ensemble.transform() before weight_report()")

    schedule = ensemble._weight_schedule
    stats: list[WeightStats] = []

    for sig in ensemble._signal_cols:
        wcol = f"w_{sig}"
        if wcol not in schedule.columns:
            continue

        col = schedule[wcol]
        stats.append(WeightStats(
            signal=sig,
            mean=float(col.mean()),
            std=float(col.std()),
            min=float(col.min()),
            max=float(col.max()),
            pct_zero=float((col == 0.0).sum()) / len(col),
        ))

    return stats


# ── Report formatting ────────────────────────────────────────────────


def format_metrics_table(
    metrics: list[SignalMetrics],
) -> str:
    """Return a formatted table of signal metrics.

    Suitable for printing or inclusion in a report.
    """
    header = (
        f"{'Signal':<30} "
        f"{'Flat IC':>8} "
        f"{'Win Rate':>9} "
        f"{'CS IC mean':>10} "
        f"{'CS IC std':>9} "
        f"{'CS IC +ve%':>9}"
    )
    rows = [header, "-" * len(header)]
    for m in metrics:
        rows.append(
            f"{m.signal:<30} "
            f"{m.flat_ic:>+8.4f} "
            f"{m.win_rate:>+9.4f} "
            f"{m.cs_ic_mean:>+10.4f} "
            f"{m.cs_ic_std:>+9.4f} "
            f"{m.cs_ic_positive_pct:>+9.2%}"
        )
    return "\n".join(rows)


def format_ensemble_report(
    em: EnsembleMetrics,
) -> str:
    """Return a formatted ensemble validation report."""
    lines: list[str] = []

    lines.append("=" * 70)
    lines.append("ENSEMBLE VALIDATION REPORT")
    lines.append("=" * 70)

    lines.append("")
    lines.append(f"Ensemble: {em.ensemble}")
    lines.append(f"  Flat IC:          {em.ensemble_flat_ic:+.4f}")
    lines.append(f"  Win Rate:         {em.ensemble_win_rate:.4f}")
    lines.append(f"  CS IC (mean):     {em.ensemble_cs_ic_mean:+.4f}")
    lines.append(f"  CS IC (std):      {em.ensemble_cs_ic_std:+.4f}")
    lines.append(f"  CS IC (+ve %):    {em.ensemble_cs_ic_positive_pct:.2%}")

    if em.ensemble_ic_decay:
        lines.append("")
        lines.append("  IC Decay (ensemble):")
        for h, ic in sorted(em.ensemble_ic_decay.items()):
            lines.append(f"    {h:>3}d: {ic:+.4f}")

    lines.append("")
    lines.append("-" * 70)
    lines.append("INDIVIDUAL SIGNALS")
    lines.append("-" * 70)

    lines.append("")
    lines.append(format_metrics_table(list(em.signals.values())))

    lines.append("")
    lines.append("-" * 70)
    lines.append("DOMINANCE CHECK")
    lines.append("-" * 70)

    if em.dominates_all:
        lines.append(f"  [PASS] Ensemble IC ({em.ensemble_flat_ic:+.4f}) exceeds all individual signals")
    else:
        lines.append(f"  [WARN] Ensemble IC ({em.ensemble_flat_ic:+.4f}) does NOT exceed all individual signals")
        for name, sm in em.signals.items():
            if abs(sm.flat_ic) > abs(em.ensemble_flat_ic):
                lines.append(f"    - {name}: {sm.flat_ic:+.4f} > ensemble")

    lines.append("")
    return "\n".join(lines)