"""Signal base interface and forward-return computation.

Every signal module implements `generate_<signal_name>(df)` which takes
a feature-enriched DataFrame and returns it with a `signal_<name>` column
appended. Signal values are directional:

    +1  →  long bias
     0  →  neutral
    -1  →  short bias

Signals should be bounded roughly in [-1, +1] for comparability, but
the only hard requirement is consistent sign convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import polars as pl


@dataclass(frozen=True)
class SignalMeta:
    """Metadata for a single signal."""

    name: str
    description: str
    category: str
    parameters: dict  # free-form config dict
    depends_on: list[str] = field(default_factory=list)
    # Feature columns required from the feature pipeline.
    # Signals can also depend on raw OHLCV columns implicitly.
    requires_features: list[str] = field(default_factory=list)


def compute_forward_returns(
    df: pl.DataFrame,
    horizons: Sequence[int] = (1, 5, 21),
    close_col: str = "close",
) -> pl.DataFrame:
    """Attach forward return columns for each horizon.

    Forward returns are computed per-ticker using shifted close prices,
    so no future leakage occurs at call time — the caller is responsible
    for ensuring the training split ends before the evaluation period.

    Args:
        df: DataFrame sorted by (ticker, date).
        horizons: Forward horizons in trading days.
        close_col: Column name for close price.

    Returns:
        DataFrame with `forward_return_<h>` columns appended.
    """
    out = df.clone()
    for h in horizons:
        future_close = pl.col(close_col).shift(-h).over("ticker")
        out = out.with_columns(
            ((future_close / pl.col(close_col)) - 1).alias(f"forward_return_{h}")
        )
    return out


def rank_ic(signal_col: str, target_col: str, df: pl.DataFrame) -> float:
    """Rank Information Coefficient (Spearman correlation).

    Computes the rank correlation between a signal and its target
    across all (ticker, date) observations.

    Args:
        signal_col: Signal column name.
        target_col: Target column name (e.g. forward_return_1).
        df: DataFrame containing both columns.

    Returns:
        Rank IC value in [-1, +1].
    """
    sub = df.select([signal_col, target_col]).drop_nulls()
    if len(sub) < 3:
        return 0.0
    # Rank both columns, then compute Pearson on ranks (= Spearman)
    ranked = sub.with_columns([
        pl.col(signal_col).rank("average").alias("_rank_s"),
        pl.col(target_col).rank("average").alias("_rank_t"),
    ])
    corr = ranked.select(
        pl.corr(pl.col("_rank_s"), pl.col("_rank_t"))
    )
    val = corr.item()
    return float(val) if val is not None else 0.0


def ic_decay(
    signal_col: str,
    df: pl.DataFrame,
    horizons: Sequence[int] = (1, 5, 21, 63),
) -> dict[int, float]:
    """IC at each forward horizon.

    Returns a dict mapping horizon days → rank IC.
    """
    result = {}
    for h in horizons:
        target = f"forward_return_{h}"
        if target in df.columns:
            result[h] = rank_ic(signal_col, target, df)
    return result


def win_rate(signal_col: str, target_col: str, df: pl.DataFrame) -> float:
    """Fraction of rows where signal and target agree in sign.

    A positive signal is a "win" when the forward return is positive,
    and a negative signal is a "win" when the forward return is negative.
    """
    sub = df.select([signal_col, target_col]).drop_nulls()
    if len(sub) == 0:
        return 0.0
    agree = (
        (pl.col(signal_col) > 0) & (pl.col(target_col) > 0)
    ) | (
        (pl.col(signal_col) < 0) & (pl.col(target_col) < 0)
    )
    count = sub.select(agree.sum()).item()
    return float(count) / len(sub)


def signal_summary(
    signal_col: str,
    df: pl.DataFrame,
    horizons: Sequence[int] | None = None,
) -> dict:
    """Produce a validation summary for a signal.

    Returns:
        Dict with IC, IC decay, win rate, and basic statistics.
    """
    summary: dict = {}

    # Basic stats
    col = df[signal_col].drop_nulls()
    summary["count"] = len(col)
    summary["mean"] = float(col.mean()) if len(col) > 0 else 0.0
    summary["std"] = float(col.std()) if len(col) > 0 else 0.0
    summary["min"] = float(col.min()) if len(col) > 0 else 0.0
    summary["max"] = float(col.max()) if len(col) > 0 else 0.0

    # IC against each available forward return
    if horizons is None:
        horizons = [
            int(h.split("_")[-1])
            for h in df.columns
            if h.startswith("forward_return_")
        ]
        horizons.sort()

    for h in horizons:
        target = f"forward_return_{h}"
        if target in df.columns:
            summary[f"ic_{h}d"] = rank_ic(signal_col, target, df)
            summary[f"win_rate_{h}d"] = win_rate(signal_col, target, df)

    return summary