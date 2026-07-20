"""Momentum signal.

Hypothesis:
    Cross-sectional momentum: stocks that have outperformed peers over
    the past N days tend to continue outperforming in the near term.

    Based on Jegadeesh & Titman (1993) "Returns to Buying Winners and
    Selling Losers."

    Signal logic:
        - Compute forward-looking return over the formation period.
        - Rank each ticker's return relative to the cross-section.
        - High rank → positive signal (long), low rank → negative (short).

    Signal values are bounded in [-1, +1] via tanh normalization.

Parameters:
    formation_days: lookback window for computing momentum (default 21)
    skip_days: days to skip at start of formation period to avoid
               short-term reversal effects (default 0 for daily data)
"""

from __future__ import annotations

import polars as pl

from .registry import registry


@registry.register(
    "momentum_21d",
    description="21-day cross-sectional momentum signal",
    category="momentum",
    parameters={"formation_days": 21},
    requires_features=["return_21d"],
)
def generate_momentum_21d(df: pl.DataFrame) -> pl.DataFrame:
    """Generate momentum signal using 21-day returns.

    Steps:
        1. Use pre-computed 21-day returns as the momentum measure.
        2. Cross-sectionally rank returns per date.
        3. Normalize to [-1, +1] via tanh.

    Args:
        df: Feature-enriched DataFrame sorted by (ticker, date).

    Returns:
        DataFrame with `signal_momentum_21d` column.
    """
    return _generate_momentum(df, return_col="return_21d", out_col="signal_momentum_21d")


@registry.register(
    "momentum_63d",
    description="63-day cross-sectional momentum signal",
    category="momentum",
    parameters={"formation_days": 63},
    requires_features=["return_63d"],
)
def generate_momentum_63d(df: pl.DataFrame) -> pl.DataFrame:
    """Generate momentum signal using 63-day returns.

    A longer formation period captures slower-trending momentum.
    """
    return _generate_momentum(df, return_col="return_63d", out_col="signal_momentum_63d")


def _generate_momentum(
    df: pl.DataFrame,
    return_col: str,
    out_col: str,
) -> pl.DataFrame:
    """Core momentum signal generation.

    Uses cross-sectional ranking of past returns to generate
    directional signals normalized to [-1, +1].
    """

    # Cross-sectional rank of returns per date (0 to 1)
    rank = (
        pl.col(return_col).rank("average").over("date")
        / pl.col(return_col).count().over("date")
    )

    # Convert rank to [-1, +1]: rank in (0, 0.5) → negative, (0.5, 1) → positive
    df = df.with_columns(
        ((rank - 0.5) * 2).tanh().alias(out_col)
    )

    return df