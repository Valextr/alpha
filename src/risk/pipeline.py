"""Risk management pipeline.

Orchestrates all risk control layers in the correct order:
1. Hard constraints (position caps, sector limits, leverage)
2. Volatility targeting
3. Drawdown circuit breaker
4. Correlation spike detection

This is the single entry point: call enforce_all_constraints() to apply
the full risk stack.
"""

from __future__ import annotations

import polars as pl

from .config import RiskConfig
from .constraints import enforce_all_constraints as _enforce_hard_constraints
from .vol_targeting import volatility_target
from .drawdown import compute_drawdown_series
from .correlation import correlation_spike_detector


def enforce_all_constraints(
    weights_df: pl.DataFrame,
    config: RiskConfig,
    *,
    returns_df: pl.DataFrame | None = None,
    equity_curve: pl.DataFrame | None = None,
    weight_col: str = "weight",
    sector_col: str = "sector",
    return_col: str = "return",
    equity_col: str = "equity",
) -> pl.DataFrame:
    """Apply the full risk management pipeline.

    Pipeline order:
    1. Hard constraints (position, sector, leverage caps)
    2. Volatility targeting (if returns_df is provided)
    3. Drawdown circuit breaker (if equity_curve is provided)
    4. Correlation spike detection (if returns_df is provided)

    Args:
        weights_df: DataFrame with ticker, date, weight columns.
            If sector limits apply, must also have a 'sector' column.
        config: RiskConfig defining all thresholds.
        returns_df: Optional DataFrame with ticker, date, return columns.
            Required for volatility targeting and correlation detection.
        equity_curve: Optional DataFrame with date, equity columns.
            Required for drawdown circuit breaker.
        weight_col: Name of the weight column in weights_df.
        sector_col: Name of the sector column in weights_df.
        return_col: Name of the return column in returns_df.
        equity_col: Name of the equity column in equity_curve.

    Returns:
        DataFrame with all risk constraints applied, plus diagnostic
        columns (drawdown, leverage_scale, correlation_spike, avg_correlation).
    """
    result = weights_df.clone()

    # Step 1: Volatility targeting (coarse scaling — requires returns data)
    if returns_df is not None:
        result = volatility_target(
            result,
            returns_df,
            config,
            weight_col=weight_col,
            return_col=return_col,
        )

    # Step 2: Correlation spike detection (position selection — requires returns data)
    if returns_df is not None:
        result = correlation_spike_detector(
            result,
            returns_df,
            config,
            weight_col=weight_col,
            return_col=return_col,
        )

    # Step 3: Drawdown circuit breaker (leverage reduction — requires equity curve)
    # This is portfolio-level, so we compute it separately and broadcast
    if equity_curve is not None:
        dd = compute_drawdown_series(equity_curve, equity_col=equity_col)
        # Compute leverage scale from drawdown
        recovery_threshold = config.max_drawdown * config.drawdown_recovery_pct

        scale = dd.select(["date", "drawdown", "peak_equity"]).with_columns(
            pl.when(pl.col("drawdown") > config.max_drawdown)
            .then(
                (1.0 - (pl.col("drawdown") - config.max_drawdown) / config.max_drawdown)
                .clip(lower_bound=0.25)
            )
            .when(
                (pl.col("drawdown") <= config.max_drawdown)
                & (pl.col("drawdown") > recovery_threshold)
            )
            .then(
                ((pl.col("drawdown") - recovery_threshold)
                 / (config.max_drawdown - recovery_threshold) + 0.25)
                .clip(lower_bound=0.25, upper_bound=1.0)
            )
            .otherwise(pl.lit(1.0))
            .alias("leverage_scale")
        )

        result = result.join(
            scale.select(["date", "leverage_scale", "drawdown", "peak_equity"]),
            on="date",
            how="left",
        ).with_columns(
            (pl.col(weight_col) * pl.col("leverage_scale").fill_null(1.0))
            .alias(weight_col)
        )

    # Step 4: Hard constraints (final safety net — always applied last)
    result = _enforce_hard_constraints(
        result,
        config,
        weight_col=weight_col,
        sector_col=sector_col,
    )

    return result