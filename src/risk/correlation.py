"""Correlation spike detection.

Monitors cross-sectional return correlations. When average pairwise
correlation spikes above a threshold, it means diversification is
breaking down — most assets are moving together. In that regime,
reducing position count is a protective measure.

This is a regime filter, not a hard constraint. It signals when the
market structure itself is changing and risk is concentrated.
"""

from __future__ import annotations

import polars as pl

from .config import RiskConfig


def compute_rolling_correlation(
    returns_df: pl.DataFrame,
    *,
    lookback: int = 63,
    return_col: str = "return",
) -> pl.DataFrame:
    """Compute rolling average pairwise correlation per date.

    For each date, compute the correlation matrix over the trailing
    window, then average all off-diagonal elements.

    Args:
        returns_df: DataFrame with ticker, date, and return columns.
        lookback: Rolling window for correlation estimation.
        return_col: Name of the return column.

    Returns:
        DataFrame with date and avg_correlation columns.
    """
    # Pivot to wide format: dates × tickers
    wide = returns_df.pivot(
        index="date",
        on="ticker",
        values=return_col,
        aggregate_function="first",
    ).sort("date")

    # Compute rolling correlation matrix averages
    corr_series = []
    dates = []
    n_tickers = len(wide.columns) - 1  # exclude date column

    for i in range(len(wide)):
        start = max(0, i - lookback + 1)
        window = wide.slice(start, i - start + 1).drop("date")

        if len(window) < 20:
            corr_series.append(None)
            dates.append(wide["date"][i])
            continue

        # Compute correlation matrix
        corr_matrix = window.corr()

        # Average off-diagonal elements
        n = corr_matrix.shape[0]
        if n < 2:
            corr_series.append(None)
        else:
            total = 0.0
            count = 0
            for r in range(n):
                for c in range(n):
                    if r != c:
                        val = corr_matrix[r, c]
                        if val is not None and not (val != val):  # not NaN
                            total += val
                            count += 1
            avg = total / count if count > 0 else None
            corr_series.append(avg)

        dates.append(wide["date"][i])

    return pl.DataFrame({
        "date": dates,
        "avg_correlation": corr_series,
    }).drop_nulls()


def correlation_spike_detector(
    weights_df: pl.DataFrame,
    returns_df: pl.DataFrame,
    config: RiskConfig,
    *,
    weight_col: str = "weight",
    return_col: str = "return",
) -> pl.DataFrame:
    """Detect correlation spikes and reduce position count.

    When average pairwise correlation exceeds the threshold, keep only
    the top positions by absolute weight until min_positions_after_spike.

    Args:
        weights_df: DataFrame with ticker, date, weight columns.
        returns_df: DataFrame with ticker, date, return columns.
        config: RiskConfig with correlation thresholds.
        weight_col: Name of the weight column.
        return_col: Name of the return column.

    Returns:
        DataFrame with 'correlation_spike' flag and potentially reduced
        weights (non-selected positions set to 0).
    """
    # Compute rolling correlations
    corr = compute_rolling_correlation(
        returns_df,
        lookback=config.correlation_window,
        return_col=return_col,
    )

    # Flag spike dates
    spike_dates = corr.filter(
        pl.col("avg_correlation") > config.correlation_spike_threshold
    ).select("date")["date"].to_list()
    spike_set = set(spike_dates)

    # Join spike flags with weights
    result = weights_df.with_columns(
        pl.col("date").is_in(spike_set).alias("correlation_spike")
    )

    if len(spike_set) == 0:
        return result.with_columns(
            pl.lit(0.0).alias("avg_correlation")
        )

    # On spike dates, keep only top positions by absolute weight
    # Determine how many positions to keep per date
    positions_per_date = (
        result.filter(pl.col("correlation_spike"))
        .group_by("date")
        .agg(pl.len().alias("cnt"))
    )
    max_positions = positions_per_date["cnt"].max() if len(positions_per_date) > 0 else 5
    kept_count = max(
        config.min_positions_after_spike,
        int(max_positions * config.fraction_positions_on_spike),
    )

    # Rank positions by absolute weight within each spike date
    result = result.with_columns(
        pl.col(weight_col).abs().rank(
            method="dense",
            descending=True,
        ).over("date").alias("_weight_rank")
    )

    # Zero out positions beyond the kept count on spike dates
    result = result.with_columns(
        pl.when(
            pl.col("correlation_spike") & (pl.col("_weight_rank") > kept_count)
        )
        .then(pl.lit(0.0))
        .otherwise(pl.col(weight_col))
        .alias(weight_col)
    ).drop(["_weight_rank"])

    # Join correlation values
    result = result.join(
        corr,
        on="date",
        how="left",
    ).with_columns(
        pl.col("avg_correlation").fill_null(0.0)
    )

    return result