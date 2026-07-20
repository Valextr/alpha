"""IC-weighted ensemble for combining directional signals.

Architecture
------------
1. Compute cross-sectional IC per signal per date (Spearman rank correlation
   across tickers).
2. Average IC over a rolling window (default 63 trading days ~3 months).
3. Convert rolling ICs to weights using one of several methods.
4. Apply weights to current signal values to produce the ensemble score.
5. Rebalance weights at a configurable frequency (default 5 days ~weekly).

Cross-sectional IC vs flat IC
-----------------------------
Cross-sectional IC measures how well a signal ranks tickers relative to each
other on a given day. This is the standard metric in quantitative research
because it reflects the signal's utility for portfolio construction.

Flat IC (in src/signals/base.py) measures overall predictive power across
all (ticker, date) observations. Both are useful, but CS IC drives the
weighting logic here.
"""

from __future__ import annotations

from typing import Sequence

import polars as pl

from .base import ic_to_weights


# ── Core IC functions ────────────────────────────────────────────────


def compute_cross_sectional_ic(
    df: pl.DataFrame,
    signal_col: str,
    target_col: str,
) -> pl.DataFrame:
    """Compute cross-sectional IC (Spearman rank correlation) per date.

    For each date, rank *signal_col* and *target_col* across the ticker
    cross-section, then compute Pearson correlation on the ranks
    (equivalent to Spearman).

    Returns
    -------
    pl.DataFrame
        Two columns: ``date`` and ``cs_ic``.
    """
    valid = df.select(["date", signal_col, target_col]).drop_nulls()

    ranked = valid.with_columns([
        pl.col(signal_col).rank("average").over("date").alias("_rank_s"),
        pl.col(target_col).rank("average").over("date").alias("_rank_t"),
    ])

    cs_ic = (
        ranked.group_by("date")
        .agg(pl.corr(pl.col("_rank_s"), pl.col("_rank_t")).alias("cs_ic"))
    )

    # With one ticker per date, correlation is NaN — drop those rows
    cs_ic = cs_ic.filter(pl.col("cs_ic").is_not_nan())

    # Clip to [-1, +1] to handle floating-point edge cases with
    # very small cross-sections (e.g., only 3 tickers per date).
    cs_ic = cs_ic.with_columns(
        pl.col("cs_ic").clip(lower_bound=-1.0, upper_bound=1.0)
    )

    return cs_ic.select(["date", "cs_ic"])


def compute_rolling_ic(
    df: pl.DataFrame,
    signal_col: str,
    target_col: str,
    window: int = 63,
) -> pl.DataFrame:
    """Compute rolling cross-sectional IC over a trailing window.

    Parameters
    ----------
    df : pl.DataFrame
        Must contain ``date``, *signal_col*, and *target_col*.
    signal_col : str
        Signal column name.
    target_col : str
        Forward return column name.
    window : int
        Rolling window size in trading days (default 63 ~3 months).

    Returns
    -------
    pl.DataFrame
        Two columns: ``date`` and ``rolling_ic``.
    """
    cs_ic = compute_cross_sectional_ic(df, signal_col, target_col)

    return (
        cs_ic.sort("date")
        .with_columns(
            pl.col("cs_ic")
            .rolling_mean(window_size=window, min_samples=max(1, window // 2))
            .alias("rolling_ic")
        )
        .select(["date", "rolling_ic"])
    )


# NOTE: ic_to_weights() is imported from base.py for use by
# ICWeightedEnsemble._compute_weight_schedule.


# ── Ensemble class ───────────────────────────────────────────────────


class ICWeightedEnsemble:
    """IC-weighted signal ensemble.

    Computes rolling cross-sectional IC for each signal, converts ICs
    to weights, and applies them to produce a composite ensemble score.

    Parameters
    ----------
    ic_window : int
        Rolling window for IC computation (trading days). Default 63.
    rebalance_freq : int
        How often to update weights (trading days). Default 5 (weekly).
    weight_method : str
        Method for converting IC to weights. One of ``"abs_ic"``,
        ``"rank_ic"``, or ``"positive_ic"``. Default ``"abs_ic"``.
    """

    def __init__(
        self,
        ic_window: int = 63,
        rebalance_freq: int = 5,
        weight_method: str = "abs_ic",
    ):
        self.ic_window = ic_window
        self.rebalance_freq = rebalance_freq
        self.weight_method = weight_method
        self._weight_schedule: pl.DataFrame | None = None
        self._signal_cols: list[str] = []

    def _compute_weight_schedule(
        self,
        all_dates: list,
        ic_wide: pl.DataFrame,
        signal_cols: list[str],
    ) -> pl.DataFrame:
        """Build a weight schedule: one row per date, columns for each signal weight.

        Weights are computed at rebalance dates from the rolling IC table,
        then forward-filled between rebalance points.
        """
        rebalance_dates = all_dates[::self.rebalance_freq]

        weight_rows: list[dict] = []
        for rd in rebalance_dates:
            # Get the most recent IC values available at or before this rebalance date
            row = ic_wide.filter(pl.col("date") <= rd).sort("date").tail(1)
            if len(row) == 0:
                continue

            ic_vals: dict[str, float] = {}
            for sig in signal_cols:
                val = row[sig].to_list()[0] if sig in row.columns else None
                ic_vals[sig] = val if val is not None else 0.0

            weights = ic_to_weights(ic_vals, self.weight_method)
            wrow = {"date": rd}
            for sig in signal_cols:
                wrow[f"w_{sig}"] = weights.get(sig, 0.0)
            weight_rows.append(wrow)

        if not weight_rows:
            # Fallback: equal weights for all dates
            equal_w = 1.0 / len(signal_cols) if signal_cols else 0.0
            return pl.DataFrame(
                {
                    "date": all_dates,
                    **{f"w_{sig}": equal_w for sig in signal_cols},
                }
            )

        weight_schedule = pl.DataFrame(weight_rows)

        # Fill all dates with equal weights first, then overlay the computed
        # schedule. This ensures early dates (before the first rebalance)
        # still have valid weights instead of null.
        default_w = 1.0 / len(signal_cols) if signal_cols else 0.0
        dates_df = pl.DataFrame({"date": all_dates})
        filled = dates_df.with_columns(
            [pl.lit(default_w).alias(f"w_{sig}") for sig in signal_cols]
        )

        if weight_schedule.is_empty():
            return filled

        # Overlay the computed weight schedule on top of the default-equal base
        filled = filled.join(weight_schedule, on="date", how="left")

        for sig in signal_cols:
            wcol = f"w_{sig}"
            if wcol in filled.columns:
                # Forward-fill from each rebalance point, then backward-fill
                # for any remaining gaps (should be rare after the above).
                filled = filled.with_columns(
                    pl.col(wcol).forward_fill().backward_fill().fill_null(default_w)
                )

        return filled

    def transform(
        self,
        df: pl.DataFrame,
        signal_cols: Sequence[str],
        target_col: str,
    ) -> pl.DataFrame:
        """Compute IC-weighted ensemble scores.

        Steps:
        1. Compute rolling cross-sectional IC for each signal.
        2. Build weight schedule (rebalance at configured frequency).
        3. Apply weights to signal values.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame containing ``date``, signal columns, and the target.
        signal_cols : Sequence[str]
            Signal column names (e.g., ``["signal_mean_reversion_21d", ...]``).
        target_col : str
            Forward return column used for IC computation
            (e.g., ``"forward_return_1"``).

        Returns
        -------
        pl.DataFrame
            Input DataFrame with ``ensemble_score`` column appended, plus
            ``w_<signal>`` columns showing the active weight for each signal.
        """
        signal_cols = list(signal_cols)
        self._signal_cols = signal_cols

        # 1. Compute rolling IC for each signal (wide format)
        ic_parts: list[pl.DataFrame] = []
        for sig in signal_cols:
            rolling = compute_rolling_ic(df, sig, target_col, self.ic_window)
            ic_parts.append(rolling.with_columns(
                pl.col("rolling_ic").alias(sig)
            ).select(["date", sig]))

        # Merge into wide format
        ic_wide = ic_parts[0]
        for part in ic_parts[1:]:
            ic_wide = ic_wide.join(part, on="date", how="full", coalesce=True)

        # 2. Build weight schedule
        all_dates = sorted(df["date"].unique().to_list())
        weight_schedule = self._compute_weight_schedule(all_dates, ic_wide, signal_cols)

        # Cache for predict()
        self._weight_schedule = weight_schedule

        # 3. Join weights and compute ensemble score
        result = df.join(weight_schedule, on="date", how="left")

        weighted_sum = pl.lit(0.0)
        for sig in signal_cols:
            wcol = f"w_{sig}"
            weighted_sum = weighted_sum + pl.col(sig) * pl.col(wcol)

        result = result.with_columns(weighted_sum.alias("ensemble_score"))

        return result

    def predict(
        self,
        df: pl.DataFrame,
        signal_cols: Sequence[str],
    ) -> pl.DataFrame:
        """Apply cached weights to produce ensemble scores.

        Use this when you have already called ``transform()`` and want to
        apply the learned weights to a new DataFrame (e.g., hold-back period).

        Parameters
        ----------
        df : pl.DataFrame
            Must contain ``date`` and signal columns.
        signal_cols : Sequence[str]
            Same signal column names used in ``transform()``.

        Returns
        -------
        pl.DataFrame
            Input DataFrame with ``ensemble_score`` column appended.
        """
        if self._weight_schedule is None:
            raise RuntimeError("Call transform() before predict()")

        signal_cols = list(signal_cols)
        result = df.join(self._weight_schedule, on="date", how="left")

        # Forward-fill any missing weights (e.g., dates beyond training)
        for sig in signal_cols:
            wcol = f"w_{sig}"
            if wcol in result.columns:
                result = result.with_columns(
                    pl.col(wcol).forward_fill()
                )

        weighted_sum = pl.lit(0.0)
        for sig in signal_cols:
            wcol = f"w_{sig}"
            if wcol in result.columns:
                weighted_sum = weighted_sum + pl.col(sig) * pl.col(wcol)

        return result.with_columns(weighted_sum.alias("ensemble_score"))