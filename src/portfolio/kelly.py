"""Kelly criterion position sizing.

Hypothesis:
    Kelly criterion provides the mathematically optimal fraction to bet
    given an edge (expected return) and odds (win/loss ratio). Using
    fractional Kelly (0.25-0.5x) preserves the optimal growth rate while
    reducing volatility and drawdown risk.

    Reference: Thorp & Kelly (1956) — "A Framework for Rational Investing"

Implementation:
    For each position (ticker × date), Kelly computes the optimal
    position weight as:

        f* = (p * b - q) / b

    where:
        p  = historical win rate conditioned on the signal
        q  = 1 - p (loss rate)
        b  = win_loss_ratio = avg_win / avg_loss (decimal odds)

    This is the "discrete-outcome" Kelly variant, appropriate for
    directional signals where positions are opened and closed on
    discrete time steps.

    Additional controls:
        - Fractional Kelly: multiply f* by a fraction (0.25-0.5x)
        - Volatility targeting: scale position by target_vol / realized_vol
        - Max position cap: hard limit on any single position
        - Max portfolio exposure: sum of absolute weights capped

    The output is a position_weights DataFrame with columns:
        ticker, date, signal_value, kelly_fraction, capped_fraction, position_weight
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class KellyConfig:
    """Configuration for Kelly criterion sizing.

    Args:
        kelly_fraction: Fraction of full Kelly to use (0.25-0.5 recommended).
            Half-Kelly optimizes Sharpe ratio; quarter-Kelly is more conservative.
        lookback: Trading days of history to estimate win rate and odds.
        target_vol: Target annualized portfolio volatility (e.g. 0.10 = 10%).
            Position sizes are scaled to achieve this volatility.
        max_position: Maximum allocation to any single position (0.0-1.0).
        max_portfolio_exposure: Maximum sum of absolute position weights.
            Caps total portfolio leverage.
        min_edge: Minimum expected edge required to take a position.
            Prevents Kelly from allocating to signals with negligible edge.
        forward_horizon: Forward return horizon to use for edge estimation.
    """

    kelly_fraction: float = 0.25
    lookback: int = 63
    target_vol: float = 0.10
    max_position: float = 0.20
    max_portfolio_exposure: float = 1.0
    min_edge: float = 0.01
    forward_horizon: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < self.kelly_fraction <= 1.0:
            raise ValueError(
                f"kelly_fraction must be in (0, 1], got {self.kelly_fraction}"
            )
        if self.lookback < 5:
            raise ValueError(f"lookback must be >= 5, got {self.lookback}")
        if not 0.0 < self.target_vol <= 1.0:
            raise ValueError(f"target_vol must be in (0, 1], got {self.target_vol}")
        if not 0.0 < self.max_position <= 1.0:
            raise ValueError(f"max_position must be in (0, 1], got {self.max_position}")
        if not 0.0 < self.max_portfolio_exposure:
            raise ValueError(
                f"max_portfolio_exposure must be > 0, got {self.max_portfolio_exposure}"
            )
        if self.forward_horizon < 1:
            raise ValueError(
                f"forward_horizon must be >= 1, got {self.forward_horizon}"
            )


def compute_kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    kelly_fraction: float = 0.25,
) -> float:
    """Compute fractional Kelly for a single position.

    Uses the discrete-outcome Kelly formula:

        f* = (p * b - q) / b

    where p = win_rate, q = 1 - p, b = win/loss ratio (decimal odds).

    The result is clamped to [0, 1] and scaled by kelly_fraction.

    Args:
        win_rate: Historical win rate conditioned on the signal.
        avg_win: Average return on winning trades.
        avg_loss: Average return on losing trades (absolute value).
        kelly_fraction: Fraction of full Kelly (0.25-0.5 recommended).

    Returns:
        Fractional Kelly position weight in [0, kelly_fraction].
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0

    # Decimal odds
    b = abs(avg_win / avg_loss)
    if b <= 0:
        return 0.0

    q = 1.0 - win_rate

    # Kelly formula: (p * b - q) / b
    full_kelly = (win_rate * b - q) / b

    # Clamp to [0, 1] — negative Kelly means no bet
    full_kelly = max(0.0, min(full_kelly, 1.0))

    # Scale by fraction
    return full_kelly * kelly_fraction


def compute_position_weights(
    df: pl.DataFrame,
    signal_col: str,
    forward_col: str,
    config: KellyConfig | None = None,
) -> pl.DataFrame:
    """Compute position weights from signal scores using Kelly criterion.

    Pipeline:
        1. Estimate edge (win rate, avg win/loss) per ticker over lookback.
        2. Compute Kelly fraction from edge metrics.
        3. Apply volatility targeting (scale by target_vol / realized_vol).
        4. Apply per-position cap.
        5. Apply portfolio exposure cap (cross-sectional scaling).

    Args:
        df: DataFrame with signal and forward return columns,
            sorted by (ticker, date). Must contain 'ticker', 'date',
            the signal column, and the forward return column.
        signal_col: Signal column name.
        forward_col: Forward return column name (e.g. 'forward_return_1').
        config: Kelly configuration. Defaults to KellyConfig().

    Returns:
        DataFrame with position weights:
            ticker, date, signal_value, kelly_fraction,
            vol_targeted_fraction, position_weight,
            portfolio_exposure (sum of absolute weights on that date)
    """
    if config is None:
        config = KellyConfig()

    # Validate input
    required_cols = ["ticker", "date", signal_col, forward_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Work on a clone, drop rows with null signal or forward return
    result = df.clone().filter(
        pl.col(signal_col).is_not_null() & pl.col(forward_col).is_not_null()
    )

    if len(result) == 0:
        return pl.DataFrame(schema={
            "ticker": pl.String,
            "date": pl.Date,
            "signal_value": pl.Float64,
            "kelly_fraction": pl.Float64,
            "vol_targeted_fraction": pl.Float64,
            "position_weight": pl.Float64,
            "portfolio_exposure": pl.Float64,
        })

    min_samples = max(1, config.lookback // 2)

    # ── Step 1: Win/loss indicators ──────────────────────────────────

    # Is win: signal and forward return agree in sign
    is_win = (
        pl.when(
            (pl.col(signal_col) > 0) & (pl.col(forward_col) > 0)
            | (pl.col(signal_col) < 0) & (pl.col(forward_col) < 0)
        )
        .then(1.0)
        .otherwise(0.0)
    )

    # Win magnitude: absolute forward return when it's a win
    win_mag = (
        pl.when(
            (pl.col(signal_col) > 0) & (pl.col(forward_col) > 0)
        ).then(pl.col(forward_col))
        .when(
            (pl.col(signal_col) < 0) & (pl.col(forward_col) < 0)
        ).then(pl.col(forward_col).abs())
    )

    # Loss magnitude: absolute forward return when it's a loss
    loss_mag = (
        pl.when(
            (pl.col(signal_col) > 0) & (pl.col(forward_col) < 0)
        ).then(pl.col(forward_col).abs())
        .when(
            (pl.col(signal_col) < 0) & (pl.col(forward_col) > 0)
        ).then(pl.col(forward_col).abs())
    )

    result = result.with_columns([
        is_win.alias("_is_win"),
        win_mag.alias("_win_mag"),
        loss_mag.alias("_loss_mag"),
    ])

    # ── Step 2: Rolling edge estimation per ticker ───────────────────

    rolling_stats = [
        pl.col("_is_win")
        .rolling_mean(window_size=config.lookback, min_samples=min_samples)
        .over("ticker")
        .alias("_rolling_win_rate"),
        pl.col("_win_mag")
        .rolling_mean(window_size=config.lookback, min_samples=min_samples)
        .over("ticker")
        .alias("_rolling_avg_win"),
        pl.col("_loss_mag")
        .rolling_mean(window_size=config.lookback, min_samples=min_samples)
        .over("ticker")
        .alias("_rolling_avg_loss"),
    ]

    result = result.with_columns(rolling_stats)

    # Fill nulls (early rows without enough history) with neutral defaults
    result = result.with_columns([
        pl.col("_rolling_win_rate").fill_null(0.5),
        pl.col("_rolling_avg_win").fill_null(0.0),
        pl.col("_rolling_avg_loss").fill_null(0.0),
    ])

    # Kelly fraction: f* = (p * b - q) / b
    # b = avg_win / avg_loss, q = 1 - win_rate
    safe_loss = pl.col("_rolling_avg_loss").fill_null(1e-6)
    b_ratio = pl.col("_rolling_avg_win") / safe_loss.clip(lower_bound=1e-10)
    q_rate = 1.0 - pl.col("_rolling_win_rate")

    full_kelly = (pl.col("_rolling_win_rate") * b_ratio - q_rate) / b_ratio.clip(lower_bound=1e-10)
    clamped_kelly = full_kelly.clip(lower_bound=0.0, upper_bound=1.0)

    result = result.with_columns([
        (clamped_kelly * config.kelly_fraction).alias("kelly_fraction"),
    ])

    # ── Step 3: Volatility targeting ─────────────────────────────────

    daily_vol = (
        pl.col(forward_col)
        .rolling_std(window_size=config.lookback, min_samples=min_samples)
        .over("ticker")
        * (252.0**0.5)
    )

    result = result.with_columns(
        daily_vol.fill_null(config.target_vol).alias("_realized_vol_annual")
    )

    # vol_scale = target_vol / realized_vol (capped at 1.0 so we don't
    # increase position size when realized vol is below target)
    vol_scale = config.target_vol / pl.col("_realized_vol_annual").clip(
        lower_bound=config.target_vol
    )

    result = result.with_columns([
        (pl.col("kelly_fraction").fill_null(0.0) * vol_scale)
        .alias("vol_targeted_fraction"),
    ])

    # ── Step 4: Per-position cap ─────────────────────────────────────

    result = result.with_columns([
        pl.col("vol_targeted_fraction")
        .fill_null(0.0)
        .clip(upper_bound=config.max_position)
        .alias("capped_fraction"),
    ])

    # Apply signal direction: positive → long, negative → short
    signal_sign = (
        pl.when(pl.col(signal_col) > 0).then(1.0)
        .when(pl.col(signal_col) < 0).then(-1.0)
        .otherwise(0.0)
    )
    result = result.with_columns([
        (pl.col("capped_fraction") * signal_sign).alias("position_weight"),
    ])

    # ── Step 5: Portfolio exposure cap ───────────────────────────────

    # Sum of absolute weights per date
    result = result.with_columns(
        pl.col("position_weight").abs().sum().over("date").alias("portfolio_exposure")
    )

    # Scale down if total exposure exceeds cap
    result = result.with_columns([
        pl.when(
            pl.col("portfolio_exposure") > config.max_portfolio_exposure
        ).then(
            pl.col("position_weight")
            * (config.max_portfolio_exposure / pl.col("portfolio_exposure"))
        ).otherwise(pl.col("position_weight")).alias("position_weight"),
    ])

    # Re-compute exposure after scaling
    result = result.with_columns(
        pl.col("position_weight").abs().sum().over("date").alias("portfolio_exposure")
    )

    # ── Build output ────────────────────────────────────────────────

    return result.select([
        "ticker",
        "date",
        pl.col(signal_col).alias("signal_value"),
        "kelly_fraction",
        "vol_targeted_fraction",
        "capped_fraction",
        "position_weight",
        "portfolio_exposure",
    ]).sort(["date", "ticker"])


def compute_position_weights_from_signal_scores(
    df: pl.DataFrame,
    signal_cols: list[str],
    config: KellyConfig | None = None,
) -> pl.DataFrame:
    """Compute position weights from multiple signal columns (ensemble-ready).

    Averages signal columns first, then passes the composite score to
    Kelly sizing. This interface is designed for Phase 4 integration:
    the ensemble produces a single weighted signal, and Kelly sizes
    positions from that composite.

    Args:
        df: DataFrame with signal columns and forward returns.
        signal_cols: List of signal column names to combine.
        config: Kelly configuration.

    Returns:
        Position weights DataFrame.
    """
    if config is None:
        config = KellyConfig()

    if not signal_cols:
        raise ValueError("signal_cols must not be empty")

    for col in signal_cols:
        if col not in df.columns:
            raise ValueError(f"Missing signal column: {col}")

    # Composite signal (mean of signals) — simple horizontal average
    expr = sum(pl.col(c) for c in signal_cols) / len(signal_cols)
    df = df.with_columns(expr.alias("_composite"))

    forward_col = f"forward_return_{config.forward_horizon}"
    if forward_col not in df.columns:
        from src.signals.base import compute_forward_returns

        df = compute_forward_returns(df, horizons=[config.forward_horizon])

    return compute_position_weights(df, "_composite", forward_col, config)