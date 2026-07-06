from __future__ import annotations

import numpy as np
import polars as pl

from .registry import registry


# ---------------------------------------------------------------------------
# Hurst Exponent (Rolling R/S Method)
# ---------------------------------------------------------------------------

@registry.register(
    "hurst_63d",
    description="Rolling 63-day Hurst exponent (mean-reversion vs trending)",
    category="advanced",
    lookback=63,
)
def compute_hurst_63d(df):
    """Compute rolling Hurst exponent using R/S analysis.

    H > 0.5: trending (persistent) series
    H = 0.5: random walk
    H < 0.5: mean-reverting (anti-persistent) series

    Uses a rolling 63-day window with multiple sub-scale lags
    for the R/S calculation. Computed per-ticker.
    """
    def _hurst_rsi(series: np.ndarray) -> float:
        """Hurst exponent via R/S analysis on a single series."""
        n = len(series)
        if n < 20:
            return np.nan

        lags = np.linspace(5, n // 2, num=10).astype(int)
        lags = lags[lags >= 5]

        if len(lags) < 2:
            return np.nan

        log_rs = []
        log_lags = []

        for lag in lags:
            chunks = (n - 1) // lag
            if chunks < 2:
                continue

            rs_values = []
            for i in range(chunks):
                chunk = series[i * lag:(i + 1) * lag]
                if len(chunk) < 5:
                    continue
                mean = np.mean(chunk)
                cumulative_dev = np.cumsum(chunk - mean)
                r = np.max(cumulative_dev) - np.min(cumulative_dev)
                s = np.std(chunk, ddof=1)
                if s > 0:
                    rs_values.append(r / s)

            if rs_values:
                log_rs.append(np.log(np.mean(rs_values)))
                log_lags.append(np.log(lag))

        if len(log_rs) < 2:
            return np.nan

        slope, _ = np.polyfit(log_lags, log_rs, 1)
        return float(slope)

    results = []
    for ticker in df["ticker"].unique():
        ticker_df = df.filter(pl.col("ticker") == ticker).sort("date")
        closes = ticker_df["close"].to_numpy()
        dates = ticker_df["date"].to_numpy()

        hurst_vals = []
        window = 63
        for i in range(len(closes)):
            if i < window - 1:
                hurst_vals.append(np.nan)
            else:
                window_data = closes[i - window + 1:i + 1]
                if np.isnan(window_data).any():
                    hurst_vals.append(np.nan)
                else:
                    hurst_vals.append(_hurst_rsi(window_data))

        results.append(
            pl.DataFrame({
                "ticker": [ticker] * len(dates),
                "date": dates,
                "_hurst": hurst_vals,
            })
        )

    if not results:
        return df

    hurst_df = pl.concat(results, how="vertical_relaxed")
    return df.join(hurst_df, on=["ticker", "date"], how="left").with_columns(
        pl.col("_hurst").alias("hurst_63d")
    ).drop("_hurst")


# ---------------------------------------------------------------------------
# Kalman Filter (Alpha-Beta)
# ---------------------------------------------------------------------------

@registry.register(
    "kalman_alpha",
    description="Kalman filter alpha (level) estimate",
    category="advanced",
    lookback=21,
)
def compute_kalman_alpha(df):
    """Simple alpha-beta Kalman filter for price level estimation.

    Tracks the 'true' price level filtering out noise.
    Computed per-ticker on close prices.
    """
    alpha = 0.2
    beta = 0.1

    results = []
    for ticker in df["ticker"].unique():
        ticker_df = df.filter(pl.col("ticker") == ticker).sort("date")
        closes = ticker_df["close"].to_numpy()
        dates = ticker_df["date"].to_numpy()

        n = len(closes)
        kalman_alpha = np.full(n, np.nan)

        if n < 2:
            results.append(pl.DataFrame({
                "ticker": [ticker] * n, "date": dates, "_kalman_alpha": kalman_alpha
            }))
            continue

        estimate = closes[0]
        for i in range(1, n):
            if np.isnan(closes[i]) or np.isnan(closes[i-1]):
                continue
            predicted = estimate
            error = closes[i] - predicted
            estimate = estimate + alpha * error
            kalman_alpha[i] = estimate

        results.append(pl.DataFrame({
            "ticker": [ticker] * n, "date": dates, "_kalman_alpha": kalman_alpha
        }))

    if not results:
        return df

    kalman_df = pl.concat(results, how="vertical_relaxed")
    return df.join(kalman_df, on=["ticker", "date"], how="left").with_columns(
        pl.col("_kalman_alpha").alias("kalman_alpha")
    ).drop("_kalman_alpha")


@registry.register(
    "kalman_beta",
    description="Kalman filter beta (trend/slope) estimate",
    category="advanced",
    lookback=21,
    depends_on=["kalman_alpha"],
)
def compute_kalman_beta(df):
    """Simple alpha-beta Kalman filter for price trend estimation.

    Tracks the slope/trend of price. Positive = uptrend, negative = downtrend.
    Computed per-ticker on close prices.
    """
    alpha = 0.2
    beta = 0.1

    results = []
    for ticker in df["ticker"].unique():
        ticker_df = df.filter(pl.col("ticker") == ticker).sort("date")
        closes = ticker_df["close"].to_numpy()
        dates = ticker_df["date"].to_numpy()

        n = len(closes)
        kalman_beta = np.full(n, np.nan)

        if n < 3:
            results.append(pl.DataFrame({
                "ticker": [ticker] * n, "date": dates, "_kalman_beta": kalman_beta
            }))
            continue

        estimate = closes[0]
        trend = closes[1] - closes[0] if not np.isnan(closes[1]) else 0.0

        for i in range(1, n):
            if np.isnan(closes[i]) or np.isnan(closes[i-1]):
                continue
            predicted = estimate + trend
            error = closes[i] - predicted
            estimate = estimate + alpha * error
            trend = trend + beta * error
            kalman_beta[i] = trend

        results.append(pl.DataFrame({
            "ticker": [ticker] * n, "date": dates, "_kalman_beta": kalman_beta
        }))

    if not results:
        return df

    kalman_df = pl.concat(results, how="vertical_relaxed")
    return df.join(kalman_df, on=["ticker", "date"], how="left").with_columns(
        pl.col("_kalman_beta").alias("kalman_beta")
    ).drop("_kalman_beta")


# ---------------------------------------------------------------------------
# Fractional Differentiation (Lopez de Prado)
# ---------------------------------------------------------------------------

@registry.register(
    "frac_diff_1d",
    description="Fractional differentiation (d=0.5) of daily returns",
    category="advanced",
    lookback=63,
)
def compute_frac_diff_1d(df):
    """Fractional differentiation of price series (Lopez de Prado).

    Makes the series stationary while preserving more information
    than integer differencing. Uses d=0.5 as default.

    Coefficients decay as binomial weights, so the feature has
    a long memory but is still stationary.

    Computed per-ticker on close prices.
    """
    d = 0.5  # fractional order
    max_lag = 63

    # Pre-compute binomial coefficients for fractional differentiation
    # c(j) = (-1)^j * (d choose j), computed recursively
    coeffs = np.zeros(max_lag + 1)
    coeffs[0] = 1.0
    for j in range(1, max_lag + 1):
        coeffs[j] = coeffs[j - 1] * (d - j + 1) / j * (-1)

    results = []
    for ticker in df["ticker"].unique():
        ticker_df = df.filter(pl.col("ticker") == ticker).sort("date")
        closes = ticker_df["close"].to_numpy()
        dates = ticker_df["date"].to_numpy()

        n = len(closes)
        frac_diff = np.full(n, np.nan)

        for i in range(n):
            if np.isnan(closes[i]):
                continue

            val = 0.0
            valid = True
            for j in range(max_lag + 1):
                if i - j < 0:
                    break
                if np.isnan(closes[i - j]):
                    valid = False
                    break
                val += coeffs[j] * closes[i - j]

            if valid:
                frac_diff[i] = val

        results.append(pl.DataFrame({
            "ticker": [ticker] * n, "date": dates, "_frac_diff": frac_diff
        }))

    if not results:
        return df

    frac_df = pl.concat(results, how="vertical_relaxed")
    return df.join(frac_df, on=["ticker", "date"], how="left").with_columns(
        pl.col("_frac_diff").alias("frac_diff_1d")
    ).drop("_frac_diff")


ADVANCED_FEATURES = [
    compute_hurst_63d,
    compute_kalman_alpha,
    compute_kalman_beta,
    compute_frac_diff_1d,
]
