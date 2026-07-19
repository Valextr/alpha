"""Portfolio construction — risk parity, HRP, and Ledoit-Wolf shrinkage.

Core approaches:
    1. **Ledoit-Wolf covariance shrinkage** — regularizes the sample covariance
       matrix toward a structured target, improving stability especially when
       N (assets) is large relative to T (observations).

    2. **Risk parity** — equal-risk-contribution portfolio: each asset
       contributes the same amount of risk to the portfolio. Uses iterative
       optimization with the shrinkage-corrected covariance.

    3. **Hierarchical Risk Parity (HRP)** — Lopez de Prado (2016): clusters
       assets by return correlation (hierarchical clustering), then recursively
       allocates capital down the dendrogram. No inverse covariance needed,
       more robust to noise.

Signal inputs:
    Portfolio construction expects a signal DataFrame (output of the signal
    pipeline) with columns like ``signal_<name>``. It does NOT generate signals
    itself — it takes weighted signals and returns position weights.

Typical workflow:
    1. Generate signals → ensemble weights
    2. Compute historical returns for the universe
    3. Call ``risk_parity_weights()`` or ``hrp_weights()`` to get allocations
    4. Combine signal direction × risk-parity magnitude = final positions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform


@dataclass(frozen=True, slots=True)
class PortfolioWeights:
    """Position weights for a portfolio.

    Attributes:
        ticker: Asset ticker.
        weight: Portfolio weight (fractional, sums to 1.0).
    """

    ticker: str
    weight: float


@dataclass(frozen=True, slots=True)
class RiskContribution:
    """Risk contribution per asset.

    Attributes:
        ticker: Asset ticker.
        marginal_risk: Marginal risk contribution (d portfolio_var / d weight).
        component_risk: weight * marginal_risk (absolute contribution).
        contribution_pct: Fraction of total portfolio risk.
    """

    ticker: str
    marginal_risk: float
    component_risk: float
    contribution_pct: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure the array is 2D T × N."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a


def _safe_cov(returns: np.ndarray) -> np.ndarray:
    """Return an N×N covariance matrix (even for N=1)."""
    returns = _ensure_2d(returns)
    cov = np.cov(returns, rowvar=False, ddof=1)
    if cov.ndim == 0:
        return np.array([[float(cov)]])
    return np.asarray(cov, dtype=np.float64)


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """Covariance → correlation, clamped to [-1, 1]."""
    std = np.sqrt(np.maximum(np.diag(cov), 0))
    denom = np.outer(std, std)
    denom = np.where(denom > 1e-15, denom, 1.0)
    return np.clip(cov / denom, -1.0, 1.0)


def _make_psd(matrix: np.ndarray, min_eig: float = 1e-12) -> np.ndarray:
    """Force a symmetric matrix to be positive semi-definite."""
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, min_eig)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


# ---------------------------------------------------------------------------
# Ledoit-Wolf shrinkage
# ---------------------------------------------------------------------------

def ledoit_wolf_shrinkage(
    returns: np.ndarray,
    *,
    target: str = "constant_correlation",
    shrink_constant: float | None = None,
) -> np.ndarray:
    """Ledoit-Wolf shrinkage estimate of a covariance matrix.

    Shrinks the sample covariance toward a structured target, reducing
    estimation error when N is large relative to T.

    Args:
        returns: T × N return matrix.
        target: One of "constant_correlation", "single_factor", "diagonal".
        shrink_constant: Fixed shrinkage in [0, 1] (skip estimation).

    Returns:
        N × N shrunk, positive-semi-definite covariance matrix.
    """
    returns = _ensure_2d(returns)
    t, n = returns.shape

    if t < 2:
        raise ValueError(f"Need at least 2 observations, got {t}")

    sample_cov = _safe_cov(returns)

    # Single-asset: no shrinkage needed
    if n == 1:
        return sample_cov.copy()

    # More assets than observations → diagonal fallback
    if t <= n:
        return np.diag(np.var(returns, axis=0, ddof=1))

    # --- target matrix ---
    if target == "diagonal":
        target_matrix = np.diag(np.diag(sample_cov))
    elif target == "single_factor":
        target_matrix = _single_factor_target(returns, sample_cov)
    elif target == "constant_correlation":
        target_matrix = _constant_correlation_target(sample_cov)
    else:
        raise ValueError(f"Unknown target: {target}")

    # --- shrinkage intensity ---
    if shrink_constant is not None:
        shrink = float(np.clip(shrink_constant, 0, 1))
    else:
        shrink = _lw_intensity(sample_cov, target_matrix, returns)

    result = shrink * target_matrix + (1 - shrink) * sample_cov
    return _make_psd(result)


def _lw_intensity(
    sample_cov: np.ndarray,
    target: np.ndarray,
    returns: np.ndarray,
) -> float:
    """Optimal Ledoit-Wolf shrinkage intensity (clamped to [0, 1]).

    Uses the analytical MSE-optimal coefficient from Ledoit & Wolf (2004).
    """
    t, n = returns.shape
    y = returns - returns.mean(axis=0)  # T×N, demeaned

    delta = sample_cov - target
    cp = float(np.sum(delta ** 2))
    if cp < 1e-30:
        return 1.0  # sample and target are identical

    # Fourth-moment contribution
    # For each (i, j), compute (1/T) * sum_t y_ti^2 * y_tj^2
    y2 = y ** 2
    fourth = (y2.T @ y2) / t  # N×N

    # Numerator: variance of sample covariance estimator
    num = cp - float(np.trace(fourth - sample_cov ** 2)) / t

    return float(np.clip(num / cp, 0.0, 1.0))


def _single_factor_target(returns: np.ndarray, sample_cov: np.ndarray) -> np.ndarray:
    """Fama-French single-factor target (market = equal-weighted portfolio)."""
    n = returns.shape[1]
    market = returns.mean(axis=1)
    mvar = float(np.var(market, ddof=1))
    if mvar < 1e-15:
        return np.diag(np.diag(sample_cov))

    betas = np.empty(n)
    for i in range(n):
        b = np.cov(returns[:, i], market, ddof=1)[0, 1]
        betas[i] = b / mvar

    target = np.outer(betas, betas) * mvar
    resid = returns - np.outer(market, betas)
    rvars = np.var(resid, axis=0, ddof=1)
    np.fill_diagonal(target, target.diagonal() + np.maximum(rvars, 0))
    return target


def _constant_correlation_target(sample_cov: np.ndarray) -> np.ndarray:
    """Constant-correlation target preserving individual variances on the diagonal."""
    n = sample_cov.shape[0]
    variances = np.diag(sample_cov)
    std = np.sqrt(np.maximum(variances, 0))

    if n == 1:
        return sample_cov.copy()

    corr = _cov_to_corr(sample_cov)
    # Average off-diagonal correlation
    off_diag_sum = float(np.sum(corr) - n)
    pairs = n * (n - 1)
    avg_corr = off_diag_sum / pairs if pairs else 0.0

    # Off-diagonal: constant correlation; diagonal: original variance
    target = np.outer(std, std) * avg_corr
    np.fill_diagonal(target, variances)
    return target


# ---------------------------------------------------------------------------
# Risk Parity (Equal Risk Contribution)
# ---------------------------------------------------------------------------

def risk_parity_weights(
    returns: np.ndarray,
    *,
    target_risk: float | None = None,
    max_iterations: int = 500,
    tolerance: float = 1e-6,
    initial_weights: np.ndarray | None = None,
    shrinkage: bool = True,
) -> np.ndarray:
    """Risk-parity (equal risk contribution) portfolio weights.

    Iteratively adjusts weights until each asset contributes equally to
    total portfolio risk.

    Args:
        returns: T × N return matrix.
        target_risk: Target annualized portfolio volatility. If given,
            weights are scaled so that annualised vol ≈ target_risk.
            The resulting weights may not sum to 1.0.
        max_iterations: Iteration cap.
        tolerance: Convergence threshold on per-asset risk-contribution
            imbalance (absolute deviation from 1/N).
        initial_weights: Starting point (default: equal).
        shrinkage: Apply Ledoit-Wolf shrinkage to the covariance.

    Returns:
        Length-N weight array.
    """
    returns = _ensure_2d(returns)
    cov = ledoit_wolf_shrinkage(returns) if shrinkage else _safe_cov(returns)
    n = cov.shape[0]

    # Single asset
    if n == 1:
        w = np.array([1.0])
        if target_risk is not None:
            daily = np.sqrt(cov[0, 0])
            annual = daily * np.sqrt(252)
            if annual > 0:
                w = w * (target_risk / annual)
        return w

    w = initial_weights.copy() if initial_weights is not None else np.ones(n) / n

    for _ in range(max_iterations):
        port_var = float(w @ cov @ w)
        port_vol = np.sqrt(max(port_var, 1e-30))

        marginal = cov @ w               # N
        component = w * marginal         # N
        total = float(component.sum())

        if total < 1e-15:
            rc = np.ones(n) / n
        else:
            rc = component / total       # fraction of total risk

        target_rc = np.ones(n) / n

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                rc > 1e-15,
                target_rc / rc,
                1.0,
            )

        w_new = w * ratio
        w_new = np.maximum(w_new, 1e-10)
        w_new /= w_new.sum()

        if float(np.max(np.abs(rc - target_rc))) < tolerance:
            w = w_new
            break

        w = w_new

    # Scale to target annualised volatility
    if target_risk is not None:
        port_var = float(w @ cov @ w)
        annual = np.sqrt(max(port_var, 0)) * np.sqrt(252)
        if annual > 0:
            w = w * (target_risk / annual)

    return w


def risk_contributions(
    weights: np.ndarray,
    returns: np.ndarray,
    *,
    shrinkage: bool = True,
) -> list[RiskContribution]:
    """Per-asset risk contributions.

    Args:
        weights: Portfolio weights.
        returns: T × N return matrix.
        shrinkage: Use Ledoit-Wolf shrinkage.

    Returns:
        List of ``RiskContribution`` objects.
    """
    returns = _ensure_2d(returns)
    cov = ledoit_wolf_shrinkage(returns) if shrinkage else _safe_cov(returns)
    w = np.asarray(weights, dtype=np.float64)
    n = len(w)

    port_var = float(w @ cov @ w)
    port_vol = np.sqrt(max(port_var, 1e-30))

    marginal = cov @ w
    component = w * marginal
    total = float(component.sum())
    pct = component / total if abs(total) > 1e-15 else np.zeros(n)

    return [
        RiskContribution(
            ticker=f"asset_{i}",
            marginal_risk=float(marginal[i]),
            component_risk=float(component[i]),
            contribution_pct=float(pct[i]),
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Hierarchical Risk Parity (HRP)
# ---------------------------------------------------------------------------

def hrp_weights(
    returns: np.ndarray,
    *,
    linkage_method: str = "single",
    cluster_count: int | None = None,
) -> np.ndarray:
    """Hierarchical Risk Parity weights (Lopez de Prado 2016).

    1. Cluster assets by return correlation.
    2. Sort the clustered assets recursively.
    3. Allocate capital greedily down the dendrogram.

    No matrix inversion — robust when N is large relative to T.

    Args:
        returns: T × N return matrix.
        linkage_method: Clustering linkage ("single", "complete",
            "average", "ward").
        cluster_count: Unused (reserved for a future variant).

    Returns:
        Length-N weight array summing to 1.0.
    """
    returns = _ensure_2d(returns)
    n = returns.shape[1]

    if n == 1:
        return np.array([1.0])

    cov = _safe_cov(returns)
    corr = _cov_to_corr(cov)

    # Correlation distance
    dist = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))
    np.fill_diagonal(dist, 0.0)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=linkage_method)

    w = _hrp_recursive(returns, Z)

    w = np.maximum(w, 1e-10)
    w /= w.sum()
    return w


def hrp_with_clusters(
    returns: np.ndarray,
    *,
    num_clusters: int = 3,
    linkage_method: str = "single",
) -> np.ndarray:
    """HRP with explicit cluster count.

    Allocates by inverse variance within each cluster; clusters are
    not cross-weighted (simpler, more transparent).

    Args:
        returns: T × N return matrix.
        num_clusters: Number of clusters.
        linkage_method: Clustering linkage.

    Returns:
        Length-N weight array summing to 1.0.
    """
    returns = _ensure_2d(returns)
    n = returns.shape[1]

    if n <= num_clusters:
        return np.ones(n) / n

    cov = _safe_cov(returns)
    corr = _cov_to_corr(cov)
    dist = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)

    Z = linkage(condensed, method=linkage_method)
    labels = fcluster(Z, t=num_clusters, criterion="maxclust")

    w = np.zeros(n)
    for cid in range(1, num_clusters + 1):
        mask = labels == cid
        idx = np.where(mask)[0]
        if len(idx) < 2:
            var_i = cov[idx[0], idx[0]]
            w[idx] = 1.0 / max(var_i, 1e-15)
        else:
            sub = cov[np.ix_(idx, idx)]
            di = 1.0 / np.maximum(np.diag(sub), 1e-15)
            w[idx] = di / di.sum()

    w /= w.sum()
    return w


def _hrp_recursive(
    returns: np.ndarray,
    Z: np.ndarray,
) -> np.ndarray:
    """Recursive bisection — Lopez de Prado 2016."""
    returns = _ensure_2d(returns)
    n = returns.shape[1]
    cov = _safe_cov(returns)
    leaves = leaves_list(Z)
    indices = list(leaves)

    weights: np.ndarray = np.zeros(n)

    def _mvp_frac(sub: list[int]) -> float:
        """MVP weight for the left half of a bipartition."""
        if len(sub) < 2:
            return 0.5
        sc = cov[np.ix_(sub, sub)]
        di = 1.0 / np.maximum(np.diag(sc), 1e-15)
        w = di / di.sum()
        return float(w[: len(sub) // 2].sum())

    def _alloc(sub: list[int], budget: float) -> None:
        if len(sub) == 1:
            weights[sub[0]] = budget
            return
        mid = len(sub) // 2
        left, right = sub[:mid], sub[mid:]
        alpha = _mvp_frac(sub)
        _alloc(left, budget * alpha)
        _alloc(right, budget * (1.0 - alpha))

    _alloc(indices, 1.0)
    return weights


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def returns_from_prices(
    df: pl.DataFrame,
    *,
    price_col: str = "close",
    group_by: str = "ticker",
    date_col: str = "date",
) -> pl.DataFrame:
    """Compute simple returns from a price DataFrame.

    Args:
        df: Must contain at least (ticker, date, close).
        price_col: Price column.
        group_by: Asset identifier column.
        date_col: Date column.

    Returns:
        DataFrame with ``return`` column appended.
    """
    return df.with_columns(
        ((pl.col(price_col) / pl.col(price_col).shift(1).over(group_by)) - 1).alias("return")
    )


def pivot_returns(
    df: pl.DataFrame,
    *,
    date_col: str = "date",
    return_col: str = "return",
    group_col: str = "ticker",
) -> tuple[np.ndarray, list[str]]:
    """Pivot a returns DataFrame into a T×N numpy array.

    Returns:
        ``(returns_matrix, ticker_list)`` — sorted by date, tickers as columns.
    """
    df = df.drop_nulls(subset=[return_col, date_col, group_col])

    pivoted = (
        df.pivot(
            values=return_col,
            index=date_col,
            on=group_col,
            aggregate_function="first",
        )
        .sort(date_col)
    )

    tickers = [c for c in pivoted.columns if c != date_col]
    mat = pivoted.select(tickers).to_numpy().astype(np.float64)
    return np.nan_to_num(mat, nan=0.0), tickers


def weights_to_dataframe(
    weights: np.ndarray,
    tickers: list[str],
) -> pl.DataFrame:
    """Weight array → Polars DataFrame with ``ticker`` and ``weight`` columns."""
    return pl.DataFrame({"ticker": tickers, "weight": weights.tolist()})


# ---------------------------------------------------------------------------
# Portfolio metrics
# ---------------------------------------------------------------------------

def portfolio_volatility(
    weights: np.ndarray,
    returns: np.ndarray,
    *,
    annualize: bool = True,
    trading_days: int = 252,
    shrinkage: bool = True,
) -> float:
    """Portfolio volatility.

    Args:
        weights: Portfolio weights.
        returns: T × N return matrix.
        annualize: Annualise using sqrt(trading_days).
        trading_days: Days per year.
        shrinkage: Ledoit-Wolf shrinkage on the covariance.

    Returns:
        Portfolio volatility (float).
    """
    returns = _ensure_2d(returns)
    w = np.asarray(weights, dtype=np.float64)
    cov = ledoit_wolf_shrinkage(returns) if shrinkage else _safe_cov(returns)
    var = float(w @ cov @ w)
    vol = np.sqrt(max(var, 0.0))
    if annualize:
        vol *= np.sqrt(trading_days)
    return float(vol)


def portfolio_sharpe(
    weights: np.ndarray,
    returns: np.ndarray,
    *,
    risk_free: float = 0.0,
    annualize: bool = True,
    trading_days: int = 252,
    shrinkage: bool = True,
) -> float:
    """Portfolio Sharpe ratio.

    Args:
        weights: Portfolio weights.
        returns: T × N return matrix.
        risk_free: Annualised risk-free rate (if annualize).
        annualize: Annualise mean return and volatility.
        trading_days: Days per year.
        shrinkage: Shrinking is only relevant for the weight computation,
            not the Sharpe itself (we use realised returns).

    Returns:
        Sharpe ratio (float). Returns 0.0 if std is zero.
    """
    returns = _ensure_2d(returns)
    w = np.asarray(weights, dtype=np.float64)
    pr = returns @ w
    mu = float(np.mean(pr))
    sigma = float(np.std(pr, ddof=1))
    if sigma < 1e-15:
        return 0.0
    if annualize:
        mu *= trading_days
        sigma *= np.sqrt(trading_days)
        rf = risk_free
    else:
        rf = risk_free / trading_days
    return float((mu - rf) / sigma)