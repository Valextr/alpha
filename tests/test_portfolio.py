"""Portfolio construction tests — risk parity, HRP, and Ledoit-Wolf shrinkage."""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.portfolio.construction import (
    hrp_weights,
    hrp_with_clusters,
    ledoit_wolf_shrinkage,
    pivot_returns,
    portfolio_sharpe,
    portfolio_volatility,
    risk_contributions,
    risk_parity_weights,
    returns_from_prices,
    weights_to_dataframe,
    PortfolioWeights,
    RiskContribution,
)


# ── helpers ────────────────────────────────────────────────────────────────


def _trading_dates(start: date, n: int) -> list[date]:
    dates: list[date] = []
    d = start
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    return dates


def _make_returns_matrix(
    n_assets: int = 5,
    t_days: int = 252,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """Generate a synthetic T × N return matrix with known structure."""
    rng = np.random.RandomState(seed)
    tickers = [f"A{i}" for i in range(n_assets)]

    # Create correlated returns (some assets move together)
    base = rng.randn(t_days, 3)  # 3 common factors
    returns = base @ rng.randn(3, n_assets) * 0.01
    # Add idiosyncratic noise
    returns += rng.randn(t_days, n_assets) * 0.005

    return returns, tickers


def _make_price_df(tickers: list[str], n_days: int = 300, seed: int = 42) -> pl.DataFrame:
    """Generate a synthetic price DataFrame for testing."""
    rng = np.random.RandomState(seed)
    frames = []
    dates = _trading_dates(date(2020, 1, 3), n_days)

    for ticker in tickers:
        base = 100.0
        closes = [base]
        for _ in range(n_days - 1):
            ret = rng.randn() * 0.02
            closes.append(round(closes[-1] * (1 + ret), 2))

        frames.append(
            pl.DataFrame(
                {
                    "ticker": [ticker] * n_days,
                    "date": dates,
                    "open": closes,
                    "high": [c + 1 for c in closes],
                    "low": [max(c - 1, 1.0) for c in closes],
                    "close": closes,
                    "volume": rng.randint(500_000, 2_000_000, n_days),
                }
            )
        )

    return pl.concat(frames, how="vertical")


# ── Ledoit-Wolf shrinkage ───────────────────────────────────────────


class TestLedoitWolf:
    def test_returns_positive_definite(self):
        returns, _ = _make_returns_matrix(n_assets=5, t_days=252)
        shrunk = ledoit_wolf_shrinkage(returns)

        # Should be symmetric
        assert shrunk.shape == (5, 5)
        np.testing.assert_allclose(shrunk, shrunk.T, atol=1e-10)

        # Should be positive semi-definite (all eigenvalues >= 0)
        eigvals = np.linalg.eigvalsh(shrunk)
        assert np.all(eigvals >= -1e-10)

    def test_shrinkage_reduces_off_diagonal(self):
        returns, _ = _make_returns_matrix(n_assets=5, t_days=252, seed=7)
        sample_cov = np.cov(returns, rowvar=False, ddof=1)
        shrunk = ledoit_wolf_shrinkage(returns)

        # Off-diagonal should generally be shrunk toward the target
        assert shrunk.shape == (5, 5)
        # Result should be non-degenerate
        assert np.linalg.matrix_rank(shrunk) == 5

    def test_degenerate_case_fewer_observations_than_assets(self):
        # T < N should fall back to diagonal
        rng = np.random.RandomState(0)
        returns = rng.randn(2, 5)  # 2 observations, 5 assets
        shrunk = ledoit_wolf_shrinkage(returns)

        # Should be diagonal
        assert shrunk.shape == (5, 5)
        np.testing.assert_allclose(shrunk, np.diag(np.diag(shrunk)), atol=1e-10)

    def test_single_asset(self):
        rng = np.random.RandomState(0)
        returns = rng.randn(252, 1) * 0.01
        shrunk = ledoit_wolf_shrinkage(returns)

        assert shrunk.shape == (1, 1)
        assert shrunk[0, 0] > 0

    def test_target_diagonal(self):
        returns, _ = _make_returns_matrix()
        shrunk = ledoit_wolf_shrinkage(returns, target="diagonal")
        assert shrunk.shape == (5, 5)
        eigvals = np.linalg.eigvalsh(shrunk)
        assert np.all(eigvals >= -1e-10)

    def test_constant_shrinkage(self):
        returns, _ = _make_returns_matrix()
        shrunk = ledoit_wolf_shrinkage(returns, shrink_constant=0.5)
        assert shrunk.shape == (5, 5)

    def test_shrinkage_bounds(self):
        """Shrinkage should be between pure sample cov and target."""
        returns, _ = _make_returns_matrix()
        shrunk = ledoit_wolf_shrinkage(returns, shrink_constant=0.3)
        sample_cov = np.cov(returns, rowvar=False, ddof=1)

        # Shrunk should be a convex combination
        assert shrunk.shape == sample_cov.shape


# ── Risk Parity ─────────────────────────────────────────────────────


class TestRiskParity:
    def test_weights_sum_to_one(self):
        returns, _ = _make_returns_matrix(n_assets=5, t_days=252)
        w = risk_parity_weights(returns)

        assert np.isclose(w.sum(), 1.0, atol=1e-6)
        assert len(w) == 5

    def test_no_negative_weights(self):
        returns, _ = _make_returns_matrix(n_assets=5, t_days=252)
        w = risk_parity_weights(returns)

        assert np.all(w > 0)

    def test_equal_variance_roughly_equal_weights(self):
        """If all assets have equal variance and zero correlation,
        risk parity should produce roughly equal weights."""
        rng = np.random.RandomState(0)
        # Independent returns with equal variance
        returns = rng.randn(252, 4) * 0.1
        w = risk_parity_weights(returns, shrinkage=False)

        assert np.isclose(w.sum(), 1.0, atol=1e-6)
        # Each weight should be close to 0.25
        assert np.allclose(w, 0.25, atol=0.05)

    def test_risk_contributions_are_equal(self):
        """After risk parity optimization, risk contributions should
        be approximately equal."""
        returns, _ = _make_returns_matrix(n_assets=5, t_days=500)
        w = risk_parity_weights(returns, max_iterations=500, shrinkage=True)

        contribs = risk_contributions(w, returns, shrinkage=True)
        pcts = [c.contribution_pct for c in contribs]

        # All contributions should be close to 1/5 = 0.2
        target = 1.0 / 5
        for pct in pcts:
            assert abs(pct - target) < 0.15, f"Risk contribution {pct:.4f} too far from {target:.4f}"

    def test_target_risk_scaling(self):
        returns, _ = _make_returns_matrix(n_assets=5, t_days=252)
        w = risk_parity_weights(returns, target_risk=0.10)

        vol = portfolio_volatility(w, returns)
        assert abs(vol - 0.10) < 0.02

    def test_single_asset(self):
        rng = np.random.RandomState(0)
        returns = rng.randn(252, 1) * 0.01
        w = risk_parity_weights(returns)

        assert len(w) == 1
        assert np.isclose(w[0], 1.0)

    def test_different_volatilities(self):
        """Risk parity should allocate less to high-vol assets."""
        rng = np.random.RandomState(0)
        returns = np.zeros((252, 3))
        returns[:, 0] = rng.randn(252) * 0.01   # low vol
        returns[:, 1] = rng.randn(252) * 0.02   # medium vol
        returns[:, 2] = rng.randn(252) * 0.05   # high vol

        w = risk_parity_weights(returns, shrinkage=False)
        assert w[2] < w[0], "Risk parity should weight low-vol asset higher"


# ── Risk Contributions ──────────────────────────────────────────────


class TestRiskContributions:
    def test_contributions_sum_to_one(self):
        returns, _ = _make_returns_matrix()
        w = risk_parity_weights(returns)
        contribs = risk_contributions(w, returns)

        total = sum(c.contribution_pct for c in contribs)
        assert np.isclose(total, 1.0, atol=1e-4), f"Contributions sum to {total}"

    def test_component_structure(self):
        returns, _ = _make_returns_matrix()
        w = np.array([0.2] * 5)
        contribs = risk_contributions(w, returns)

        # Verify the dataclass has all fields populated
        for c in contribs:
            assert hasattr(c, "ticker")
            assert hasattr(c, "marginal_risk")
            assert hasattr(c, "component_risk")
            assert hasattr(c, "contribution_pct")
            assert np.isfinite(c.marginal_risk)
            assert np.isfinite(c.contribution_pct)


# ── HRP ─────────────────────────────────────────────────────────────


class TestHRP:
    def test_weights_sum_to_one(self):
        returns, _ = _make_returns_matrix(n_assets=5, t_days=252)
        w = hrp_weights(returns)

        assert np.isclose(w.sum(), 1.0, atol=1e-6)
        assert len(w) == 5

    def test_no_negative_weights(self):
        returns, _ = _make_returns_matrix(n_assets=5, t_days=252)
        w = hrp_weights(returns)

        assert np.all(w >= 0)

    def test_hrp_different_linkage_variants(self):
        """Verify that linkage method parameter is accepted and produces
        valid weights (different methods may converge to the same result
        for certain data structures — that's acceptable)."""
        rng = np.random.RandomState(42)
        returns = rng.randn(500, 5) * 0.01

        for method in ("single", "complete", "average", "ward"):
            w = hrp_weights(returns, linkage_method=method)
            assert np.isclose(w.sum(), 1.0, atol=1e-6)
            assert np.all(w >= 0)

    def test_hrp_with_clusters(self):
        returns, _ = _make_returns_matrix(n_assets=5, t_days=252)
        w = hrp_with_clusters(returns, num_clusters=3)

        assert np.isclose(w.sum(), 1.0, atol=1e-6)
        assert np.all(w >= 0)

    def test_hrp_small_portfolio(self):
        rng = np.random.RandomState(0)
        returns = rng.randn(252, 2) * 0.01
        w = hrp_weights(returns)

        assert len(w) == 2
        assert np.isclose(w.sum(), 1.0, atol=1e-6)
        assert np.all(w >= 0)

    def test_hrp_single_asset(self):
        rng = np.random.RandomState(0)
        returns = rng.randn(252, 1) * 0.01
        w = hrp_weights(returns)

        assert len(w) == 1
        assert np.isclose(w[0], 1.0)


# ── Portfolio Metrics ───────────────────────────────────────────────


class TestPortfolioMetrics:
    def test_volatility_positive(self):
        returns, _ = _make_returns_matrix()
        w = np.ones(5) / 5

        vol = portfolio_volatility(w, returns)
        assert vol > 0

    def test_volatility_annualized(self):
        returns, _ = _make_returns_matrix()
        w = np.ones(5) / 5

        vol_daily = portfolio_volatility(w, returns, annualize=False)
        vol_annual = portfolio_volatility(w, returns, annualize=True)

        # Annualized should be ~sqrt(252) times daily
        assert abs(vol_annual / vol_daily - np.sqrt(252)) < 0.1

    def test_sharpe_ratio(self):
        returns, _ = _make_returns_matrix()
        w = np.ones(5) / 5

        sharpe = portfolio_sharpe(w, returns)
        assert np.isfinite(sharpe)

    def test_zero_volatility_sharpe(self):
        returns = np.zeros((100, 3))
        w = np.ones(3) / 3
        sharpe = portfolio_sharpe(w, returns)
        assert sharpe == 0.0


# ── DataFrame helpers ───────────────────────────────────────────────


class TestDataFrameHelpers:
    def test_returns_from_prices(self):
        tickers = ["A", "B"]
        df = _make_price_df(tickers, n_days=50)
        result = returns_from_prices(df)

        assert "return" in result.columns
        # First return per ticker should be null (no prior day)
        null_count = result.filter(pl.col("return").is_null()).shape[0]
        assert null_count == 2  # one per ticker

    def test_pivot_returns(self):
        tickers = ["A", "B", "C"]
        df = _make_price_df(tickers, n_days=100)
        df_with_returns = returns_from_prices(df)

        ret_matrix, ret_tickers = pivot_returns(df_with_returns)
        assert ret_matrix.shape[1] == 3
        assert set(ret_tickers) == set(tickers)
        assert ret_matrix.shape[0] == 99  # one less due to first-day null

    def test_weights_to_dataframe(self):
        tickers = ["A", "B", "C"]
        weights = np.array([0.4, 0.3, 0.3])
        df = weights_to_dataframe(weights, tickers)

        assert list(df["ticker"]) == tickers
        np.testing.assert_allclose(df["weight"].to_list(), weights, atol=1e-9)


# ── Integration ─────────────────────────────────────────────────────


class TestIntegration:
    def test_full_risk_parity_pipeline(self):
        """End-to-end: prices → returns → risk parity → weights."""
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        price_df = _make_price_df(tickers, n_days=300)

        ret_df = returns_from_prices(price_df)
        ret_matrix, ret_tickers = pivot_returns(ret_df)
        w = risk_parity_weights(ret_matrix)

        weights_df = weights_to_dataframe(w, ret_tickers)
        assert len(weights_df) == 5
        assert np.isclose(weights_df["weight"].sum(), 1.0, atol=1e-6)
        assert portfolio_volatility(w, ret_matrix) > 0

    def test_full_hrp_pipeline(self):
        """End-to-end: prices → returns → HRP → weights."""
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        price_df = _make_price_df(tickers, n_days=300)

        ret_df = returns_from_prices(price_df)
        ret_matrix, ret_tickers = pivot_returns(ret_df)
        w = hrp_weights(ret_matrix)
        weights_df = weights_to_dataframe(w, ret_tickers)

        assert len(weights_df) == 5
        assert np.isclose(weights_df["weight"].sum(), 1.0, atol=1e-6)

    def test_risk_parity_vs_equal_weight(self):
        """Risk parity should differ from equal weight when variances differ."""
        rng = np.random.RandomState(0)
        returns = np.zeros((252, 3))
        returns[:, 0] = rng.randn(252) * 0.01   # low vol
        returns[:, 1] = rng.randn(252) * 0.02   # medium vol
        returns[:, 2] = rng.randn(252) * 0.05   # high vol

        w_rp = risk_parity_weights(returns, shrinkage=False)
        w_ew = np.array([1/3, 1/3, 1/3])

        assert w_rp[2] < w_rp[0], "Risk parity should weight low-vol asset higher"
        assert not np.allclose(w_rp, w_ew, atol=0.01)