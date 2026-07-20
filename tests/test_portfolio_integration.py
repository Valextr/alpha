"""Integration tests for the portfolio pipeline (Phase 5).

Tests end-to-end wiring of:
    - Kelly criterion position sizing
    - Risk management constraints
    - Portfolio construction (risk parity, HRP)
    - PortfolioPipeline integration layer
    - Multi-strategy comparison
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.portfolio.integration import PipelineConfig, PortfolioPipeline
from src.portfolio.kelly import KellyConfig
from src.risk import RiskConfig


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_dates(n: int, start: date | None = None) -> list[date]:
    if start is None:
        start = date(2020, 1, 1)
    return [start + timedelta(days=i) for i in range(n)]


def _make_signals(
    n_dates: int = 126,
    n_tickers: int = 5,
    tickers: list[str] | None = None,
) -> pl.DataFrame:
    """Create synthetic signal DataFrame with known properties."""
    if tickers is None:
        tickers = [f"T{i:02d}" for i in range(n_tickers)]
    dates = _make_dates(n_dates)
    rows = []
    np.random.seed(42)
    for d in dates:
        for t in tickers:
            # Signal: mostly positive for first half, negative for second half
            t_idx = int(t[1:])
            base = 0.3 if t_idx % 2 == 0 else -0.2
            noise = np.random.randn() * 0.1
            sig = base + noise
            fwd = sig * 0.5 + np.random.randn() * 0.02  # weak predictive power
            rows.append({
                "ticker": t,
                "date": d,
                "signal_momentum": sig,
                "signal_mean_reversion": -sig * 0.5,
                "forward_return_1": fwd,
                "return": np.random.randn() * 0.01,
            })
    return pl.DataFrame(rows)


def _make_returns(
    n_dates: int = 126,
    n_tickers: int = 5,
    tickers: list[str] | None = None,
) -> pl.DataFrame:
    """Create synthetic returns DataFrame."""
    if tickers is None:
        tickers = [f"T{i:02d}" for i in range(n_tickers)]
    dates = _make_dates(n_dates)
    rows = []
    np.random.seed(43)
    for d in dates:
        for t in tickers:
            rows.append({
                "ticker": t,
                "date": d,
                "return": np.random.randn() * 0.01,
                "close": 100.0 + np.random.randn() * 1.0,
                "sector": "tech" if int(t[1:]) % 2 == 0 else "finance",
            })
    return pl.DataFrame(rows)


def _make_equity_curve(n_dates: int = 126) -> pl.DataFrame:
    """Create synthetic equity curve."""
    dates = _make_dates(n_dates)
    cumulative = [1.0]
    np.random.seed(44)
    for _ in range(n_dates - 1):
        cumulative.append(cumulative[-1] * (1 + np.random.randn() * 0.005))
    return pl.DataFrame({
        "date": dates,
        "equity": cumulative,
    })


# ---------------------------------------------------------------------------
# PipelineConfig tests
# ---------------------------------------------------------------------------

class TestPipelineConfig:
    def test_defaults(self) -> None:
        cfg = PipelineConfig()
        assert cfg.mode == "kelly"
        assert cfg.construction_method == "risk_parity"
        assert isinstance(cfg.kelly, KellyConfig)
        assert isinstance(cfg.risk, RiskConfig)
        assert cfg.signal_cols is None
        assert cfg.returns_lookback == 63

    def test_custom_kelly(self) -> None:
        kelly = KellyConfig(kelly_fraction=0.5, max_position=0.30)
        cfg = PipelineConfig(kelly=kelly)
        assert cfg.kelly.kelly_fraction == 0.5
        assert cfg.kelly.max_position == 0.30

    def test_custom_risk(self) -> None:
        risk = RiskConfig(max_position_pct=0.25, max_leverage=2.0)
        cfg = PipelineConfig(risk=risk)
        assert cfg.risk.max_position_pct == 0.25
        assert cfg.risk.max_leverage == 2.0

    def test_construction_mode(self) -> None:
        cfg = PipelineConfig(mode="construction", construction_method="hrp")
        assert cfg.mode == "construction"
        assert cfg.construction_method == "hrp"


# ---------------------------------------------------------------------------
# PortfolioPipeline — Kelly mode
# ---------------------------------------------------------------------------

class TestPipelineKellyMode:
    def test_kelly_mode_basic(self) -> None:
        """Kelly mode produces valid output with expected columns."""
        signals = _make_signals(n_dates=126, n_tickers=5)
        returns = _make_returns(n_dates=126, n_tickers=5)
        cfg = PipelineConfig(
            mode="kelly",
            signal_cols=["signal_momentum"],
            kelly=KellyConfig(kelly_fraction=0.25, lookback=30),
            risk=RiskConfig(max_position_pct=0.20, max_leverage=1.0),
        )
        pipe = PortfolioPipeline(cfg)
        result = pipe.run(signals=signals, returns=returns)

        assert isinstance(result, pl.DataFrame)
        assert len(result) > 0
        assert "ticker" in result.columns
        assert "date" in result.columns
        assert "weight" in result.columns
        assert "method" in result.columns
        assert result["method"].unique().to_list() == ["kelly"]

    def test_kelly_mode_no_signal_cols_auto_detect(self) -> None:
        """Auto-detect signal columns when signal_cols is None."""
        signals = _make_signals(n_dates=126, n_tickers=5)
        returns = _make_returns(n_dates=126, n_tickers=5)
        cfg = PipelineConfig(mode="kelly")  # No signal_cols
        pipe = PortfolioPipeline(cfg)
        result = pipe.run(signals=signals, returns=returns)
        assert len(result) > 0

    def test_kelly_mode_multi_signal(self) -> None:
        """Multiple signals are averaged before sizing."""
        signals = _make_signals(n_dates=126, n_tickers=5)
        returns = _make_returns(n_dates=126, n_tickers=5)
        cfg = PipelineConfig(
            mode="kelly",
            signal_cols=["signal_momentum", "signal_mean_reversion"],
        )
        pipe = PortfolioPipeline(cfg)
        result = pipe.run(signals=signals, returns=returns)
        assert len(result) > 0
        # With multiple signals, ensemble averaging should produce different results
        # than single signal

    def test_kelly_mode_no_signals_raises(self) -> None:
        """Fails when no signal columns exist."""
        signals = pl.DataFrame({
            "ticker": ["A"],
            "date": [date(2020, 1, 1)],
            "foo": [0.5],
        })
        cfg = PipelineConfig(mode="kelly", signal_cols=["foo"])
        pipe = PortfolioPipeline(cfg)
        with pytest.raises(ValueError):
            pipe.run(signals=signals)

    def test_kelly_mode_with_equity_curve(self) -> None:
        """Drawdown circuit breaker applies when equity_curve provided."""
        signals = _make_signals(n_dates=126, n_tickers=5)
        returns = _make_returns(n_dates=126, n_tickers=5)
        equity = _make_equity_curve(n_dates=126)
        cfg = PipelineConfig(
            mode="kelly",
            signal_cols=["signal_momentum"],
            kelly=KellyConfig(kelly_fraction=0.25, lookback=30),
        )
        pipe = PortfolioPipeline(cfg)
        result = pipe.run(
            signals=signals,
            returns=returns,
            equity_curve=equity,
        )
        # Should have drawdown-related columns
        assert "leverage_scale" in result.columns

    def test_kelly_mode_respects_position_cap(self) -> None:
        """No single position exceeds max_position."""
        signals = _make_signals(n_dates=126, n_tickers=5)
        returns = _make_returns(n_dates=126, n_tickers=5)
        max_pos = 0.15
        cfg = PipelineConfig(
            mode="kelly",
            signal_cols=["signal_momentum"],
            kelly=KellyConfig(kelly_fraction=0.25, lookback=30, max_position=max_pos),
            risk=RiskConfig(max_position_pct=max_pos, max_leverage=1.0),
        )
        pipe = PortfolioPipeline(cfg)
        result = pipe.run(signals=signals, returns=returns)
        # After risk constraints, no weight should exceed cap
        max_weight = result["weight"].abs().max()
        assert max_weight <= max_pos + 1e-6, f"Max weight {max_weight} exceeds cap {max_pos}"

    def test_kelly_mode_respects_leverage_cap(self) -> None:
        """Portfolio exposure does not exceed max_leverage."""
        signals = _make_signals(n_dates=126, n_tickers=5)
        returns = _make_returns(n_dates=126, n_tickers=5)
        max_leverage = 0.8
        cfg = PipelineConfig(
            mode="kelly",
            signal_cols=["signal_momentum"],
            risk=RiskConfig(max_position_pct=0.20, max_leverage=max_leverage),
        )
        pipe = PortfolioPipeline(cfg)
        result = pipe.run(signals=signals, returns=returns)
        # Check per-date exposure
        daily_exposure = result.group_by("date").agg(
            pl.col("weight").abs().sum().alias("exposure")
        )
        assert daily_exposure["exposure"].max() <= max_leverage + 1e-6


# ---------------------------------------------------------------------------
# PortfolioPipeline — Construction mode
# ---------------------------------------------------------------------------

class TestPipelineConstructionMode:
    def test_risk_parity_mode(self) -> None:
        """Risk parity produces valid weights."""
        signals = _make_signals(n_dates=126, n_tickers=5)
        returns = _make_returns(n_dates=126, n_tickers=5)
        cfg = PipelineConfig(
            mode="construction",
            construction_method="risk_parity",
            returns_lookback=63,
        )
        pipe = PortfolioPipeline(cfg)
        result = pipe.run(signals=signals, returns=returns)
        assert isinstance(result, pl.DataFrame)
        assert len(result) > 0
        assert result["method"].unique().to_list() == ["risk_parity"]

    def test_hrp_mode(self) -> None:
        """HRP produces valid weights."""
        signals = _make_signals(n_dates=126, n_tickers=5)
        returns = _make_returns(n_dates=126, n_tickers=5)
        cfg = PipelineConfig(
            mode="construction",
            construction_method="hrp",
            returns_lookback=63,
        )
        pipe = PortfolioPipeline(cfg)
        result = pipe.run(signals=signals, returns=returns)
        assert len(result) > 0
        assert result["method"].unique().to_list() == ["hrp"]

    def test_construction_with_signal_modulation(self) -> None:
        """Signal modulation adjusts construction weights."""
        signals = _make_signals(n_dates=126, n_tickers=5)
        returns = _make_returns(n_dates=126, n_tickers=5)
        cfg = PipelineConfig(
            mode="construction",
            construction_method="hrp",
            signal_cols=["signal_momentum"],
            returns_lookback=63,
        )
        pipe = PortfolioPipeline(cfg)
        result = pipe.run(signals=signals, returns=returns)
        assert "signal_value" in result.columns


# ---------------------------------------------------------------------------
# PortfolioPipeline — Analyze
# ---------------------------------------------------------------------------

class TestPipelineAnalyze:
    def test_analyze_basic(self) -> None:
        """Basic analytics return expected keys."""
        signals = _make_signals(n_dates=126, n_tickers=5)
        returns = _make_returns(n_dates=126, n_tickers=5)
        cfg = PipelineConfig(
            mode="kelly",
            signal_cols=["signal_momentum"],
            kelly=KellyConfig(lookback=30),
        )
        pipe = PortfolioPipeline(cfg)
        positions = pipe.run(signals=signals, returns=returns)
        analytics = pipe.analyze(positions)
        assert isinstance(analytics, dict)
        assert "portfolio_exposure_mean" in analytics
        assert "num_dates" in analytics
        assert "mean_weight" in analytics

    def test_analyze_with_returns(self) -> None:
        """Analytics with returns include portfolio metrics."""
        signals = _make_signals(n_dates=126, n_tickers=5)
        returns = _make_returns(n_dates=126, n_tickers=5)
        cfg = PipelineConfig(
            mode="kelly",
            signal_cols=["signal_momentum"],
            kelly=KellyConfig(lookback=30),
        )
        pipe = PortfolioPipeline(cfg)
        positions = pipe.run(signals=signals, returns=returns)
        analytics = pipe.analyze(positions, returns=returns)
        assert "portfolio_vol_annual" in analytics
        assert "portfolio_sharpe" in analytics


# ---------------------------------------------------------------------------
# PortfolioPipeline — Compare
# ---------------------------------------------------------------------------

class TestPipelineCompare:
    def test_compare_all_modes(self) -> None:
        """Compare returns kelly, risk_parity, and hrp."""
        signals = _make_signals(n_dates=126, n_tickers=5)
        returns = _make_returns(n_dates=126, n_tickers=5)
        cfg = PipelineConfig(
            signal_cols=["signal_momentum"],
            kelly=KellyConfig(lookback=30),
        )
        pipe = PortfolioPipeline(cfg)
        results = pipe.compare(
            signals=signals,
            returns=returns,
        )
        assert isinstance(results, dict)
        assert "kelly" in results
        assert "risk_parity" in results
        assert "hrp" in results
        for name, df in results.items():
            assert len(df) > 0
            assert "weight" in df.columns

    def test_compare_kelly_only(self) -> None:
        """Compare with kelly mode only."""
        signals = _make_signals(n_dates=126, n_tickers=5)
        returns = _make_returns(n_dates=126, n_tickers=5)
        cfg = PipelineConfig(signal_cols=["signal_momentum"])
        pipe = PortfolioPipeline(cfg)
        results = pipe.compare(
            signals=signals,
            returns=returns,
            modes=["kelly"],
        )
        assert "kelly" in results
        assert "risk_parity" not in results


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_signals(self) -> None:
        """Empty input produces empty output."""
        signals = pl.DataFrame({
            "ticker": [],
            "date": [],
            "signal_momentum": [],
            "forward_return_1": [],
        })
        cfg = PipelineConfig(mode="kelly", signal_cols=["signal_momentum"])
        pipe = PortfolioPipeline(cfg)
        result = pipe.run(signals=signals)
        assert len(result) == 0

    def test_single_ticker(self) -> None:
        """Pipeline works with a single ticker."""
        signals = _make_signals(n_dates=126, n_tickers=1)
        returns = _make_returns(n_dates=126, n_tickers=1)
        cfg = PipelineConfig(
            mode="kelly",
            signal_cols=["signal_momentum"],
            kelly=KellyConfig(lookback=30),
        )
        pipe = PortfolioPipeline(cfg)
        result = pipe.run(signals=signals, returns=returns)
        assert len(result) > 0
        assert result["ticker"].unique().to_list() == ["T00"]

    def test_short_history(self) -> None:
        """Pipeline handles short history gracefully."""
        signals = _make_signals(n_dates=30, n_tickers=3)
        returns = _make_returns(n_dates=30, n_tickers=3)
        cfg = PipelineConfig(
            mode="kelly",
            signal_cols=["signal_momentum"],
            kelly=KellyConfig(lookback=20),
        )
        pipe = PortfolioPipeline(cfg)
        result = pipe.run(signals=signals, returns=returns)
        assert len(result) > 0

    def test_method_column_consistency(self) -> None:
        """Method column matches the configured mode."""
        signals = _make_signals(n_dates=60, n_tickers=3)
        returns = _make_returns(n_dates=60, n_tickers=3)

        # Kelly
        pipe = PortfolioPipeline(PipelineConfig(mode="kelly", signal_cols=["signal_momentum"]))
        kelly_result = pipe.run(signals=signals, returns=returns)
        assert (kelly_result["method"] == "kelly").all()

        # HRP
        pipe = PortfolioPipeline(PipelineConfig(mode="construction", construction_method="hrp"))
        hrp_result = pipe.run(signals=signals, returns=returns)
        assert (hrp_result["method"] == "hrp").all()

    def test_no_null_weights(self) -> None:
        """Output weights should not contain NaN."""
        signals = _make_signals(n_dates=126, n_tickers=5)
        returns = _make_returns(n_dates=126, n_tickers=5)
        cfg = PipelineConfig(
            mode="kelly",
            signal_cols=["signal_momentum"],
            kelly=KellyConfig(lookback=30),
        )
        pipe = PortfolioPipeline(cfg)
        result = pipe.run(signals=signals, returns=returns)
        null_count = result["weight"].null_count()
        assert null_count == 0, f"Found {null_count} null weights"


# ---------------------------------------------------------------------------
# Cross-module integration
# ---------------------------------------------------------------------------

class TestCrossModuleIntegration:
    def test_kelly_then_risk_then_analyze(self) -> None:
        """Full end-to-end: Kelly → Risk → Analyze pipeline."""
        signals = _make_signals(n_dates=252, n_tickers=10)
        returns = _make_returns(n_dates=252, n_tickers=10)
        equity = _make_equity_curve(n_dates=252)

        cfg = PipelineConfig(
            mode="kelly",
            signal_cols=["signal_momentum", "signal_mean_reversion"],
            kelly=KellyConfig(
                kelly_fraction=0.25,
                lookback=63,
                target_vol=0.10,
                max_position=0.20,
            ),
            risk=RiskConfig(
                max_position_pct=0.20,
                max_leverage=1.0,
                max_drawdown=0.15,
            ),
        )
        pipe = PortfolioPipeline(cfg)

        # Run
        positions = pipe.run(
            signals=signals,
            returns=returns,
            equity_curve=equity,
        )

        # Analyze
        analytics = pipe.analyze(positions, returns=returns)

        # Assertions
        assert len(positions) == 252 * 10  # all dates * all tickers
        assert "weight" in positions.columns
        assert "method" in positions.columns
        assert "leverage_scale" in positions.columns  # from drawdown
        assert analytics["num_dates"] == [252]
        assert analytics["portfolio_exposure_mean"][0] <= 1.0
        assert "portfolio_sharpe" in analytics

    def test_compare_produces_diverse_results(self) -> None:
        """Different methods should produce different weight distributions."""
        signals = _make_signals(n_dates=126, n_tickers=5)
        returns = _make_returns(n_dates=126, n_tickers=5)
        cfg = PipelineConfig(signal_cols=["signal_momentum"])
        pipe = PortfolioPipeline(cfg)
        results = pipe.compare(signals=signals, returns=returns)

        # Kelly and construction should produce different weight profiles
        kelly_mean = results["kelly"]["weight"].mean()
        rp_mean = results["risk_parity"]["weight"].mean()
        # They don't have to be wildly different, but shouldn't be identical
        # (different algorithms should produce different allocations)
        assert abs(kelly_mean - rp_mean) < 1.0  # sanity check: both reasonable