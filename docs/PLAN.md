# Alpha — Project Plan

## Overview

Build a systematic quantitative trading system from scratch. Greenfield architecture, modern tooling, rigorous validation.

**Timeline:** ~3-4 months to paper trading. 6+ months forward testing before live.

---

## Phase 1: Data Pipeline

**Goal:** Survivorship-bias-free data ingestion to Parquet lakehouse.

### Tasks
- [ ] Set up Polygon.io account (or evaluate free alternatives for prototyping)
- [ ] Design data schema: daily bars, corporate actions, dividends, splits
- [ ] Build ingestion module:
  - [ ] Fetch OHLCV + volume for target universe (equities + futures)
  - [ ] Adjust for splits/dividends
  - [ ] Write to Parquet with partitioning (by asset, by year)
- [ ] DuckDB catalog: create views for bronze/silver/gold layers
- [ ] Validation checks:
  - [ ] No future data leakage
  - [ ] No survivorship bias (dead tickers included)
  - [ ] Gap detection (missing dates flagged)
- [ ] Initial dataset: 100+ symbols, 10+ years daily data

### Deliverables
- Ingestion script (idempotent, resumable)
- Parquet lakehouse with schema documentation
- DuckDB query catalog

### Reference
- medallion-pub `pipeline/ingestion.py` (architecture reference only)
- r/algotrading: survivorship bias is the #1 data quality issue

---

## Phase 2: Feature Store

**Goal:** Point-in-time correct feature engineering. No look-ahead bias.

### Tasks
- [ ] Design feature schema:
  - [ ] Price-derived (returns, log returns, volatility, drawdown)
  - [ ] Volume-derived (relative volume, accumulation/distribution)
  - [ ] Cross-sectional (z-scores, ranks, percentiles)
  - [ ] Regime-aware (market state features)
- [ ] Implement feature pipeline:
  - [ ] Polars-based transformations (fast, no pandas)
  - [ ] Point-in-time correct joins (no future data)
  - [ ] Feature caching (avoid recomputation)
- [ ] Feature validation:
  - [ ] No NaN/Inf leakage
  - [ ] Stationarity checks
  - [ ] Correlation matrix (flag redundant features)

### Deliverables
- Feature store module
- Feature catalog with descriptions
- Point-in-time correctness tests

---

## Phase 3: Signal Factory

**Goal:** Composable signal modules, each independently validated.

### Target Signals (start with 8, expand over time)

1. **Mean Reversion** — Z-score on rolling window (López de Prado)
2. **Momentum** — Cross-sectional rank (Jegadeesh & Titman)
3. **Volatility Regime** — HMM on price/vol/volume
4. **Trend Filter** — Kalman filter adaptive
5. **Stationarity** — Fractional differentiation (López de Prado)
6. **Cross-Asset Spread** — Cointegration pairs (Engle-Granger)
7. **Volume Anomaly** — Relative volume spike
8. **Regime-Specific** — Conditional signal weights

### Tasks
- [ ] Design signal interface (standardized input/output schema)
- [ ] Implement each signal as independent module
- [ ] Per-signal validation:
  - [ ] Information Coefficient (IC) analysis
  - [ ] IC decay over horizons
  - [ ] Win rate / Sharpe per signal
- [ ] Document each signal's hypothesis and rationale

### Deliverables
- 8 signal modules with standardized interface
- IC analysis report per signal
- Signal documentation

---

## Phase 4: Ensemble & Weights

**Goal:** Combine weak signals into a strong meta-predictor.

### Tasks
- [ ] Implement IC-validated weighting:
  - [ ] Rank IC per signal per window
  - [ ] Weight proportional to IC rank
  - [ ] Rebalance weights periodically
- [ ] LightGBM meta-learner:
  - [ ] Train on signal outputs as features
  - [ ] Target: next-period return direction
  - [ ] Early stopping on OOS data
- [ ] Ensemble validation:
  - [ ] Combined IC vs individual IC
  - [ ] Feature importance analysis
  - [ ] No single-signal dominance

### Deliverables
- Ensemble module with IC weighting + LightGBM
- Meta-learner training pipeline
- Ensemble performance report

---

## Phase 5: Portfolio & Risk

**Goal:** Position sizing, risk management, portfolio construction.

### Tasks
- [ ] Kelly criterion implementation:
  - [ ] Fractional Kelly (0.25-0.5x)
  - [ ] Per-position sizing
  - [ ] Portfolio-level Kelly cap
- [ ] Risk management:
  - [ ] Max position size (percentage of portfolio)
  - [ ] Max sector exposure
  - [ ] Max leverage
  - [ ] Volatility targeting
- [ ] Portfolio construction:
  - [ ] Risk parity allocation
  - [ ] Hierarchical Risk Parity (HRP)
  - [ ] Ledoit-Wolf covariance shrinkage
- [ ] Drawdown controls:
  - [ ] Max drawdown threshold reduces leverage
  - [ ] Correlation spike detection reduces positions

### Deliverables
- Portfolio construction module
- Risk management module
- Kelly sizing implementation

---

## Phase 6: Validation Engine

**Goal:** Rigorous out-of-sample validation. Kill overfitting.

### Tasks
- [ ] 3-part data segmentation:
  - [ ] In-Sample (training) — ~60%
  - [ ] Out-of-Sample (validation) — ~20%
  - [ ] Hold-Back (final test) — ~20% (never touched until Phase 8)
- [ ] Walk-forward analysis:
  - [ ] Rolling train/test windows (2yr train / 3mo test)
  - [ ] Parameter stability clustering
  - [ ] Performance consistency across windows
- [ ] Parameter perturbation tests:
  - [ ] Vary key parameters +/-20%
  - [ ] Verify strategy doesn't collapse
  - [ ] Document sensitivity
- [ ] Regime analysis:
  - [ ] Performance in bull/bear/sideways/crash regimes
  - [ ] No single-regime dependency
- [ ] Statistical tests:
  - [ ] Deflated Sharpe Ratio (López de Prado)
  - [ ] Bootstrap confidence intervals
  - [ ] Multiple testing correction

### Deliverables
- Validation framework
- Walk-forward results
- Perturbation test report
- Regime analysis report

---

## Phase 7: Paper Trading

**Goal:** Live execution simulation on Interactive Brokers.

### Tasks
- [ ] IB account setup (paper trading):
  - [ ] TWS API configuration
  - [ ] ib_insync integration
- [ ] Execution engine:
  - [ ] Signal to order pipeline
  - [ ] Order types (market, limit, bracket)
  - [ ] Position tracking
  - [ ] Fill reconciliation
- [ ] Risk guardrails:
  - [ ] Pre-trade position limits
  - [ ] Kill switch (manual + automated)
  - [ ] Daily P&L tracking
- [ ] Monitoring:
  - [ ] Real-time dashboard
  - [ ] Alert system (drawdown, error, anomaly)
  - [ ] Daily summary report

### Deliverables
- Paper trading engine
- IB integration module
- Monitoring dashboard
- Risk guardrails

---

## Phase 8: Forward Test

**Goal:** 6+ months of paper trading before any real capital.

### Criteria for Pass
- [ ] 6+ months of continuous paper trading
- [ ] Performance consistent with OOS backtest (within 20%)
- [ ] No unexpected drawdowns
- [ ] Execution quality acceptable (slippage within bounds)
- [ ] System stability (no crashes, no data gaps)
- [ ] Lumpy equity curve (smooth = suspicious)

### Tasks
- [ ] Run paper trading continuously
- [ ] Weekly performance review
- [ ] Monthly parameter review (no changes unless justified)
- [ ] Document all observations
- [ ] Final go/no-go decision

---

## Cross-Cutting Concerns

### Code Quality
- Type hints on all public functions
- Docstrings with signal hypotheses
- Unit tests for each module
- Git commits per logical change

### Security
- API keys in environment variables (never committed)
- No hardcoded credentials
- Private repository

### Performance
- Polars over pandas for all data operations
- DuckDB for queries (no loading everything into memory)
- Parquet for storage (columnar, compressed)
- LightGBM for ML (fast training, good accuracy)

---

## Decision Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-07-02 | Greenfield over fork | Full control, modern tools, no inherited overfitting |
| 2026-07-02 | Python + Polars + DuckDB | Community standard, fast, modern |
| 2026-07-02 | IB for execution | Gold standard for futures + equities |
| 2026-07-02 | Daily bars first | Sweet spot for retail, less noise |
| 2026-07-02 | 6+ months forward test | Community consensus, kill overfitting |
