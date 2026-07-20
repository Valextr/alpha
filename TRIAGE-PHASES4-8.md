# Alpha Project — Phase 4-8 Triage Plan

**Date:** 2026-07-19
**Based on:** docs/PLAN.md, current codebase state, Phase 1-3 completion report

---

## Current State Assessment

### What exists
| Component | Status | Details |
|-----------|--------|---------|
| Data Pipeline (P1) | ✅ Complete | DuckDB catalog, bronze/silver/gold layers, 5 tickers, 12 years |
| Feature Store (P2) | ✅ Complete | 37 features, 57+ columns, Polars-based, point-in-time correct |
| Signal Factory (P3) | ⚠️ Partial | 4/8 signals implemented (mean reversion + momentum variants) |
| Ensemble (P4) | ❌ Not started | No ensemble modules exist |
| Portfolio (P5) | ❌ Not started | No portfolio construction exists |
| Validation (P6) | ❌ Not started | No walk-forward or hold-back framework |
| Paper Trading (P7) | ❌ Not started | No IB integration |
| Forward Test (P8) | ❌ Not started | Dependent on P7 |

### Test coverage
- 99 tests passing (Phase 1-3)
- Signal tests exist for mean reversion + momentum
- No ensemble, portfolio, or validation tests

### Data status
- 5 tickers × 12 years of daily data ingested
- Synthetic data tests in place for pipeline
- No hold-back segmentation defined

---

## Phase 4: Ensemble & Weights

**Goal:** Combine weak signals into a strong meta-predictor.

### P0: Design ensemble architecture
- **Problem:** No ensemble framework exists. Need standardized interface for combining signals.
- **Options:** IC-weighted linear combination vs LightGBM meta-learner vs both
- **Decision needed:** Which ensemble approach first? (Recommend: IC-weighted first for interpretability)
- **Files:** `src/ensemble/` (new directory)
- **Effort:** 2-3 days

### P1: Implement IC-weighted ensemble
- **Problem:** Signals need to be weighted by their predictive power
- **Approach:** Rank IC per signal per rolling window, weight proportional to IC rank
- **Dependencies:** Phase 3 signal validation (IC analysis must exist)
- **Files:** `src/ensemble/ic_weighted.py`
- **Effort:** 3-5 days

### P1: Implement LightGBM meta-learner
- **Problem:** Linear weighting may miss non-linear signal interactions
- **Approach:** Train LightGBM on signal outputs as features, target: next-period return direction
- **Dependencies:** IC-weighted ensemble must exist first (provides training targets)
- **Files:** `src/ensemble/lightgbm.py`
- **Effort:** 5-7 days

### P2: Ensemble validation
- **Problem:** Need to verify ensemble outperforms individual signals
- **Approach:** Combined IC vs individual IC, feature importance analysis
- **Files:** `tests/test_ensemble.py`
- **Effort:** 2-3 days

---

## Phase 5: Portfolio & Risk

**Goal:** Position sizing, risk management, portfolio construction.

### P0: Implement Kelly criterion sizing
- **Problem:** No position sizing framework exists
- **Approach:** Fractional Kelly (0.25-0.5x), per-position sizing
- **Dependencies:** Ensemble module must produce weighted signals
- **Files:** `src/portfolio/kelly.py`
- **Effort:** 3-5 days

### P1: Implement risk management
- **Problem:** No risk controls exist
- **Approach:** Max position size, sector exposure limits, leverage caps
- **Dependencies:** Kelly sizing must exist
- **Files:** `src/risk/management.py`
- **Effort:** 5-7 days

### P2: Implement portfolio construction
- **Problem:** No portfolio optimization exists
- **Approach:** Risk parity allocation, Ledoit-Wolf covariance shrinkage
- **Dependencies:** Risk management must exist
- **Files:** `src/portfolio/construction.py`
- **Effort:** 7-10 days

---

## Phase 6: Validation Engine

**Goal:** Rigorous out-of-sample validation. Kill overfitting.

### P0: Implement data segmentation
- **Problem:** No train/validation/hold-back split exists
- **Approach:** 60% training, 20% validation, 20% hold-back
- **Critical decision:** Define hold-back cutoff date NOW before any Phase 4 code touches data
- **Files:** `src/validation/segmentation.py`
- **Effort:** 2-3 days

### P1: Implement walk-forward analysis
- **Problem:** No walk-forward validation exists
- **Approach:** Rolling train/test windows (2yr train / 3mo test)
- **Dependencies:** Data segmentation must exist
- **Files:** `src/validation/walkforward.py`
- **Effort:** 5-7 days

### P2: Implement perturbation tests
- **Problem:** No robustness testing exists
- **Approach:** Vary key parameters ±20%, verify strategy doesn't collapse
- **Dependencies:** Walk-forward analysis must exist
- **Files:** `src/validation/perturbation.py`
- **Effort:** 3-5 days

---

## Phase 7: Paper Trading

**Goal:** Live execution simulation on Interactive Brokers.

### P0: Set up IB paper trading
- **Problem:** No IB account or integration exists
- **Approach:** TWS API configuration, ib_insync integration
- **Dependencies:** None (can start anytime)
- **Effort:** 1-2 days

### P1: Implement execution engine
- **Problem:** No order management exists
- **Approach:** Signal → order pipeline, position tracking, fill reconciliation
- **Dependencies:** IB setup must exist
- **Files:** `src/execution/engine.py`
- **Effort:** 10-14 days

### P2: Implement monitoring
- **Problem:** No dashboard or alerts exist
- **Approach:** Real-time dashboard, drawdown alerts, daily P&L tracking
- **Dependencies:** Execution engine must exist
- **Files:** `src/monitoring/` (new directory)
- **Effort:** 5-7 days

---

## Phase 8: Forward Test

**Goal:** 6+ months of paper trading before any real capital.

### P0: Define forward test criteria
- **Problem:** No success criteria defined
- **Approach:** Performance within 20% of OOS backtest, no unexpected drawdowns
- **Dependencies:** All Phase 7 tasks complete
- **Files:** `docs/FORWARD-TEST-CRITERIA.md`
- **Effort:** 1 day

### P1: Execute forward test
- **Problem:** No automated forward test runner exists
- **Approach:** Continuous paper trading with weekly reviews
- **Dependencies:** All Phase 7 tasks complete
- **Duration:** 6+ months minimum

---

## Execution Order (Critical Path)

```
1. Define hold-back cutoff date (P6-P0)
2. Complete remaining Phase 3 signals (P3-P1)
3. Design ensemble architecture (P4-P0)
4. Implement IC-weighted ensemble (P4-P1)
5. Implement LightGBM meta-learner (P4-P1)
6. Implement Kelly sizing (P5-P0)
7. Implement risk management (P5-P1)
8. Implement portfolio construction (P5-P2)
9. Implement walk-forward validation (P6-P1)
10. Implement perturbation tests (P6-P2)
11. Set up IB paper trading (P7-P0)
12. Implement execution engine (P7-P1)
13. Implement monitoring (P7-P2)
14. Define forward test criteria (P8-P0)
15. Execute forward test (P8-P1)
```

---

## Key Decisions Needed

**Hold-back cutoff date: 2023-01-01** (agreed 2026-07-19)
| Segment | Period | Coverage | Regimes |
|---------|--------|----------|---------|
| **Train (60%)** | 2014-2020 | ~6 years | Pre-COVID, steady markets |
| **Validation (20%)** | 2020-2023 | ~3 years | COVID crash → recovery → inflation |
| **Hold-back (20%)** | 2023-2026 | ~3 years | AI boom, rate cuts, current |
2. **Ensemble approach:** IC-weighted first or parallel development?
3. **LightGBM complexity:** Simple linear weighting sufficient for Phase 4?
4. **Paper trading timeline:** Can start IB setup anytime (no dependencies)
5. **Forward test duration:** 6 months minimum or extend to 12 months?

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Overfitting in ensemble | High | Critical | Strict hold-back from Phase 4 start |
| IB API complexity | Medium | High | Paper trading first, extensive testing |
| Execution slippage | High | Medium | Paper trading catches this before live |
| Parameter instability | Medium | High | Perturbation tests before Phase 7 |
| Data gaps in forward test | High | Medium | Automated gap detection and alerts |

---

## Next Steps

1. Create kanban tasks for each phase item
2. Set hold-back cutoff date immediately
3. Begin Phase 4 with ensemble architecture design
4. Start IB paper trading setup in parallel (no dependencies)