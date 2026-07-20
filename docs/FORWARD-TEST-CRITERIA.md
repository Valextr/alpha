# Forward Test — Success Criteria

**Date:** 2026-07-19
**Phase:** 8 (Forward Test)
**Prerequisites:** Phase 4-7 complete, Phase 6 hold-back segmentation locked

---

## Purpose

The forward test is the final gate before deploying real capital. It runs the
complete system (ensemble → portfolio → execution) on live Interactive Brokers
paper trading for a minimum of 6 months. The criteria below define exactly what
"pass" means — no hand-waving, no subjective judgment after the fact.

If any **Critical** criterion fails, the system does NOT graduate to live
capital. If a **Warning** criterion fails, the system graduates with documented
risk exposure and reduced initial capital allocation.

---

## 1. Duration Criterion

| Metric | Threshold | Severity |
|--------|-----------|----------|
| Minimum live trading days | ≥ 252 trading days (≈ 6 calendar months) | Critical |
| Maximum gap between trades | ≤ 30 calendar days without any signal activity | Warning |
| Calendar diversity | At least 1 trading day in each of 6+ distinct calendar months | Critical |

**Rationale:** Six months covers seasonal regimes and provides statistical
power. Gaps >30 days suggest infrastructure instability. Monthly diversity
ensures the test spans different market conditions.

---

## 2. Performance vs Backtest (OOS Comparison)

The forward test equity curve is compared against the **Out-of-Sample
backtest** run on the validation (20%) dataset. The hold-back (20%) dataset
remains untouched until the forward test concludes and is used only as a final
confirmation checkpoint.

### 2.1 Return metrics

| Metric | Threshold | Severity |
|--------|-----------|----------|
| Forward CAGR within X% of OOS CAGR | Forward ≥ 80% of OOS CAGR | Critical |
| Forward net return sign matches OOS | Both positive or both negative | Warning |
| Monthly hit rate | Forward ≥ 70% of OOS monthly win rate | Warning |

**Rationale:** A forward test returning less than 80% of the OOS backtest
suggests live execution costs, slippage, or signal decay. Complete sign
mismatch (positive OOS vs negative forward) indicates the strategy does not
generalize.

### 2.2 Risk-adjusted metrics

| Metric | Threshold | Severity |
|--------|-----------|----------|
| Forward Sharpe ratio | ≥ 0.5 (absolute minimum) AND ≥ 60% of OOS Sharpe | Critical |
| Forward Sortino ratio | ≥ 0.7 (absolute minimum) | Warning |
| Forward Calmar ratio (CAGR / max drawdown) | ≥ 1.0 | Warning |

**Rationale:** Sharpe ≥ 0.5 is the floor for any strategy worth deploying.
Sortino ≥ 0.7 ensures downside risk is controlled. Calmar ≥ 1.0 means the
strategy earns more per unit of drawdown than it endures.

---

## 3. Drawdown Criteria

| Metric | Threshold | Severity |
|--------|-----------|----------|
| Maximum forward drawdown | ≤ 150% of OOS max drawdown | Critical |
| Maximum forward drawdown (absolute) | ≤ 20% from peak | Critical |
| Drawdown duration | Any drawdown > 5% recovers within 90 trading days | Warning |
| Drawdown frequency | Number of drawdowns > 3% ≤ 2× OOS count | Warning |

**Rationale:** Forward drawdowns exceeding 150% of OOS suggests unseen risk.
The 20% absolute cap is a hard stop — no strategy deploying real capital should
exceed 20% drawdown in paper trading. Recovery time > 90 days for a 5% drawdown
signals regime breakdown.

---

## 4. Execution Quality

| Metric | Threshold | Severity |
|--------|-----------|----------|
| Average slippage per trade | ≤ 2× the instrument's typical bid-ask spread | Critical |
| Fill rate | ≥ 95% of orders filled within 1 bar (daily) | Critical |
| Partial fills | ≤ 5% of orders partially filled | Warning |
| Rejected orders | 0% (any rejection requires investigation) | Warning |
| Signal-to-order latency | ≤ 5 minutes from signal generation to order submission | Warning |

**Rationale:** Slippage > 2× spread destroys alpha. Fill rate < 95% on daily
bars suggests the strategy requires liquidity it cannot access. Any rejected
order indicates a configuration or risk-limit problem.

---

## 5. System Stability

| Metric | Threshold | Severity |
|--------|-----------|----------|
| Uptime (execution engine) | ≥ 99% of market hours | Critical |
| Data gap incidents | 0 missed trading days in data ingestion | Critical |
| Unhandled exceptions | ≤ 2 per month (all resolved within 24 hours) | Warning |
| Manual interventions | ≤ 1 per week (restarts, config fixes) | Warning |
| Kill switch activations | Any activation triggers mandatory 1-week cooling-off review | Critical |

**Rationale:** ≥ 99% uptime means the system can run autonomously. Data gaps
corrupt signal computation and are unacceptable. > 1 manual intervention/week
means the system is not production-ready.

---

## 6. Signal Integrity

| Metric | Threshold | Severity |
|--------|-----------|----------|
| Signal IC (live) | ≥ 50% of OOS IC per signal | Warning |
| Ensemble IC (live) | ≥ 60% of OOS ensemble IC | Critical |
| Signal regime shift | No single signal dominates > 40% of ensemble weight for > 20 consecutive days | Warning |
| Predictive power stability | Rolling 20-day IC does not cross zero for > 5 consecutive days | Warning |

**Rationale:** Live IC dropping below 50% of OOS per signal suggests
market microstructure changed. Ensemble IC below 60% of OOS is the critical
line — the combined predictor is losing edge. Single-signal dominance indicates
ensemble weights are not adapting properly.

---

## 7. Risk Management Compliance

| Metric | Threshold | Severity |
|--------|-----------|----------|
| Position size violations | 0 instances where Kelly sizing was exceeded | Critical |
| Sector exposure violations | 0 instances exceeding max sector allocation | Critical |
| Leverage cap violations | 0 instances exceeding max leverage | Critical |
| Volatility targeting drift | Live realized vol within ±25% of target vol | Warning |

**Rationale:** Any risk limit violation in paper trading is a critical failure —
it would have real consequences in live trading. Volatility drift outside ±25%
suggests the vol-targeting module is not adapting to market conditions.

---

## 8. Equity Curve Shape (Sanity Check)

| Metric | Threshold | Severity |
|--------|-----------|----------|
| Equity curve smoothness (autocorrelation) | Daily return autocorrelation at lag-1 < 0.3 | Critical |
| Equity curve smoothness (Sharpe ceiling) | Forward Sharpe ≤ 3.0 (if > 3.0, investigate for errors) | Warning |
| Lumpy distribution test | Daily returns are NOT normally distributed (Jarque-Bera p < 0.05) | Warning |

**Rationale:** An autocorrelated equity curve suggests look-ahead bias or
position carryover errors. Sharpe > 3 in live trading is statistically
extraordinary and usually indicates a measurement error. Non-normal returns
(lumpy equity curve) is actually a GOOD sign — it means the strategy has
genuine directional bets, not a risk-free arbitrage.

---

## 9. Go/No-Go Decision Matrix

After 6+ months, score each criterion:

| Score | Definition |
|-------|------------|
| PASS | All Critical criteria met, ≥ 75% of Warning criteria met |
| CONDITIONAL PASS | All Critical criteria met, 50-74% of Warning criteria met |
| FAIL | Any Critical criterion failed |

### Outcome actions

| Outcome | Action |
|---------|--------|
| **PASS** | System graduates to live capital. Start with 25% of intended AUM, scale to 100% over 3 months if live performance holds. |
| **CONDITIONAL PASS** | Document which Warning criteria failed. Start live at 10% of intended AUM. Extend paper trading by 3 months while running live at reduced scale. Re-evaluate after extension. |
| **FAIL** | Do NOT deploy live capital. Diagnose which Critical criterion(s) failed. Return to the relevant Phase (4-7) for fixes. Rebuild and re-run the full 6-month forward test from scratch. |

---

## 10. Monitoring Cadence

| Frequency | Activity |
|-----------|----------|
| **Daily** | P&L logged, position snapshot, data gap check |
| **Weekly** | Performance vs OOS comparison, slippage review, signal IC check |
| **Monthly** | Full criterion scorecard, regime analysis, parameter stability check |
| **Quarterly** | Deep review: all criteria, Deflated Sharpe Ratio, perturbation sensitivity |

---

## 11. Deferral Rules

The forward test clock **pauses** (does not count toward the 6-month minimum) during:

- Market closures > 3 consecutive trading days (holidays excluded)
- IB outage confirmed by IB status page > 4 hours
- Scheduled system maintenance (data re-ingestion, model retraining) < 48 hours

Any deferral > 48 hours requires documentation and extends the forward test
proportionally.

---

## Appendix A: OOS Reference Values (To Be Filled After Phase 6)

Populate these thresholds after Phase 6 (Validation Engine) completes:

| Metric | OOS Value | Forward Threshold |
|--------|-----------|-------------------|
| OOS CAGR | [TBD] | ≥ 80% of OOS |
| OOS Sharpe | [TBD] | ≥ 60% of OOS |
| OOS Max Drawdown | [TBD] | ≤ 150% of OOS |
| OOS Monthly Win Rate | [TBD] | ≥ 70% of OOS |
| OOS Ensemble IC | [TBD] | ≥ 60% of OOS |
| OOS Per-signal IC | [TBD] | ≥ 50% of OOS |
| OOS Volatility | [TBD] | ±25% of OOS |

---

## Appendix B: Statistical Tests (Phase 6 Deliverables)

These statistical tests run on the forward test results:

1. **Deflated Sharpe Ratio** (López de Prado): Adjusts Sharpe for multiple
   testing. DSR > 0 confirms the Sharpe is not a product of data mining.
2. **Bootstrap confidence intervals**: 10,000 resamples of the forward test
   equity curve. If the 95% CI for Sharpe includes 0, the strategy is not
   statistically significant.
3. **Diebold-Mariano test**: Tests whether forward test performance is
   statistically different from the OOS backtest. p < 0.05 means the difference
   is significant.