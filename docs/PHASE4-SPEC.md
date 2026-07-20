# Alpha — Phase 4: Ensemble Architecture Specification

## Objective

Combine weak individual signals (mean reversion, momentum, and future signals)
into a single meta-predictor that outperforms any individual signal in
Information Coefficient (IC) and directional accuracy.

**Target:** IC-weighted linear ensemble as the Phase 4 baseline. This approach
is interpretable, cannot overfit the way a meta-learner can, and establishes a
measurable baseline before any LightGBM layer is considered.

---

## 1. Design Decisions

### 1.1 Approach: IC-weighted linear combination (first)

**Decision:** Start with IC-weighted linear weighting, not LightGBM.

**Rationale:**
- Linear weighting is transparent: you can audit exactly why the ensemble is
  long or short on any given day by inspecting the weights and signal values.
- A LightGBM meta-learner introduces a training loop with hyperparameters,
  early-stopping thresholds, and feature-engineering choices — all of which
  are overfit vectors. We need the linear baseline first to know whether the
  signals have enough collective edge to justify a black box.
- IC weighting updates automatically each window. No training cycle, no
  retraining schedule, no OOS contamination risk from a stale model.
- If IC-weighted ensemble demonstrates meaningful edge, we layer LightGBM
  on top as Phase 4.3 — with the IC-weighted output serving as a control.

### 1.2 IC weighting method: raw IC, not IC rank

**Decision:** Weight proportional to rolling IC magnitude, not IC rank.

**Rationale:**
- Rank compression (highest IC = rank N, second = rank N-1) loses the
  magnitude gap between a strong signal (IC=0.08) and a weak one (IC=0.02).
- Raw IC weighting preserves that gap: the strong signal gets 4x the weight.
- Negative IC signals receive zero weight (we don't short signals that
  predict backwards — the signal logic is already directional).
- This matches practitioner approaches (AQR, WorldQuant) where IC magnitude
  drives allocation directly.

### 1.3 Rolling IC window: 63 trading days

**Decision:** Compute rolling IC over 63 trading days (~3 months), rebalancing
weights weekly (every 5 trading days).

**Rationale:**
- 63 days balances responsiveness with statistical stability. A shorter
  window (e.g., 21 days) produces noisy IC estimates with only ~5 tickers.
- Weekly rebalancing avoids excessive turnover while still adapting to
  regime changes. Daily rebalancing is too frequent for reliable IC estimates.
- With 5 tickers × 63 days = 315 observations per window, the IC estimate
  has reasonable statistical power for rank correlation.

### 1.4 Target horizon: 5-day forward return

**Decision:** The ensemble weights against `forward_return_5` as the primary
target horizon.

**Rationale:**
- 1-day forward returns are dominated by microstructure noise.
- 21-day returns capture too much macro information not attributable to our
  signals.
- 5-day horizon matches the sweet spot where cross-sectional alpha factors
  typically express their predictive power.
- Individual signals targeting different horizons are still valid — the
  ensemble simply weights them by their 5-day IC.

---

## 2. Architecture

### 2.1 Data Flow

```
Feature-Enriched DataFrame (Phase 2 output)
    |
    v
Signal Pipeline (Phase 3)
    |
    v
Signal-Enriched DataFrame (signal_<name> columns)
    |
    v
Forward Returns (forward_return_<h> columns)
    |
    v
+---------------------------+
|  Ensemble Module (Phase 4) |
|                           |
|  1. Rolling IC computation |
|     - per signal, per window|
|  2. Weight computation     |
|     - IC → weight mapping  |
|     - normalization        |
|  3. Signal combination     |
|     - weighted sum         |
|  4. Output: signal_ensemble|
+---------------------------+
    |
    v
Ensemble-Enriched DataFrame
    |
    v
Validation (combined IC vs individual IC)
```

### 2.2 Component Map

```
src/ensemble/
    __init__.py              # Package exports
    base.py                  # EnsembleMeta, data classes, IC-to-weight utils
    ic_weighted.py           # IC-weighted linear ensemble
    pipeline.py              # Ensemble pipeline orchestrator
    validation.py            # Ensemble validation (combined IC, attribution)

tests/
    test_ensemble.py         # Ensemble tests (synthetic + real data)
```

### 2.3 Interface Contract

The ensemble follows the same I/O convention as signals:

**Input:** `pl.DataFrame` sorted by `(ticker, date)` containing:
- All `signal_<name>` columns from the signal pipeline
- Forward return columns (`forward_return_<h>`) for IC computation
- Standard columns: `ticker`, `date`, `close`, etc.

**Output:** `pl.DataFrame` with an additional column:
- `signal_ensemble` — combined signal bounded in `[-1, +1]`
- `ensemble_weight_<name>` — per-date weight for each signal (for audit)

**Sign convention:** Same as individual signals:
- `+1` → long bias (expect price to rise)
- `0` → neutral
- `-1` → short bias (expect price to fall)

---

## 3. Detailed Design

### 3.1 Rolling IC Computation

For each signal and each date, compute the rank IC over the trailing
63-day window against the 5-day forward return:

```python
@dataclass
class RollingICResult:
    """Result of rolling IC computation for one signal."""
    signal_name: str          # e.g., "signal_mean_reversion_21d"
    dates: list[date]         # all dates in the window
    ic_values: list[float]    # IC per date (one per rolling window)
    overall_ic: float         # IC over the full dataset (for reference)
```

**Algorithm:**
1. For each date `t`, extract the 63-day trailing window `[t-63, t]`
2. Within that window, compute rank IC between `signal_<name>` and
   `forward_return_5` across all `(ticker, date)` observations
3. Store the IC value for date `t`
4. Repeat for all signals

**Edge cases:**
- If fewer than 30 observations exist in a window, IC defaults to 0.0
  (insufficient data for reliable correlation)
- NaN signal values are excluded from the IC computation (not imputed)

### 3.2 IC-to-Weight Mapping

Convert rolling IC to portfolio weights:

```python
def ic_to_weights(ic_dict: dict[str, float]) -> dict[str, float]:
    """Convert IC values to normalized weights.

    Rules:
        - IC > 0 → weight proportional to IC
        - IC <= 0 → weight = 0 (drop the signal)
        - Weights sum to 1.0
        - If all ICs are non-positive, equal-weight fallback
    """
    positive = {k: v for k, v in ic_dict.items() if v > 0}
    if not positive:
        # Fallback: equal weight across all signals
        n = len(ic_dict)
        return {k: 1.0 / n for k in ic_dict}
    total = sum(positive.values())
    return {k: v / total for k, v in positive.items()}
```

**Weight rebalancing schedule:**
- Weights are recomputed every 5 trading days (weekly)
- Between rebalances, the last computed weights are held constant
- This produces a step function of weights, not a smooth curve

### 3.3 Signal Combination

Combine individual signals using the current weights:

```python
def combine_signals(
    df: pl.DataFrame,
    signal_cols: list[str],
    weights: dict[str, float],
    out_col: str = "signal_ensemble",
) -> pl.DataFrame:
    """Weighted sum of signal columns.

    For each row: ensemble = sum(weight_i * signal_i) for all signals.
    Result is naturally bounded in [-1, +1] because individual signals
    are bounded and weights sum to 1.0.
    """
    exprs = []
    for col in signal_cols:
        w = weights.get(col, 0.0)
        if w > 0:
            exprs.append(pl.lit(w) * pl.col(col))

    if not exprs:
        return df.with_columns(pl.lit(0.0).alias(out_col))

    # Sum the weighted signals, filling NaN with 0
    combined = functools.reduce(
        lambda a, b: a.fill_null(0.0) + b.fill_null(0.0),
        [pl.lit(w) * pl.col(col) for col, w in weights.items() if w > 0],
    )
    return df.with_columns(combined.alias(out_col))
```

### 3.4 Ensemble Metadata Tracking

For auditability, the ensemble outputs weight history:

```python
@dataclass
class EnsembleMeta:
    """Metadata for the ensemble at a given date."""
    date: date
    weights: dict[str, float]        # signal_name → weight
    ic_snapshot: dict[str, float]    # signal_name → rolling IC
    signal_count: int                # number of signals contributing
    effective_ic: float              # IC of the ensemble itself
```

Each rebalance date produces an `EnsembleMeta` snapshot. These are stored
as a list for later analysis (attribution reports, regime analysis).

---

## 4. Configuration

### 4.1 Configurable Parameters

```python
@dataclass(frozen=True)
class EnsembleConfig:
    """Configuration for the IC-weighted ensemble."""

    # IC computation
    ic_lookback: int = 63           # rolling window for IC (trading days)
    ic_target_horizon: int = 5      # forward return horizon for IC target
    ic_min_observations: int = 30   # minimum observations for valid IC

    # Weight rebalancing
    rebalance_frequency: int = 5    # days between weight updates

    # Signal selection
    signal_columns: list[str] | None = None  # None = all signal_* columns
    min_signal_ic: float = 0.0      # minimum IC to include a signal

    # Output
    output_column: str = "signal_ensemble"
    track_weights: bool = True      # emit ensemble_weight_* columns
```

### 4.2 Default Configuration

The defaults encode the design decisions from Section 1:

| Parameter | Default | Rationale |
|---|---|---|
| `ic_lookback` | 63 | 3 months of IC history for stability |
| `ic_target_horizon` | 5 | 5-day expression window |
| `rebalance_frequency` | 5 | Weekly rebalance |
| `min_signal_ic` | 0.0 | Drop signals with non-positive IC |

---

## 5. Validation Design

### 5.1 Ensemble vs Individual Signal Comparison

The validation module compares:
- **Ensemble IC** vs each **individual signal IC** (same horizon)
- **Ensemble win rate** vs each **individual signal win rate**
- **Ensemble IC decay** across horizons (1d, 5d, 21d, 63d)

**Success criteria:**
- Ensemble IC must exceed the mean individual IC
- Ensemble IC should be within 20% of the best individual IC
  (i.e., the ensemble doesn't dilute the best signal too much)
- Ensemble win rate should exceed the median individual win rate

### 5.2 Weight Attribution Report

For each validation period:
- Show which signals contributed positive weight
- Show which signals were dropped (IC <= 0)
- Show weight concentration (Gini coefficient of weight distribution)
- Flag if a single signal dominates (>50% weight)

### 5.3 Stability Checks

- **IC stability:** rolling IC should not swing from +0.05 to -0.05 within
  one window (indicates noisy signal, not robust edge)
- **Weight stability:** weight changes between rebalances should be tracked;
  large swings (>20% shift in one rebalance) are flagged

---

## 6. Implementation Plan

### 6.1 Phase 4.1: Architecture Design (this task)
- ✅ Produce PHASE4-SPEC.md with interface, algorithm, and config design
- ✅ Define validation criteria
- ✅ Specify file structure

### 6.2 Phase 4.2: IC-weighted ensemble implementation (next task)
- `base.py` — `EnsembleMeta`, `EnsembleConfig`, `ic_to_weights()`
- `ic_weighted.py` — rolling IC, weight computation, signal combination
- `pipeline.py` — `run_ensemble(df, config)` function
- `__init__.py` — package exports
- Tests: synthetic data + real gold data

### 6.3 Phase 4.3: LightGBM meta-learner (future, conditional)
- Only after IC-weighted ensemble demonstrates edge
- Requires Phase 6 validation infrastructure (walk-forward OOS split)
- Uses IC-weighted output as a baseline control

### 6.4 Phase 4.4: Ensemble validation suite (parallel with 4.2)
- `validation.py` — IC comparison, attribution report, stability checks
- `test_ensemble.py` — full test suite

---

## 7. File Structure

```
src/ensemble/
    __init__.py              # Package exports
    base.py                  # EnsembleMeta, EnsembleConfig, ic_to_weights()
    ic_weighted.py           # RollingIC, ICWeightedEnsemble class
    pipeline.py              # run_ensemble(df, config) entry point
    validation.py            # ensemble_vs_individual(), attribution_report()

tests/
    test_ensemble.py         # IC-weighted ensemble tests
```

---

## 8. Dependencies

### 8.1 Upstream (hard)
- Phase 2 feature pipeline (feature-enriched DataFrame input)
- Phase 3 signal pipeline (signal columns present)
- Forward returns computed (for IC calculation)

### 8.2 Downstream (consumers)
- Phase 5 portfolio construction (uses `signal_ensemble` as input)
- Phase 6 validation engine (walk-forward analysis on ensemble output)

### 8.3 New Python Dependencies (none for Phase 4.2)
- IC-weighted ensemble uses only polars + standard library
- LightGBM (Phase 4.3) would add `lightgbm>=4.0` dependency

---

## 9. I/O Examples

### 9.1 Input DataFrame (signal-enriched with forward returns)

```
ticker  date        close   signal_mean_reversion_21d  signal_momentum_21d  forward_return_5
AAPL    2025-01-02  195.50  0.32                       0.15                 0.012
AAPL    2025-01-03  196.10  0.28                       0.18                 0.008
MSFT    2025-01-02  420.30 -0.45                      -0.22                 -0.005
...
```

### 9.2 Output DataFrame (ensemble-enriched)

```
ticker  date        signal_ensemble  ensemble_weight_mean_reversion_21d  ensemble_weight_momentum_21d
AAPL    2025-01-02  0.271           0.60                                  0.40
AAPL    2025-01-03  0.268           0.60                                  0.40
MSFT    2025-01-02 -0.357           0.60                                  0.40
...
```

The ensemble value for AAPL on 2025-01-02 is:
`0.60 * 0.32 + 0.40 * 0.15 = 0.192 + 0.060 = 0.252`

The weight columns show the current rebalance snapshot. Weights are held
constant between rebalances.

---

## 10. Status

- [x] Architecture design (IC-weighted approach specified)
- [x] Interface contract defined
- [x] Configuration parameters specified
- [x] Validation criteria defined
- [x] IC-weighted ensemble implementation (Phase 4.2)
- [x] LightGBM meta-learner (Phase 4.3)
- [x] Validation suite (Phase 4.4)
- [x] Unified EnsemblePipeline orchestrator (IC + LightGBM + validation)
- [x] End-to-end integration tests (39 new tests, 113 total passing)

---

## 11. Success Criteria

Phase 4.1 (architecture design) is complete when:
- [x] PHASE4-SPEC.md exists with full interface specification
- [x] IC-weighted approach is justified vs alternatives
- [x] File structure is defined
- [x] Dependencies and I/O contracts are documented

Phase 4 (ensemble) is considered **initially complete** when:
- [x] IC-weighted ensemble produces `signal_ensemble` column
- [x] LightGBM meta-learner produces `ensemble_prediction` column
- [x] Unified EnsemblePipeline supports both modes via single `run()` entry point
- [x] Validation report shows weight attribution
- [x] All ensemble tests pass (113 tests, zero failures)
- [x] PHASE4-SPEC.md is updated with final status