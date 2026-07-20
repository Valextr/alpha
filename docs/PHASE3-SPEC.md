# Alpha — Phase 3: Signal Factory Specification

## Objective

Build composable signal modules that transform feature-enriched OHLCV data
into directional trading signals. Each signal is independently validated via
Information Coefficient (IC) analysis, IC decay, and win rate metrics.

**Target:** 8 signal modules, all sharing a standardized interface and
independently auditable. Phase 3 ships 4 signals (mean reversion + momentum);
4 more planned for future iterations.

---

## 1. Architecture

```
Feature-Enriched DataFrame (Phase 2 output)
    |
    v
Signal Pipeline
    |
    +-- Mean Reversion Signals (2 variants)
    +-- Momentum Signals (2 variants)
    +-- [Volatility Regime — planned]
    +-- [Trend Filter — planned]
    +-- [Stationarity — planned]
    +-- [Cross-Asset Spread — planned]
    +-- [Volume Anomaly — planned]
    +-- [Regime-Specific — planned]
    |
    v
Signal-Enriched DataFrame
    |
    v
Forward Returns (for validation)
    |
    v
IC Analysis + Win Rate Report
```

### Data Flow

1. **Input:** Feature-enriched DataFrame sorted by `(ticker, date)`.
   Contains OHLCV + all Phase 2 features (57+ columns).
2. **Signal Generation:** Each signal function receives the current DataFrame
   state and returns an enriched copy with its signal column appended.
   Signals are applied in dependency order.
3. **Forward Returns:** Computed *after* signal generation to verify no
   look-ahead bias leaked into signal computation.
4. **Validation:** Per-signal IC, IC decay across horizons, and win rate
   computed against forward returns.

### Key Principles

- **Standardized I/O:** Every signal function takes a `pl.DataFrame` and
  returns a `pl.DataFrame` with a `signal_<name>` column appended.
- **Bounded output:** Signals are normalized to `[-1, +1]` via `tanh()` for
  comparability across signal types.
- **Sign convention:** `+1` → long bias, `0` → neutral, `-1` → short bias.
- **Auto-registration:** Signals register themselves at import via
  `@registry.register(...)` decorator.
- **Dependency ordering:** Pipeline applies signals in dependency order
  (leaf signals first).
- **Per-signal validation:** Each signal can be independently evaluated
  using IC analysis, IC decay, and win rate.

---

## 2. Signal Interface

### Signal Function Signature

```python
@registry.register(
    "signal_name",
    description="Human-readable description",
    category="category",
    parameters={"key": "value"},
    depends_on=[],           # Other signal names this depends on
    requires_features=["col"],  # Feature columns needed
)
def generate_signal_name(df: pl.DataFrame) -> pl.DataFrame:
    """Return df with signal_<name> column appended."""
    ...
```

### Signal Metadata

| Field | Type | Description |
|---|---|---|
| name | str | Unique signal identifier |
| description | str | Human-readable description |
| category | str | Signal category (mean_reversion, momentum, etc.) |
| parameters | dict | Free-form config dict |
| depends_on | list[str] | Other signal names required |
| requires_features | list[str] | Feature columns required from Phase 2 |

### Output Convention

Every signal appends a column named `signal_<name>` to the DataFrame:
- Values bounded roughly in `[-1, +1]`
- `+1` = long bias (expect price to rise)
- `0` = neutral
- `-1` = short bias (expect price to fall)

---

## 3. Registry API

Mirrors the `FeatureRegistry` pattern from Phase 2.

| Method | Description |
|---|---|
| `registry.register(...)` | Register a signal (returns decorator) |
| `registry.list_signals()` | All registered signal metadata |
| `registry.get_signal(name)` | Get signal metadata |
| `registry.get_generate_fn(name)` | Get the generation function |
| `registry.signals_by_category()` | Group signals by category |
| `registry.validate_dependencies()` | Check unresolved dependencies |
| `registry.reset()` | Clear all registered signals (for testing) |

---

## 4. Signal Pipeline

### Usage

```python
from src.signals.pipeline import generate_all, generate_all_with_forward_returns
from src.signals.base import compute_forward_returns, rank_ic, ic_decay, win_rate, signal_summary

# Generate all signals
df_with_signals = generate_all(df_with_features)

# Generate with forward returns for validation
df = generate_all_with_forward_returns(df_with_features, horizons=[1, 5, 21])

# Validate a signal
summary = signal_summary("signal_momentum_21d", df, horizons=[1, 5, 21])
ic = rank_ic("signal_momentum_21d", "forward_return_1", df)
decay = ic_decay("signal_momentum_21d", df)
wr = win_rate("signal_momentum_21d", "forward_return_1", df)
```

### Pipeline Functions

| Function | Description |
|---|---|
| `generate_all(df, categories=None)` | Generate all signals (or filtered subset) |
| `generate_all_with_forward_returns(df, ...)` | Signals + forward returns for validation |
| `compute_forward_returns(df, horizons)` | Attach `forward_return_<h>` columns |

---

## 5. Validation Metrics

### Information Coefficient (IC)

Rank IC (Spearman correlation) between signal values and forward returns
across all `(ticker, date)` observations. Measures how well the signal
ranks assets by future performance.

- `rank_ic(signal_col, target_col, df)` → float in `[-1, +1]`

### IC Decay

IC at each forward horizon (1d, 5d, 21d, 63d). Shows how predictive
power decays over time — a signal with fast decay is short-term alpha,
while slow decay suggests a persistent factor.

- `ic_decay(signal_col, df, horizons)` → dict mapping horizon → IC

### Win Rate

Fraction of rows where signal and forward return agree in sign.
A 50% win rate is random; above 50% indicates predictive power.

- `win_rate(signal_col, target_col, df)` → float in `[0, 1]`

### Signal Summary

Aggregates all metrics into a single report:

- `signal_summary(signal_col, df, horizons)` → dict with:
  - Basic stats: count, mean, std, min, max
  - Per-horizon: `ic_<h>d`, `win_rate_<h>d`

---

## 6. Implemented Signals

### 6.1 Mean Reversion (2 variants)

**Hypothesis:** Prices that deviate from their local mean tend to revert.

**Source:** López de Prado, "Advances in Financial Machine Learning"
(Ch. 10: Fractional Differentiation).

**Logic:**
1. Compute rolling z-score of price relative to its local mean
2. Negate (below mean → positive signal = long)
3. Apply `tanh()` to bound to `[-1, +1]`
4. Cross-sectionally rank each date's signals for inter-ticker comparability

| Signal | Lookback | Requires |
|---|---|---|
| `mean_reversion_21d` | 21 trading days | `log_return_1d` |
| `mean_reversion_63d` | 63 trading days | `log_return_1d` |

### 6.2 Momentum (2 variants)

**Hypothesis:** Cross-sectional momentum — stocks that outperformed peers
over the past N days tend to continue outperforming in the near term.

**Source:** Jegadeesh & Titman (1993) "Returns to Buying Winners and Selling Losers."

**Logic:**
1. Use pre-computed returns over formation period
2. Cross-sectionally rank returns per date
3. Convert rank to `[-1, +1]` via `tanh()`

| Signal | Formation | Requires |
|---|---|---|
| `momentum_21d` | 21 trading days | `return_21d` |
| `momentum_63d` | 63 trading days | `return_63d` |

---

## 7. Planned Signals

| Signal | Category | Method |
|---|---|---|
| Volatility Regime | regime | HMM on price/vol/volume |
| Trend Filter | trend | Kalman filter adaptive |
| Stationarity | stationarity | Fractional differentiation (López de Prado) |
| Cross-Asset Spread | pairs | Cointegration pairs (Engle-Granger) |
| Volume Anomaly | volume | Relative volume spike detection |
| Regime-Specific | meta | Conditional signal weights |

---

## 8. File Structure

```
src/signals/
    __init__.py           # Package exports + eager module imports
    base.py               # SignalMeta, forward returns, IC/win rate/validation
    registry.py           # SignalRegistry singleton + registry instance
    pipeline.py           # generate_all(), generate_all_with_forward_returns()
    mean_reversion.py     # Mean reversion signals (2 variants)
    momentum.py           # Momentum signals (2 variants)
    # (future: volatility_regime.py, trend_filter.py, etc.)

tests/
    test_signals.py       # 25 tests: registry, generation, pipeline, IC, real data
```

---

## 9. Testing

### Synthetic Data Tests

- Registry: signals registered, listable, categorizable, deps valid
- Generation: each signal produces its column, bounded in [-1, +1], has both signs
- Pipeline: `generate_all` produces all signal columns, category filter works
- Forward returns: columns produced, computed per-ticker (no cross-ticker leakage)
- IC validation: rank_ic returns valid number, ic_decay covers all horizons,
  win_rate returns valid number, signal_summary has all expected keys

### Real Data Tests

- Signals on gold layer data (2023, 5 tickers): all signal columns present,
  signals have meaningful variance (std > 0)
- IC on real data: IC values in [-1, +1] for all signals

---

## 10. Status

- [x] Signal interface (standardized input/output schema)
- [x] Signal registry with auto-registration
- [x] Pipeline orchestrator (dependency-ordered generation)
- [x] Forward return computation (per-ticker, no leakage)
- [x] Validation suite (IC, IC decay, win rate, signal summary)
- [x] Mean Reversion signals (2 variants: 21d, 63d)
- [x] Momentum signals (2 variants: 21d, 63d)
- [x] Test suite (25 tests, including real data integration)
- [x] Documentation (this spec)
- [ ] Volatility Regime signal (planned)
- [ ] Trend Filter signal (planned)
- [ ] Stationarity signal (planned)
- [ ] Cross-Asset Spread signal (planned)
- [ ] Volume Anomaly signal (planned)
- [ ] Regime-Specific signal (planned)

---

## 11. Success Criteria

Phase 3 is considered **initially complete** when:
- [x] Signal interface is standardized (consistent input/output schema)
- [x] At least 2 signal modules implemented (4 variants across 2 categories)
- [x] Per-signal validation working (IC analysis, IC decay, win rate)
- [x] PHASE3-SPEC.md documents the architecture

Phase 3 is **fully complete** when all 8 target signals are implemented
and validated.