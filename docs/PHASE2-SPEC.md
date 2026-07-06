# Alpha - Phase 2: Feature Store Specification

## Objective

Build a point-in-time correct feature engineering pipeline that transforms
gold-layer OHLCV data into a rich feature set for signal generation.
Zero look-ahead bias. Every feature auditable and reproducible.

**Target:** 6 feature categories, 30+ features, all validated.

---

## 1. Architecture

```
Gold Layer (OHLCV + metadata)
    |
    v
Feature Pipeline
    |
    +-- Price Features (14 features)
    +-- Volatility Features (6 features)
    +-- Volume Features (TBD)
    +-- Cross-Sectional Features (TBD)
    +-- Regime Features (TBD)
    +-- Advanced Features (TBD)
    |
    v
Feature-Enriched DataFrame
    |
    v
Signal Factory (Phase 3)
```

### Key Principles

- **Point-in-time correct:** All rolling windows look backward only
- **Polars-native:** No pandas, all transformations use Polars expressions
- **Feature registry:** Every feature has metadata (name, description, category, lookback, dependencies)
- **Auto-registration:** Features register themselves at module import via decorator
- **Dependency ordering:** Pipeline applies features in lookback order
- **Validation:** NaN rates, Inf values, constant columns, large values checked

---

## 2. Feature Registry

Every feature is registered with metadata:

```python
@registry.register(
    "return_1d",
    description="1-day simple return",
    category="price",
    lookback=1,
)
def compute_return_1d(df):
    ...
```

### Metadata Fields

| Field | Type | Description |
|---|---|---|
| name | str | Unique feature identifier |
| description | str | Human-readable description |
| category | str | Feature category (price, volatility, etc.) |
| lookback | int | Minimum rows needed before feature produces values |
| depends_on | list[str] | Feature names this feature requires |

### Registry API

- `registry.register(...)` - Register a feature (returns decorator)
- `registry.list_features()` - All registered features
- `registry.get_feature(name)` - Get feature metadata
- `registry.features_by_category()` - Group by category
- `registry.validate_dependencies()` - Check unresolved dependencies

---

## 3. Feature Categories

### 3.1 Price Features (14 features) - COMPLETE

| Feature | Description | Lookback | Formula |
|---|---|---|---|
| return_1d | 1-day simple return | 1 | (close / prev_close) - 1 |
| return_5d | 5-day simple return | 5 | (close / close_5d_ago) - 1 |
| return_21d | 21-day simple return | 21 | (close / close_21d_ago) - 1 |
| return_63d | 63-day simple return | 63 | (close / close_63d_ago) - 1 |
| log_return_1d | 1-day log return | 1 | ln(close / prev_close) |
| log_return_5d | 5-day log return | 5 | ln(close / close_5d_ago) |
| log_return_21d | 21-day log return | 21 | ln(close / close_21d_ago) |
| log_return_63d | 63-day log return | 63 | ln(close / close_63d_ago) |
| cum_return_5d | 5-day cumulative return | 5 | close / close_5d_ago |
| cum_return_21d | 21-day cumulative return | 21 | close / close_21d_ago |
| cum_return_63d | 63-day cumulative return | 63 | close / close_63d_ago |
| drawdown_from_peak | Drawdown from 252d peak | 252 | (close / rolling_max_252) - 1 |
| price_displacement_5d | 5-day abs displacement | 5 | \|close - close_5d_ago\| / close_5d_ago |
| price_displacement_21d | 21-day abs displacement | 21 | \|close - close_21d_ago\| / close_21d_ago |

### 3.2 Volatility Features (6 features) - COMPLETE

| Feature | Description | Lookback | Depends On |
|---|---|---|---|
| vol_5d | 5-day realized vol | 5 | log_return_1d |
| vol_21d | 21-day realized vol | 21 | log_return_1d |
| vol_63d | 63-day realized vol | 63 | log_return_1d |
| vol_annual | Annualized vol | 21 | vol_21d |
| vol_of_vol_21d | Vol of vol (regime stability) | 25 | vol_5d |
| vol_ratio_short_long | Short/long vol ratio | 63 | vol_5d, vol_63d |

### 3.3 Volume Features (planned)

- relative_volume_21d - Today's volume / 21-day avg
- volume_zscore_63d - Volume z-score over 63 days
- accumulation_distribution - Price-volume flow proxy
- volume_shock - Binary: volume > 2 std above 63d mean

### 3.4 Cross-Sectional Features (planned)

- cs_return_zscore_21d - Cross-sectional z-score of 21d returns
- cs_return_rank_21d - Cross-sectional rank of 21d returns
- cs_vol_rank_21d - Cross-sectional rank of 21d volatility
- cs_volume_rank_21d - Cross-sectional rank of relative volume
- sector_relative_return - Ticker return vs sector median

### 3.5 Regime Features (planned)

- regime_market_trend - S&P trend (SMA50 vs SMA200 proxy)
- regime_vol_state - Current vol vs 1-year vol percentile
- regime_breadth - % of universe above SMA20
- regime_vol_regime - HMM-derived regime (if advanced features built)

### 3.6 Advanced Features (planned)

- frac_diff_1d - Fractional differentiation (Lopez de Prado)
- kalman_alpha - Kalman filter alpha estimate
- kalman_beta - Kalman filter beta estimate
- hurst_63d - Rolling Hurst exponent (mean-reversion vs trending)

---

## 4. Pipeline

### Usage

```python
from src.features.pipeline import compute_features, compute_and_validate

# Compute all features
enriched = compute_features(df)

# Compute specific categories
enriched = compute_features(df, categories=["price", "volatility"])

# Compute with validation
enriched, validation = compute_and_validate(df)
```

### Dependency Resolution

Features are applied in lookback order (shortest first). This ensures
that dependencies are computed before features that depend on them.

### Validation

The `validate_features()` function checks:

- **NaN rates:** Flags columns with >1% NaN (expected for rolling features at start)
- **Inf values:** Flags any column with infinity values (should be 0)
- **Constant columns:** Flags columns with zero variance
- **Large values:** Flags columns with abs(values) > 1e6

---

## 5. File Structure

```
src/features/
    __init__.py          # Package exports
    registry.py           # FeatureRegistry, FeatureMeta, registry singleton
    base.py               # Feature ABC, safe_rolling, cross_sectional, validate
    price.py              # 14 price features + PRICE_FEATURES list
    volatility.py         # 6 volatility features + VOLATILITY_FEATURES list
    volume.py             # (planned) Volume features
    cross_sectional.py    # (planned) Cross-sectional features
    regime.py             # (planned) Regime features
    advanced.py           # (planned) Advanced features
    pipeline.py           # compute_features(), compute_and_validate()
```

---

## 6. Testing

### Point-in-Time Correctness

- All rolling operations use backward-looking windows only
- Cross-sectional features computed per-date group, never globally
- No forward fills or future data leakage

### Validation Tests

- NaN rates match expected lookback (e.g., 63d features on 250 rows = 25.2% NaN)
- No Inf values in any feature
- Feature values in reasonable ranges (returns < 1, vol < 0.5, etc.)

---

## 7. Status

- [x] Feature registry with auto-registration
- [x] Base utilities (safe_rolling, cross_sectional, validate)
- [x] Price features (14 features)
- [x] Volatility features (6 features)
- [x] Pipeline orchestrator
- [x] Volume features (4 features: relative_volume_21d, volume_zscore_63d, accumulation_distribution, volume_shock)
- [x] Cross-sectional features (5 features: cs_return_zscore_21d, cs_return_rank_21d, cs_vol_rank_21d, cs_volume_rank_21d, sector_relative_return_21d)
- [x] Regime features (4 features: regime_market_trend, regime_vol_state, regime_breadth, regime_vol_regime)
- [ ] Advanced features
- [ ] Full validation test suite

---

## 8. Success Criteria

Phase 2 is complete when:
- [ ] 30+ features across 6 categories
- [ ] All features auto-registered with metadata
- [ ] Point-in-time correctness verified
- [ ] Validation passes with no Inf/constant feature columns
- [ ] Pipeline processes full universe (58+ tickers, 10+ years) in < 30s
- [ ] Documentation complete (feature catalog, formulas, dependencies)
