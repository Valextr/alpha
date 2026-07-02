# Alpha — Phase 1: Data Pipeline Specification

## Objective

Build a survivorship-bias-free data ingestion pipeline that delivers clean, adjusted daily bar data to a Parquet lakehouse queryable via DuckDB.

**Target:** 100+ symbols, 10+ years daily data, ready for feature engineering in Phase 2.

---

## 1. Data Sources

### Primary: Polygon.io

**Why:** Survivorship-bias-free, includes delisted tickers, corporate actions, clean splits/dividends. Community standard per r/algotrading.

**Plan:** Start with free tier for prototyping (limited to ~100 symbols, delayed data). Upgrade to basic plan ($29/mo) when moving to serious backtesting.

**API endpoints needed:**
- `/v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}` — Daily OHLCV
- `/v3/reference/tickers` — Ticker list (including delisted)
- `/v2/reference/dividends/{ticker}` — Dividend history
- `/v3/reference/tickers/{ticker}/actions` — Corporate actions (splits, mergers)

### Fallback/Supplement: yfinance

**Why:** Free, good for initial prototyping and testing pipeline logic before Polygon.io is set up.

**Caveat:** Has survivorship bias — dead tickers disappear. Only use for pipeline development, never for final backtests.

### Future: IB Historical Data

**Why:** Free with IB account, includes futures data.

**Caveat:** Rate-limited, requires TWS/Gateway running. Defer to Phase 7 (Paper Trading) when IB is set up.

---

## 2. Target Universe

### Initial Universe (Phase 1)

Start with liquid US equities only. Add futures in Phase 7.

| Category | Tickers | Rationale |
|---|---|---|
| S&P 500 constituents | ~500 | Core universe, liquid, diverse |
| Sector ETFs | XLF, XLE, XLK, XLV, etc. | Sector rotation signals |
| Broad market | SPY, QQQ, IWM | Benchmark + regime detection |

**Total:** ~100-150 symbols for Phase 1.

### Selection Criteria
- Average daily volume > 1M shares
- Market cap > $2B (mid/large cap only)
- No recent delistings or mergers
- 10+ years of continuous data

---

## 3. Schema Design

### Bronze Layer (Raw)

Raw data as received from API, no adjustments.

```python
# daily_bronze.parquet
{
    "ticker": str,           # e.g., "AAPL"
    "date": date,            # Trading date
    "open": float64,
    "high": float64,
    "low": float64,
    "close": float64,
    "volume": int64,
    "vwap": float64,         # If available
    "source": str,           # "polygon" | "yfinance" | "ib"
    "ingested_at": datetime  # When we fetched it
}
```

### Silver Layer (Adjusted)

Adjusted for corporate actions. This is the primary data layer for feature engineering.

```python
# daily_silver.parquet
{
    "ticker": str,
    "date": date,
    "open": float64,         # Adjusted
    "high": float64,         # Adjusted
    "low": float64,          # Adjusted
    "close": float64,        # Adjusted
    "volume": int64,
    "vwap": float64,
    "adj_close": float64,    # Forward-adjusted close
    "split_factor": float64, # Cumulative split adjustment
    "dividend_yield": float64, # Trailing 12-month dividend yield
    "is_market_date": bool,  # Was market open this day?
    "data_quality": str      # "good" | "gap" | "thin" | "suspicious"
}
```

### Gold Layer (Enriched)

Ready for signal generation. Includes cross-sectional features and quality flags.

```python
# daily_gold.parquet
{
    # All silver fields +
    "universe_date": bool,    # Is this ticker in the investable universe this day?
    "sector": str,            # GICS sector
    "industry": str,          # GICS industry
    "market_cap_bucket": str, # "mega" | "large" | "mid" | "small"
    "avg_volume_20d": float64, # Rolling 20-day avg volume
    "avg_volume_60d": float64, # Rolling 60-day avg volume
    "volume_ratio": float64,  # Today's volume / 60-day avg
}
```

---

## 4. Lakehouse Structure

```
data/
├── bronze/
│   ├── daily/
│   │   └── year=2015/
│   │       ├── ticker=AAPL/part-0.parquet
│   │       ├── ticker=MSFT/part-0.parquet
│   │       └── ...
│   ├── dividends/
│   │   └── ticker=AAPL/part-0.parquet
│   └── corporate_actions/
│       └── ticker=AAPL/part-0.parquet
├── silver/
│   └── daily/
│       └── year=2015/
│           ├── ticker=AAPL/part-0.parquet
│           └── ...
├── gold/
│   └── daily/
│       └── year=2015/
│           ├── ticker=AAPL/part-0.parquet
│           └── ...
└── _catalog/
    └── duckdb_config.sql
```

**Partitioning:** By `year` and `ticker`. This gives:
- Fast date range queries (year partition)
- Fast single-ticker queries (ticker partition)
- Manageable file sizes (~1 year × 1 ticker = ~252 rows, small but queryable)

**File format:** Parquet with Snappy compression (DuckDB default, fast decompression).

---

## 5. Ingestion Flow

### Pipeline Steps

```
1. Fetch ticker universe → ticker_universe.csv
2. For each ticker:
   a. Fetch daily bars (OHLCV) → bronze/daily/
   b. Fetch dividends → bronze/dividends/
   c. Fetch corporate actions → bronze/corporate_actions/
3. Apply adjustments (splits, dividends) → silver/daily/
4. Enrich with cross-sectional features → gold/daily/
5. Build DuckDB catalog → _catalog/duckdb_config.sql
```

### Idempotency

Each ingestion run must be idempotent:
- Check existing data before fetching
- Overwrite only new/updated data
- Log what was fetched vs skipped

### Error Handling

- Retry failed API calls (3 attempts, exponential backoff)
- Log gaps (missing dates, missing tickers)
- Flag suspicious data (volume spikes, price gaps > 20%)
- Never silently drop data — flag and continue

---

## 6. Validation Checks

### Data Quality

| Check | Threshold | Action |
|---|---|---|
| Missing dates | Any trading day gap | Flag in `data_quality` |
| Volume = 0 | On market day | Flag as "thin" |
| Price gap > 20% | Adj close vs prev adj close | Flag as "suspicious" |
| Negative prices | Any | Drop row, log warning |
| Future dates | Any date > today | Drop row, log error |

### Survivorship Bias Check

- Compare our ticker list against a known delisted ticker list
- Verify we have data for tickers that were delisted
- Log any missing delisted tickers

### Point-in-Time Correctness

- Verify no data exists after the ingestion timestamp
- For live data: only include data up to market close of previous day
- Never include "today's" data in backtest sets

---

## 7. DuckDB Catalog

Create a DuckDB database file with views that abstract the Parquet files:

```sql
-- _catalog/duckdb_config.sql

-- Bronze views
CREATE OR REPLACE VIEW bronze_daily AS
SELECT * FROM read_parquet('data/bronze/daily/**/*.parquet');

CREATE OR REPLACE VIEW bronze_dividends AS
SELECT * FROM read_parquet('data/bronze/dividends/**/*.parquet');

-- Silver views
CREATE OR REPLACE VIEW silver_daily AS
SELECT * FROM read_parquet('data/silver/daily/**/*.parquet');

-- Gold views
CREATE OR REPLACE VIEW gold_daily AS
SELECT * FROM read_parquet('data/gold/daily/**/*.parquet');

-- Convenience: universe view (only investable dates)
CREATE OR REPLACE VIEW universe_daily AS
SELECT * FROM gold_daily
WHERE universe_date = true
  AND data_quality = 'good';
```

This means all downstream code queries DuckDB views, never touches Parquet files directly.

---

## 8. Implementation Plan

### File Structure

```
src/data/
├── __init__.py
├── ingestion.py          # Main pipeline orchestrator
├── sources/
│   ├── __init__.py
│   ├── polygon.py        # Polygon.io API client
│   ├── yfinance.py       # yfinance fallback
│   └── base.py           # Abstract source interface
├── adjust.py             # Split/dividend adjustments
├── enrich.py             # Gold layer enrichment
├── validate.py           # Data quality checks
├── catalog.py            # DuckDB catalog management
└── config.py             # Data pipeline config
```

### Config

```python
# src/data/config.py

DATA_DIR = Path("data")
UNIVERSE_FILE = Path("data/universe.csv")
DUCKDB_FILE = Path("data/alpha.duckdb")

# Polygon.io
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
POLYGON_BASE_URL = "https://api.polygon.io"

# Ingestion defaults
DEFAULT_START_DATE = "2014-01-01"  # 10+ years back
DEFAULT_END_DATE = "2025-12-31"    # Leave 2026 for forward test
BATCH_SIZE = 100                   # Symbols per API batch
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5                    # Seconds between retries

# Quality thresholds
MAX_PRICE_GAP = 0.20              # 20% gap threshold
MIN_VOLUME = 0                    # 0 volume on market day = flag
```

### CLI Interface

```bash
# Fetch all data for configured universe
python -m src.data.ingestion fetch --all

# Fetch specific ticker
python -m src.data.ingestion fetch --ticker AAPL

# Update with latest data (only new dates)
python -m src.data.ingestion update

# Run validation checks
python -m src.data.validate check --all

# Rebuild DuckDB catalog
python -m src.data.catalog rebuild
```

---

## 9. Dependencies

```txt
# Core
polars>=1.0          # Data processing (replaces pandas)
duckdb>=1.0          # Query engine
pyarrow>=15.0        # Parquet I/O

# Data sources
polygon-api-client>=1.0  # Official Polygon.io SDK
yfinance>=0.2.40         # Fallback for prototyping
ib_insync>=0.9           # IB (Phase 7, install now for testing)

# Utilities
requests>=2.31           # HTTP client
tqdm>=4.66               # Progress bars
pydantic>=2.0            # Config validation
loguru>=0.7              # Logging
```

---

## 10. Milestones

| Milestone | Deliverable | Duration |
|---|---|---|
| M1 | yfinance pipeline working (prototyping) | 3-4 days |
| M2 | Polygon.io integration + bronze layer | 3-4 days |
| M3 | Silver layer (adjustments) | 2-3 days |
| M4 | Gold layer (enrichment) | 2-3 days |
| M5 | Validation + DuckDB catalog | 2 days |
| M6 | Full pipeline test (100+ symbols, 10+ years) | 2 days |

**Total:** ~2-3 weeks for Phase 1.

---

## 11. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Polygon.io free tier limits | Can't fetch full universe | Start with yfinance prototype, upgrade when needed |
| API rate limits | Slow ingestion | Batch requests, respect rate limits, add delays |
| Data quality issues | Garbage in, garbage out | Validation checks, quality flags, never silently drop |
| Corporate action edge cases | Misadjusted prices | Log all adjustments, verify against known splits |
| Storage growth | Disk space | Parquet compression is good (~10:1 ratio), monitor |

---

## 12. Success Criteria

Phase 1 is complete when:
- [ ] 100+ symbols with 10+ years of daily data ingested
- [ ] All data adjusted for splits and dividends
- [ ] Survivorship bias verified (delisted tickers present)
- [ ] DuckDB catalog queryable with no errors
- [ ] Validation checks pass with no critical issues
- [ ] Pipeline is idempotent and resumable
- [ ] Documentation complete (schema, usage, known issues)
