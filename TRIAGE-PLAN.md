# Alpha Project — Triage Plan

**Date:** 2026-07-18
**Based on:** Repository analysis report (t_3ce6c472), project plan (docs/PLAN.md)

---

## Purpose

Convert the analysis findings into an actionable, prioritized task backlog for the Alpha project. This plan categorizes all identified work, defines priority criteria, and sequences tasks so that blockers are resolved first before downstream phases can proceed.

---

## Priority Framework

Tasks are categorized by severity and urgency:

| Priority | Label | Definition | Examples |
|----------|-------|------------|----------|
| P0 | CRITICAL | Produces silently wrong results or blocks all downstream work | Data leakage bug, no data ingested |
| P1 | HIGH | Performance regression or architectural debt that compounds over time | map_elements, wrong partitioning |
| P2 | MEDIUM | Code quality, maintainability, test coverage gaps | Dead code, missing unit tests |
| P3 | LOW | Documentation, polish, ad-hoc tooling | README staleness, check_data.py path |
| P4 | FUTURE | New features not yet on the critical path | Polygon.io, Phase 3+ work |

---

## Work Categories

### 1. BUG FIXES (critical path)

Bugs that produce incorrect results or block progress.

#### P0: Fix per-ticker scope in price feature rolling operations

- **File:** src/features/price.py, src/features/volatility.py, src/features/volume.py
- **Problem:** All rolling operations (`shift`, `rolling_mean`, etc.) use `pl.col("close").shift(N)` which is a global DataFrame operation. If the DataFrame sort order changes, data leaks between tickers. This silently produces wrong feature values.
- **Impact:** Every feature computed on a multi-ticker DataFrame is wrong. The regime features already use `group_by("date").agg()` correctly — the bug is isolated to price, volatility, and volume modules.
- **Fix:** Scope all rolling operations with `.over("ticker")` or compute within a `group_by("ticker")` context.
- **Verification:** Run `test_features_batches.py` after data is available; verify feature values are identical regardless of DataFrame row order.

#### P0: Ingest initial data so tests can run

- **File:** src/data/ingestion.py (and the data/ directory itself)
- **Problem:** No data has ever been ingested. The `data/` directory doesn't exist. 6 of 8 tests fail because they can't find gold-layer parquet files. This blocks all downstream development.
- **Impact:** Nothing downstream can be tested or validated until data exists.
- **Fix:** Run the pipeline with a small universe (5-10 tickers) to verify the full bronze->silver->gold chain works end-to-end. Use `alpha-ingest --tickers AAPL,MSFT,GOOGL,AMZN,META`.
- **Verification:** `alpha-validate` passes; 6 integration tests pass.

#### P1: Fix multi-year bronze partitioning

- **File:** src/data/ingestion.py (`fetch_bronze_for_ticker`)
- **Problem:** Extracts the year from the first date in the result and saves ALL data to that single year partition. A ticker with 10+ years of daily data ends up in one year folder instead of being split across year partitions.
- **Impact:** Year partitioning is broken, which means the DuckDB catalog views and any time-based filtering that relies on partition pruning will be inefficient or wrong.
- **Fix:** Partition each ticker's data by actual year, not the first year. Same fix needed for silver and gold saves.
- **Verification:** After ingest, confirm parquet files exist under `data/bronze/daily/year=YYYY/` for multiple years.

### 2. PERFORMANCE & ARCHITECTURE DEBT

Issues that work correctly now but will degrade as the dataset grows.

#### P1: Replace map_elements in enrich.py

- **File:** src/data/enrich.py (line 118)
- **Problem:** `pl.col("ticker").map_elements(bucket, ...)` is the slowest operation in Polars (Python-side row iteration). At 50K+ rows this is noticeably slow and the function is deprecated.
- **Impact:** Enrichment step will become a bottleneck as the universe grows beyond 58 tickers.
- **Fix:** Replace with a join to a market-cap mapping DataFrame or use Polars `replace` strategy.
- **Verification:** Benchmark enrichment step before and after; expect 10-100x speedup.

#### P2: Clean dead code in enrich.py

- **File:** src/data/enrich.py (lines 46-68)
- **Problem:** ~20 lines of abandoned code from a first attempt at `enrich_with_sector` using `when/then` chains with a dummy `.over([])`. The actual working implementation uses a join on lines 61-68.
- **Impact:** Confusion for future readers; no functional impact.
- **Fix:** Remove the dead `when/then` block (lines 46-60).
- **Verification:** Code review; no behavior change expected.

### 3. TEST COVERAGE

Gaps in the test suite that prevent confident iteration.

#### P2: Add synthetic data tests for data pipeline

- **File:** tests/conftest.py (new), tests/test_data_pipeline.py (new)
- **Problem:** Only integration tests exist, which require actual data on disk. The data pipeline modules (adjust.py, enrich.py, validate.py, catalog.py) have zero test coverage.
- **Impact:** Cannot verify data pipeline transformations without running the full ingestion pipeline. Refactoring is risky.
- **Fix:** Create a `tests/conftest.py` with fixtures that generate synthetic OHLCV data (small universes, known splits/dividends). Write unit tests for each data module.
- **Verification:** All data pipeline modules testable without network access or disk I/O.

#### P2: Verify feature tests after data is available

- **File:** tests/test_features_batches.py, tests/test_advanced_features.py
- **Problem:** 6 integration tests currently fail because no data exists. Once data is ingested, these tests should pass if the per-ticker scope fix is applied.
- **Impact:** Cannot verify feature correctness until both P0 fixes are in.
- **Fix:** After P0 fixes, re-run the test suite.
- **Verification:** All 8 tests pass (2 registry + 6 data-dependent).

### 4. DOCUMENTATION & POLISH

Stale or misleading documentation, ad-hoc tooling improvements.

#### P3: Update README with Phase 2 completion status

- **File:** README.md
- **Problem:** README says "20/30+ features" but 37 features are implemented across all 6 categories. Phase 2 should be marked as substantially complete.
- **Impact:** Misleading project status for anyone reading the repo.
- **Fix:** Update feature count to 37; mark Phase 2 as "Nearly Complete" with the per-ticker scope bug noted as the remaining blocker.

#### P3: Fix check_data.py relative path

- **File:** check_data.py
- **Problem:** Connects to `data/alpha.duckdb` using a relative path that only works from the project root.
- **Impact:** Script breaks if run from any other directory.
- **Fix:** Use `Path(__file__).resolve().parent / "data" / "alpha.duckdb"` or move the script into `src/data/`.

### 5. FUTURE WORK (Phase 2 completion + Phase 3 prep)

New features not on the critical path for immediate progress.

#### P4: Implement Polygon.io source

- **File:** src/data/sources/polygon.py (new)
- **Problem:** yfinance has survivorship bias and is explicitly prototyping-only. The Polygon source module doesn't exist yet.
- **Impact:** Data quality ceiling for production use.
- **Fix:** Implement `PolygonDataSource` following the `DataSource` ABC pattern.
- **Prerequisite:** Needs a Polygon.io API key and account.

#### P4: Start Phase 3 — Signal Factory design

- **File:** docs/PHASE3-SPEC.md (new), src/signals/ (new directory)
- **Problem:** Phase 3 is completely absent. The system produces features but has no way to turn them into trading signals.
- **Impact:** Cannot proceed with backtesting or validation.
- **Fix:** Design the signal interface (standardized input/output schema). Implement first 2-3 signal modules (Mean Reversion, Momentum).
- **Prerequisite:** Requires P0 fixes to be complete so features are actually correct.

#### P4: Implement pipeline resumption logic

- **File:** src/data/ingestion.py
- **Problem:** `run_pipeline` always fetches everything from scratch. The spec says the pipeline should be "idempotent and resumable."
- **Impact:** Re-running after a failure repeats all network calls.
- **Fix:** Track which tickers were already fetched and only re-fetch missing or stale ones.

---

## Execution Order (Critical Path)

The tasks should be executed in this sequence to unblock downstream work as fast as possible:

```
1. Fix per-ticker scope (P0 bug fix)
   |
   +--> 2. Fix multi-year partitioning (P1)
   |        (fix before first ingest so data lands correctly)
   |
   +--> 3. Ingest initial data (P0 blocker)
   |
   +--> 4. Re-run all tests (P2 verification)
   |
   +--> 5. Replace map_elements (P1)
   |        (can be done in parallel with #3)
   |
   +--> 6. Dead code cleanup (P2)
   |        (low effort, can be done anytime)
   |
   +--> 7. Synthetic data tests (P2)
   |
   +--> 8. README update + check_data.py fix (P3)
   |
   +--> 9. Polygon.io source (P4)
   |
   +--> 10. Phase 3 Signal Factory (P4)
```

Note: Tasks 5-8 can be done in parallel or interleaved with task 3's data ingestion, since they don't depend on each other.

---

## Task Assignment Strategy

| Category | Recommended assignee | Rationale |
|----------|---------------------|-----------|
| Bug fixes (P0) | venus | Needs deep code understanding; Venus already analyzed the repo |
| Performance (P1) | venus | Requires Polars expertise; small scope |
| Test coverage (P2) | venus | Requires understanding of existing code; synthetic data generation |
| Documentation (P3) | venus | Low effort; Venus has context |
| Future work (P4) | venus | Design decisions needed; Venus knows the architecture |

---

## Success Criteria

The triage is complete when:

1. The P0 per-ticker scope fix is merged and verified
2. Initial data is ingested and all 8 tests pass
3. The multi-year partitioning bug is resolved
4. All remaining issues are tracked as discrete Kanban tasks with clear acceptance criteria
5. The project can proceed to Phase 3 without data-quality blockers

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| yfinance rate limits block full ingest | High | Medium | Start with 5-ticker universe; use retry logic |
| Per-ticker fix changes feature values for existing analysis | Medium | High | No existing analysis exists yet (greenfield) |
| Polygon.io API costs | Low | Medium | yfinance is sufficient for prototyping through Phase 3 |
| Feature bugs surface after Phase 3 signals are built | Medium | High | Synthetic tests + verification before Phase 3 |