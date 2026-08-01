# Alpha

Quantitative trading system.

## Architecture

```
Data Pipeline → Feature Store → Signal Factory → Ensemble → Portfolio → Validation → Execution
```

### Stack

| Layer | Tool |
|---|---|
| Data ingestion | Polygon.io / IB TWS API |
| Storage | Parquet lakehouse + DuckDB |
| Feature engineering | Polars (point-in-time correct) |
| Signal generation | Composable modules |
| Ensemble | IC-validated weights |
| Portfolio | Position sizing + risk parity + caps |
| Validation | Walk-forward + perturbation tests |
| Execution | Interactive Brokers (`ib_insync`) |
| Monitoring | Custom analytics + dashboard |

## Quick Start

```bash
# Install uv (if not already)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies (creates .venv, installs everything)
uv sync --all-extras

# Run pipeline (default universe, 10+ years)
uv run python -m src.data.ingestion run

# Fetch specific tickers
uv run python -m src.data.ingestion run --tickers AAPL MSFT GOOGL --start 2020-01-01

# Query via DuckDB
uv run python -m src.data.catalog stats

# Validate data quality
uv run python -m src.data.validate gold

# Run tests
uv run pytest
```

## Status

- ✅ **Phase 0:** Project setup
- ✅ **Phase 1:** Data Pipeline
- ✅ **Phase 2:** Feature Store
- ✅ **Phase 3:** Signal Factory
- ✅ **Phase 4:** Ensemble & Weights
- ✅ **Phase 5:** Portfolio & Risk
- ✅ **Phase 6:** Validation Engine
- ✅ **Phase 7:** Paper Trading
- ✅ **Phase 8:** Forward Test
- ✅ **Phase 9:** Signal improvement
- ✅ **Phase 10:** Intraday pivot

### Current State

**Architecture:** Intraday bar-by-bar trading via systemd service.

**Deployment:**
- `alpha-intraday.timer` — Mon-Fri at 8:30 AM ET
- `alpha-intraday.service` — connects to IBKR Gateway, streams 1-minute bars, computes signals, executes trades until market close
- Business logic is available separately
- This repo contains the framework: execution engine, broker interface, data pipeline, monitoring, validation

### Known issues

- **Test suite OOM:** Running the full test suite triggers out-of-memory errors on machines with limited RAM. Memory guardrails are in place. Default suite: 518 tests pass in ~40s.

## License

Private