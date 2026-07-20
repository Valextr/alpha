# Phase 7: Interactive Brokers Paper Trading Setup

**Date:** 2026-07-19
**Status:** Requires human action (account creation)

---

## Overview

Phase 7 connects the Alpha system to Interactive Brokers for paper trading. This bridges the gap between backtesting (Phase 4-6) and live deployment.

Architecture:
```
Signals -> Ensemble -> Portfolio -> ExecutionEngine -> ib_async -> IB Gateway -> IBKR
```

---

## Prerequisites (Human Action Required)

### 1. Open a Live IBKR Account

Paper trading is **not available** without an approved live account. You do NOT need to fund it, but you must complete the application.

1. Go to https://www.interactivebrokers.com/en/open-account/
2. Complete the application (identity verification, risk questionnaire)
3. Wait for approval (typically 1-2 business days)
4. You do **not** need to deposit funds for paper trading

### 2. Request a Paper Trading Account

Once your live account is approved:

1. Log in to Client Portal
2. Navigate to Account Management -> Paper Trading Account
3. Click "Request Paper Trading Account"
4. You'll receive a separate paper trading username

### 3. Download IB Gateway (Linux)

IB Gateway is the lightweight, headless alternative to full TWS. Preferred for server/container deployments.

1. Log in to Client Portal
2. Go to Downloads -> Gateway for API
3. Download the Linux version (`.tar.gz`)
4. Extract to a persistent location (e.g., `~/ibkr/ibgateway/`)

**Download URL (after login):** `https://www.interactivebrokers.com/en/trading/ibgateway-latest.php`

---

## Technical Setup

### 1. Install Dependencies

Already done — `ib_async` is installed:

```bash
cd ~/working/alpha
uv add "ib_async>=2.0"
```

**Note:** The original `ib_insync` package is archived. We use `ib_async` (maintained fork) instead.

### 2. Configure IB Gateway for API Access

After installing IB Gateway:

1. Launch IB Gateway: `~/ibkr/ibgateway/vXX.X/ibgateway.sh`
2. Login with your **paper trading** username/password
3. Go to Configure -> Global Settings -> API -> Settings
4. Enable:
   - [x] "Enable ActiveX and Socket Clients"
   - Socket port: **4001** (default)
   - [x] "Allow connections from localhost only" (or add your IP)
5. Under "Trusted IPs", add `127.0.0.1` (and your machine's IP if connecting remotely)
6. Check "Use remote account" if needed

### 3. Environment Variables

Copy `.env.template` to `.env` and fill in your credentials:

```bash
cp .env.template .env
```

Required variables:
- `IBKR_USERNAME` — Paper trading username
- `IBKR_PASSWORD` — Paper trading password
- `IBKR_HOST` — Host (default: `127.0.0.1`)
- `IBKR_PORT` — API port (default: `4001`)
- `IBKR_TRADE_MODE` — `PAPER` or `LIVE` (start with `PAPER`)

### 4. Verify Connection

Run the connection test:

```bash
cd ~/working/alpha
python -c "
from src.execution.ibkr_client import IBKRClient
client = IBKRClient()
client.connect()
print('Connected:', client.connected())
print('Account:', client.account_summary())
client.disconnect()
"
```

---

## IB Gateway vs TWS

| Feature | IB Gateway | TWS |
|---------|-----------|-----|
| Resource usage | Low (~200 MB RAM) | High (~2 GB RAM) |
| GUI | Minimal | Full trading platform |
| Headless | Yes (with xvfb) | Yes (with xvfb) |
| API support | Full | Full |
| Manual trading | No | Yes |
| Recommended for | API/bot trading | Manual + API trading |

**Recommendation:** Use IB Gateway for this project. Lower resource footprint, same API surface.

---

## Headless Linux Setup (Optional)

For fully headless operation in a container or server:

1. Install Xvfb: `sudo dnf install xorg-x11-server-Xvfb`
2. Start virtual display: `Xvfb :1 -screen 0 1024x768x24 &`
3. Run IB Gateway: `DISPLAY=:1 ~/ibkr/ibgateway/vXX.X/ibgateway.sh`
4. For auto-login, use IBController or the `ibga` Docker image

**Docker alternative:** https://github.com/heshiming/ibga

---

## Security Considerations

1. **Never commit `.env`** — it contains credentials
2. Paper trading passwords differ from live passwords
3. Use separate API client IDs if running multiple connections
4. IBKR logs all API activity — monitor for unexpected connections
5. Enable two-factor authentication on the live account (not typically required for paper)

---

## ib_async Library Notes

- `ib_async` is the maintained fork of the original `ib_insync` (which is archived)
- Connection is TCP socket on port 4001
- All state (positions, orders, fills) is kept in sync automatically
- Paper trading uses the exact same API as live trading
- GitHub: https://github.com/ib-api-reloaded/ib_async

### Common Gotchas

- IB Gateway must be running BEFORE your Python code connects
- First connection after IB Gateway launch takes ~30 seconds (warm-up)
- Market data subscriptions are per-symbol (rate limits apply)
- Paper trading has simulated slippage but does model realistic fills
- Historical data limits: paper trading may have reduced data depth vs live

---

## Next Steps

After completing the human steps above:
1. Fill in `.env` with paper trading credentials
2. Launch IB Gateway
3. Run the connection test
4. Phase 7.2 (execution engine) can proceed
5. Phase 7.3 (monitoring dashboard) depends on live data flowing