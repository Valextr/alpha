"""Signal-to-order pipeline, position tracking, and fill reconciliation.

The execution engine sits between the ensemble/portfolio layer and the broker:

    Signals → Ensemble → Portfolio → ExecutionEngine → Broker → Market

Responsibilities:
- Convert portfolio targets into executable orders
- Track positions, fills, and P&L
- Reconcile broker fills against internal state
- Enforce risk guardrails (position limits, kill switch, daily P&L caps)
- Support paper trading simulation and live trading via Interactive Brokers
"""