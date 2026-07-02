# Alpha

Greenfield quantitative trading system. Built from scratch, not forked.

## Philosophy

- **Understand every line** — no inherited architecture, no black boxes
- **Modern tooling** — Polars, DuckDB, LightGBM over legacy pandas/sklearn
- **Overfitting is the enemy** — 3-part data segmentation, walk-forward validation, parameter perturbation
- **Forward-test before trusting** — 6+ months paper trading on IB before any real capital
- **Lumpy equity curves are honest** — smooth returns mean overfitting

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
| Signal generation | Composable modules (8+ signals) |
| Ensemble | LightGBM meta-learner + IC-validated weights |
| Portfolio | Kelly criterion + risk parity + position caps |
| Validation | Walk-forward + 3-part split + perturbation tests |
| Execution | Interactive Brokers (`ib_insync`) |
| Monitoring | Custom analytics + forward-test dashboard |

## Reference

`gurmansaran/medallion-pub` serves as a reference library for:
- Kalman filter implementations (3 variants)
- Fractional differentiation (López de Prado)
- Walk-forward validation structure
- Kelly sizing implementation

Not forked. Not locked in. Just reference.

## Status

🟢 Phase 0: Project setup
⬜ Phase 1: Data Pipeline
⬜ Phase 2: Feature Store
⬜ Phase 3: Signal Factory
⬜ Phase 4: Ensemble & Weights
⬜ Phase 5: Portfolio & Risk
⬜ Phase 6: Validation Engine
⬜ Phase 7: Paper Trading
⬜ Phase 8: Forward Test (6+ months)

## License

Private — personal research project.
