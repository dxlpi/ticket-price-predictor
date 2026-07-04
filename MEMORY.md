# Project Memory

Durable, externalized tribal knowledge for the ticket-price-predictor — the decisions,
gotchas, and hard-won findings an agent or new contributor needs but can't infer from code.
This file is the **index**; the detail lives in the linked stores.

## Decision & obstacle log

The full architecture-decision record lives in [`docs/issues/`](docs/issues/README.md) —
12 numbered entries (*Problem · Impact · Root Cause · Solution · Outcome*). Start there for
"why is it built this way" questions.

## Non-obvious findings (the short list)

- **Data leakage is the cardinal risk.** Raw data is split temporally *before* any feature
  extraction; the pipeline is fitted on train only. See [`docs/issues/001-data-leakage-in-training-pipeline.md`](docs/issues/001-data-leakage-in-training-pipeline.md)
  and [`.claude/rules/design-philosophy.md`](.claude/rules/design-philosophy.md).
- **Some intuitive ideas measurably hurt.** Deduplication (−$6.79), segment-aware outlier
  capping (−$6.07), and listing-level features all regressed MAE and were removed —
  [`docs/issues/004`](docs/issues/004-deduplication-hurts-performance.md), [`007`](docs/issues/007-segment-aware-outlier-capping-regression.md), [`005`](docs/issues/005-listing-features-add-noise.md).
- **The error ceiling is data, not model.** Unseen events dominate error; all hyperparameter
  tuning converges to the same MAE floor — [`docs/issues/008-dataset-size-bottleneck.md`](docs/issues/008-dataset-size-bottleneck.md).
- **Layer invariant:** nothing in `ml/` may be imported by lower layers. See
  [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Where knowledge is externalized

- [`docs/issues/`](docs/issues/README.md) — decision & obstacle log (canonical)
- [`.claude/rules/`](.claude/rules/) — auto-loaded engineering rules (design philosophy, validation, conventions)
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — layer diagram and dependency direction
- [`docs/model-card.md`](docs/model-card.md) — model performance, limitations, training data
- [`CLAUDE.md`](CLAUDE.md) — project structure, patterns, and "Key Findings" summary
