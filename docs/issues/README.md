# Issue & Decision Log

This directory is the project's **architecture-decision / obstacle log** — the durable
record of non-obvious problems, failed and successful experiments, and the reasoning
behind them. Each entry follows the format defined in [`CLAUDE.md`](../../CLAUDE.md)
(*Problem · Impact · Root Cause · Solution · Outcome*) with a **Status** and **Severity**.

New entries: use the next number as `NNN-short-description.md`.

## Index

| # | Title | Severity | Status |
|---|-------|----------|--------|
| [001](001-data-leakage-in-training-pipeline.md) | Data leakage in training pipeline | Critical | ✅ Resolved |
| [002](002-zone-mapping-misclassification.md) | Zone mapping misclassification (sections 400–499) | High | ✅ Resolved |
| [003](003-artist-name-normalization.md) | Artist name normalization | Medium | ✅ Resolved |
| [004](004-deduplication-hurts-performance.md) | Deduplication hurts model performance | Medium | ✅ Resolved (kept duplicates) |
| [005](005-listing-features-add-noise.md) | Listing-level features add noise | Medium | ✅ Resolved (features disabled) |
| [006](006-high-value-ticket-prediction-error.md) | High-value ticket prediction error (Q4 quartile) | High | ⚠️ Open (known limitation) |
| [007](007-segment-aware-outlier-capping-regression.md) | Segment-aware outlier capping regression | Medium | ✅ Resolved (reverted) |
| [008](008-dataset-size-bottleneck.md) | Dataset size as primary improvement bottleneck | High | ⚠️ Open (ongoing collection) |
| [009](009-city-name-normalization.md) | City name normalization inconsistencies | Medium | ✅ Resolved |
| [010](010-training-improvement-v29.md) | Training improvement — v29 (section feature) | — | ✅ Passed (feature kept) |
| [011](011-v30-pipeline-improvements.md) | v30 pipeline improvements — serialization, log-transform fix | High | ✅ Resolved |
| [012](012-v38-seen-event-mae.md) | v38 — seen-event MAE reduction (45% target) | High | 🔄 In Progress |

## Open items

- **[006](006-high-value-ticket-prediction-error.md)** — Q4 ($440+) tickets carry ~10× the
  MAE of the cheapest quartile; tail predictions remain the dominant error source.
- **[008](008-dataset-size-bottleneck.md)** — model quality is bottlenecked by data coverage
  (unseen events), not model capacity; addressed by ongoing hourly collection.
- **[012](012-v38-seen-event-mae.md)** — active work to drive seen-event MAE down toward the
  45% target.

## Related knowledge

- [`../ARCHITECTURE.md`](../ARCHITECTURE.md) — layer diagram and dependency rules
- [`../../.claude/rules/design-philosophy.md`](../../.claude/rules/design-philosophy.md) — split-before-fit, Bayesian smoothing, fallback chains
- [`../model-card.md`](../model-card.md) — current model performance and limitations
- [`../../CLAUDE.md`](../../CLAUDE.md) — "Key Findings" section summarizes experiment outcomes
