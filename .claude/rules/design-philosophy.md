# Design Philosophy

## Core Principles

1. **Split-before-fit**: Raw data is split temporally before any feature extraction. The feature pipeline is fitted on training data only. This prevents data leakage.

2. **Bayesian smoothing over raw aggregates**: All group-level statistics use Bayesian smoothing to prevent small-sample memorization. Different modules use different smoothing factors calibrated to their sample sizes.

3. **Fallback chains for robustness**: When group-level data is insufficient, the system falls back through a hierarchy (e.g., event_zone → event → artist_zone → artist → global).

4. **Pydantic + Parquet dual schema**: Every data model has both runtime validation (Pydantic) and storage typing (PyArrow `parquet_schema()`). Both must stay in sync.

5. **Repository pattern for data access**: All Parquet I/O goes through repositories that handle partitioning, deduplication, and schema enforcement.

## Source Tree Policy

See `docs/ARCHITECTURE.md` for the full layer diagram and dependency rules.

**Dependency direction**: Code flows from external services → API clients → schemas → scrapers/ingestion → storage → normalization/preprocessing → ML pipeline. Each layer may only import from layers above it.

**Key invariant**: Nothing in `ml/` may be imported by `schemas/`, `scrapers/`, `storage/`, `normalization/`, or `preprocessing/`.

## Modification Policy

| Directory | Risk | Rule |
|-----------|------|------|
| `schemas/` | High | Changes cascade to storage, scrapers, ingestion, ML. Update `parquet_schema()` alongside Pydantic model. |
| `ml/training/` | High | Preserve split-before-fit invariant. Changes to `trainer.py` or `splitter.py` require leakage-guardian review. |
| `ml/features/` | Medium | New extractors must implement `FeatureExtractor` ABC. Register in `pipeline.py`. |
| `storage/` | High | Parquet schema changes affect stored data. Consider migration. |
| `scrapers/` | Medium | Anti-detection patterns are fragile. Test against live sites. |
| `config.py` | Medium | `MLConfig` is frozen — add fields with defaults only. |
