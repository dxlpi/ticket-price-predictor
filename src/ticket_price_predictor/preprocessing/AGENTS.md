# Preprocessing Module — Agent Reference

A modular, composable pipeline for cleaning, validating, and transforming the three
core data types: **listings**, **events**, and **snapshots**. Compass only — read the
source for signatures and edge cases.

## Architecture

Three phases, run in sequence by a `PreprocessingPipeline`:

1. **Cleaners** (`cleaners.py`) — detect and *flag* quality issues (never drop rows)
2. **Validators** (`validators.py`) — check schema, referential integrity, temporal sanity
3. **Transformers** (`transformers.py`) — enrich (joins, imputation, type/zone/temporal features)

Every stage implements the `Preprocessor` ABC (`base.py`): `process(df) -> ProcessingResult`,
where `ProcessingResult` carries `data`, `issues`, and `metrics`. The pipeline aggregates
issues/metrics across stages and degrades gracefully — a failing stage logs and continues.

## Key files

| File | Owns |
|------|------|
| `base.py` | `Preprocessor` ABC, `PreprocessingPipeline`, `ProcessingResult` |
| `config.py` | `PreprocessingConfig` — tunable thresholds (IQR multiplier, price bounds, imputation) |
| `cleaners.py` | `TextNormalizer`, `PriceOutlierHandler`, `DuplicateHandler` |
| `validators.py` | `SchemaValidator`, `ReferentialValidator`, `TemporalValidator` |
| `transformers.py` | `EventMetadataJoiner`, `MissingValueImputer`, `TypeConverter`, `SeatZoneEnricher`, `TemporalFeatureEnricher` |
| `pipeline.py` | `PipelineBuilder` — preset (`listings`/`events`/`snapshots`) and custom pipelines |
| `quality.py` | `QualityReporter`, `QualityMetrics`, `QualityThresholds` — metrics + alert levels |

## Non-obvious rules

- **Cleaners flag, they don't drop.** `PriceOutlierHandler` and `DuplicateHandler` add
  boolean columns (`is_price_outlier`, `is_duplicate`); the caller decides what to filter.
- **Text normalization is additive.** `TextNormalizer` writes `*_normalized` columns and
  leaves originals intact, so downstream joins can choose either form.
- **Preprocessing runs AFTER the temporal split in training** (`ModelTrainer`), never before
  — applying it to the full frame first would leak. See [`../../../docs/issues/001-data-leakage-in-training-pipeline.md`](../../../docs/issues/001-data-leakage-in-training-pipeline.md).
- **`strict_mode=False` by default** — validators warn rather than abort. Set `strict_mode=True`
  only for one-off audits, not the training path.

## Common usage

```python
from ticket_price_predictor.preprocessing import PipelineBuilder, PreprocessingConfig

pipeline = PipelineBuilder.build_listings_pipeline(events_df=events_df, config=PreprocessingConfig())
result = pipeline.process(listings_df)
clean = result.data[~result.data["is_price_outlier"] & ~result.data["is_duplicate"]]
```

Integration hooks: `ListingCollector(preprocess=True)` and `ModelTrainer(preprocess=True)`;
bulk CLI at `scripts/preprocess_data.py`. Preset stage lists live in `pipeline.py`.

## Testing

See [`../../../tests/test_preprocessing.py`](../../../tests/test_preprocessing.py). Cover the
standard edge cases: empty frames, missing columns, type mismatches, >50% null columns, and
future/past-event timestamps (validators use a ±2-day tolerance).
