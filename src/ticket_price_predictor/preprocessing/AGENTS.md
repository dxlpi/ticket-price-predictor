# Preprocessing Module - Agent Reference

## Module Overview

The preprocessing module provides a modular, composable pipeline for data cleaning, validation, and transformation. It handles three primary data types:

- **Listings** - VividSeats/StubHub ticket listings with prices and seat information
- **Events** - Concert/sporting event metadata (dates, venues, artists)
- **Snapshots** - Aggregated price snapshots at the seat zone level

The module follows a three-phase architecture:
1. **Cleaners** - Detect and flag data quality issues (outliers, duplicates, text normalization)
2. **Validators** - Check schema, referential integrity, and temporal consistency
3. **Transformers** - Enrich data (joins, imputation, type conversion, feature creation)

All preprocessors implement the `Preprocessor` abstract base class, enabling composition into `PreprocessingPipeline` instances.

## Architecture

### Core Classes

**Base Classes:**
- `Preprocessor` - Abstract base class for all preprocessor implementations
  - Method: `process(df: pd.DataFrame) -> ProcessingResult`
  - Returns: `ProcessingResult` with data, issues list, and metrics dict

- `ProcessingResult` - Immutable result container
  - `data: pd.DataFrame` - Output data
  - `issues: list[str]` - Issues/warnings raised
  - `metrics: dict[str, Any]` - Stage-specific metrics

**Configuration:**
- `PreprocessingConfig` - Dataclass with tunable thresholds
  - `iqr_multiplier: float = 1.5` - IQR coefficient for outlier detection
  - `price_min/max: float` - Absolute price bounds (1.0 - 50000.0)
  - `venue_capacity_default: int = 15000` - Fallback capacity
  - `imputation_strategy: str = "median"` - Imputation method
  - `normalize_case: bool = True` - Case normalization
  - `strict_mode: bool = False` - Fail-fast validation

### Pipeline Classes

**PreprocessingPipeline:**
- Chains multiple preprocessors in sequence
- Aggregates issues and metrics from all stages
- Optional checkpoint support for debugging/resuming
- Graceful degradation on stage failure (logs but continues)

**PipelineBuilder:**
- Factory for creating preset and custom pipelines
- Supports both static factory and instance convenience methods
- Preset configurations: `listings`, `events`, `snapshots`

## Available Preprocessors

### Cleaners (Detection & Flagging)

**TextNormalizer** (`cleaners.py`)
- Normalizes text fields for consistent matching and analysis
- Adds `*_normalized` columns without modifying originals
- Includes city name mappings (NYC→New York, LA→Los Angeles, etc.)
- **Input columns:** `artist_or_team`, `venue_name`, `city`
- **Output columns:** `artist_normalized`, `venue_normalized`, `city_normalized`
- **Metrics:** `rows_processed`, `columns_added`

**PriceOutlierHandler** (`cleaners.py`)
- Detects outliers using IQR method and absolute bounds
- Flags (doesn't remove) outliers with reason codes
- Reasons: `below_min`, `above_max`, `below_iqr`, `above_iqr`
- **Input column:** `listing_price`
- **Output columns:** `is_price_outlier`, `outlier_reason`
- **Metrics:** `outlier_count`, `outlier_percentage`, `by_reason` (dict), IQR bounds, quantiles

**DuplicateHandler** (`cleaners.py`)
- Detects duplicate listings within configurable time window (default 6 hours)
- Groups by: `event_id`, `section`, `row`, `seat_from`, `seat_to`
- Keeps first occurrence, flags subsequent duplicates
- **Input columns:** `event_id`, `section`, `row`, `timestamp`, `seat_from` (optional), `seat_to` (optional)
- **Output column:** `is_duplicate`
- **Metrics:** `duplicate_count`, `duplicate_percentage`, `time_window_hours`
- **Constructor:** `DuplicateHandler(time_window_hours=6)`

### Validators (Integrity & Consistency)

**SchemaValidator** (`validators.py`)
- Validates column presence and data types
- Checks null rates in required columns (threshold: 50%)
- Data types: flexible datetime/numeric matching
- **Supports:** listings, events, snapshots (via `data_type` param)
- **Metrics:** `validation_issues`, `error_count`, `warning_count`

**ReferentialValidator** (`validators.py`)
- Validates seat zone mapping for listings
- Validates enum values for snapshots
- Optional event_id referential integrity check
- **Data types:** listings, snapshots
- **Optional param:** `events_df` for event_id validation
- **Metrics:** `validation_issues`, `error_count`, `warning_count`

**TemporalValidator** (`validators.py`)
- Checks timestamps not in future
- Checks events haven't already occurred (optional via `allow_past_events`)
- Validates days_to_event calculation (±2 day tolerance)
- **Metrics:** `validation_issues`, `error_count`, `warning_count`
- **Constructor:** `TemporalValidator(allow_past_events=False)`

### Transformers (Enrichment & Preparation)

**EventMetadataJoiner** (`transformers.py`)
- Left-joins venue_capacity from events DataFrame
- Tracks missing event_id references
- **Input param:** `events_df` with event_id and venue_capacity
- **Output column:** `venue_capacity`
- **Metrics:** `rows_joined`, `missing_event_ids_count`

**MissingValueImputer** (`transformers.py`)
- Imputes venue_capacity (city median, fallback to default)
- Imputes face_value as 50% of listing_price
- Adds `*_imputed` boolean tracking columns
- **Input columns:** `venue_capacity`, `face_value`, `listing_price`, `city` (optional)
- **Output columns:** `venue_capacity_imputed`, `face_value_imputed`
- **Metrics:** `imputation_counts` (dict per column)

**TypeConverter** (`transformers.py`)
- Converts timestamp/event_datetime to timezone-aware UTC datetime
- Converts price columns to float64
- Converts categorical columns (seat_zone, event_type) to pd.Category
- **Columns handled:**
  - Datetime: `timestamp`, `event_datetime`
  - Price: `listing_price`, `face_value`, `total_price`
  - Categorical: `seat_zone`, `normalized_seat_zone`, `event_type`
- **Metrics:** `conversions_applied` (dict of column→conversion type)

**SeatZoneEnricher** (`transformers.py`)
- Normalizes section names to SeatZone enum values
- Uses SeatZoneMapper for intelligent zone detection
- Defaults to UPPER_TIER for unmappable sections
- **Input column:** `section`
- **Output column:** `normalized_seat_zone` (SeatZone enum)
- **Metrics:** `zones_normalized_count`, `unmappable_count`

**TemporalFeatureEnricher** (`transformers.py`)
- Adds temporal features from timestamps
- **Output columns:**
  - `hour_of_day` - Hour extracted from timestamp
  - `days_to_event` - Fractional days between timestamp and event_datetime
  - `is_weekend` - Boolean for Saturday/Sunday
- **Metrics:** `features_added` (count)

## Pipeline Presets

### Listings Pipeline (`build_listings_pipeline`)
Comprehensive pipeline for ticket listing data (11 stages):

1. Schema validation (listings)
2. Text normalization
3. Event metadata join (optional)
4. Type conversion
5. Seat zone enrichment
6. Temporal features
7. Missing value imputation
8. Price outlier detection
9. Duplicate detection
10. Temporal validation
11. Referential validation

**Usage:**
```python
from ticket_price_predictor.preprocessing import PipelineBuilder

# Static usage
pipeline = PipelineBuilder.build_listings_pipeline(
    events_df=events_df,
    config=config,
    checkpoint_dir="./checkpoints"
)
result = pipeline.process(listings_df)

# Instance usage
builder = PipelineBuilder(config=config)
pipeline = builder.build_preset("listings", events_df=events_df)
```

### Events Pipeline (`build_events_pipeline`)
Lightweight pipeline for event metadata (5 stages):

1. Schema validation (events)
2. Text normalization
3. Type conversion
4. Missing value imputation
5. Temporal validation

**Usage:**
```python
pipeline = PipelineBuilder.build_events_pipeline(config=config)
result = pipeline.process(events_df)
```

### Snapshots Pipeline (`build_snapshots_pipeline`)
Minimal pipeline for price snapshots (5 stages):

1. Schema validation (snapshots)
2. Type conversion
3. Price outlier detection
4. Temporal validation
5. Referential validation

**Usage:**
```python
pipeline = PipelineBuilder.build_snapshots_pipeline(
    events_df=events_df,  # for referential validation
    config=config
)
result = pipeline.process(snapshots_df)
```

### Custom Pipelines (`build_custom_pipeline`)
Build pipelines with arbitrary stages:

```python
stages = [
    SchemaValidator("listings"),
    TextNormalizer(config),
    PriceOutlierHandler(config),
]
pipeline = PipelineBuilder.build_custom_pipeline(stages, name="my_pipeline")
```

## Configuration Reference

**PreprocessingConfig Fields:**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `iqr_multiplier` | float | 1.5 | IQR coefficient for outlier bounds |
| `price_min` | float | 1.0 | Minimum valid price absolute bound |
| `price_max` | float | 50000.0 | Maximum valid price absolute bound |
| `venue_capacity_default` | int | 15000 | Default capacity if missing |
| `imputation_strategy` | str | "median" | Method for imputation (currently median only) |
| `normalize_case` | bool | True | Normalize text to lowercase |
| `strict_mode` | bool | False | Fail-fast validation (abort on error) |

**Example:**
```python
config = PreprocessingConfig(
    iqr_multiplier=2.0,
    price_min=5.0,
    price_max=10000.0,
    venue_capacity_default=20000,
)
pipeline = PipelineBuilder.build_listings_pipeline(config=config)
```

## Usage Examples

### Basic Pipeline Execution

```python
import pandas as pd
from ticket_price_predictor.preprocessing import PipelineBuilder, PreprocessingConfig

# Load data
listings_df = pd.read_parquet("listings.parquet")
events_df = pd.read_parquet("events.parquet")

# Build pipeline
config = PreprocessingConfig()
pipeline = PipelineBuilder.build_listings_pipeline(
    events_df=events_df,
    config=config
)

# Process
result = pipeline.process(listings_df)

# Inspect results
print(f"Issues: {result.issues}")
print(f"Metrics: {result.metrics}")
print(f"Output shape: {result.data.shape}")
```

### Preset by Type

```python
from ticket_price_predictor.preprocessing import PipelineBuilder

builder = PipelineBuilder(config)

# Build by type string
listings_pipeline = builder.build_preset("listings", events_df=events_df)
events_pipeline = builder.build_preset("events")
snapshots_pipeline = builder.build_preset("snapshots", events_df=events_df)

# Process each
listings_result = listings_pipeline.process(listings_df)
events_result = events_pipeline.process(events_df)
snapshots_result = snapshots_pipeline.process(snapshots_df)
```

### Custom Pipeline Composition

```python
from ticket_price_predictor.preprocessing import (
    PipelineBuilder,
    TextNormalizer,
    PriceOutlierHandler,
    SchemaValidator,
)

config = PreprocessingConfig()
stages = [
    SchemaValidator("listings"),
    TextNormalizer(config),
    PriceOutlierHandler(config),
    # Add only what you need
]

pipeline = PipelineBuilder.build_custom_pipeline(stages, name="custom_listings")
result = pipeline.process(listings_df)
```

### With Checkpoints

```python
pipeline = PipelineBuilder.build_listings_pipeline(
    events_df=events_df,
    checkpoint_dir="./preprocessing_checkpoints"
)

result = pipeline.process(listings_df)

# Resume from checkpoint if pipeline failed
if "CRITICAL" in str(result.issues):
    df = pipeline.resume_from_checkpoint(stage_index=5)
    # Continue processing with checkpoint data
```

### Quality Reporting

```python
from ticket_price_predictor.preprocessing import QualityReporter, QualityThresholds

thresholds = QualityThresholds(
    drop_rate_warning=5.0,
    drop_rate_error=20.0,
    outlier_rate_warning=5.0,
)

reporter = QualityReporter(thresholds=thresholds)
metrics = reporter.extract_metrics(result)

# Text summary
print(reporter.generate_text_summary(metrics))

# JSON export (for monitoring)
json_export = reporter.generate_json_export(metrics)
```

## Integration Points

### ListingCollector Integration

When collecting listings, enable preprocessing automatically:

```python
from ticket_price_predictor.ingestion.collectors import ListingCollector

collector = ListingCollector(preprocess=True)  # Enables preprocessing
listings = collector.collect(artist="Bruno Mars", event_id="12345")
```

The collector applies the listings pipeline with default config before returning data.

### ModelTrainer Integration

When training models, preprocessing is applied leak-free:

```python
from ticket_price_predictor.ml.training.trainer import ModelTrainer

trainer = ModelTrainer(preprocess=True)  # Enables preprocessing
result = trainer.train(
    data_loader=loader,
    config=training_config,
    preprocess=True,
)
```

The trainer applies preprocessing AFTER split-first temporal division to prevent data leakage.

### CLI Script: preprocess_data.py

Standalone script for bulk preprocessing:

```bash
python scripts/preprocess_data.py \
    --input-dir data/raw \
    --output-dir data/preprocessed \
    --dataset-type listings \
    --checkpoint-dir data/checkpoints \
    --config preprocessing_config.json
```

**Script features:**
- Batch preprocessing of parquet files
- Configurable via JSON or CLI flags
- Checkpoint support for fault tolerance
- Quality report generation

## Quality Reporting

### QualityMetrics

Comprehensive metrics dataclass:
- `input_rows`, `output_rows`, `dropped_rows`, `flagged_rows`
- `column_completeness` - Dict[str, float] % non-null per column
- `outlier_counts` - Dict[str, int] by column
- `validation_errors` - Dict[str, int] by error type
- Properties: `drop_rate`, `retention_rate`

### QualityThresholds

Configurable alert thresholds:
- `drop_rate_warning/error` - Default 5.0% / 20.0%
- `null_rate_warning/error` - Default 10.0% / 50.0%
- `outlier_rate_warning/error` - Default 5.0% / 15.0%
- `duplicate_rate_warning/error` - Default 5.0% / 20.0%

### QualityReporter

Methods:
- `extract_metrics(result: ProcessingResult) -> QualityMetrics` - Parse pipeline output
- `generate_text_summary(metrics: QualityMetrics) -> str` - Human-readable report
- `generate_json_export(metrics: QualityMetrics) -> str` - JSON for monitoring
- `check_thresholds(metrics: QualityMetrics) -> AlertLevel` - OK/WARNING/ERROR
- `compare_against_baseline(current, baseline) -> dict` - Track changes

## Common Patterns

### Pattern: Filter Flagged Rows After Processing

```python
result = pipeline.process(listings_df)

# Keep only non-outlier, non-duplicate rows
clean_df = result.data[
    ~result.data["is_price_outlier"] & ~result.data["is_duplicate"]
]

print(f"Dropped {len(result.data) - len(clean_df)} rows")
```

### Pattern: Iterate Pipeline Development

```python
from ticket_price_predictor.preprocessing import PipelineBuilder

# Start minimal
stages = [SchemaValidator("listings"), TextNormalizer(config)]
pipeline = PipelineBuilder.build_custom_pipeline(stages)
result = pipeline.process(df)
print(f"Issues: {len(result.issues)}")

# Add stages incrementally
stages.append(PriceOutlierHandler(config))
pipeline = PipelineBuilder.build_custom_pipeline(stages)
result = pipeline.process(df)
# Evaluate impact on issues/metrics
```

### Pattern: Reuse Preprocessor Chains

```python
# Save pipeline configuration
config = PreprocessingConfig(iqr_multiplier=2.0, price_min=5.0)
pipeline = PipelineBuilder.build_listings_pipeline(config=config)

# Apply consistently across multiple datasets
for week_df in weekly_datasets:
    result = pipeline.process(week_df)
    reporter.extract_metrics(result)
```

## Testing Preprocessors

Key test patterns:
- Empty DataFrames - All preprocessors handle gracefully
- Missing columns - Cleaners issue warnings, validators error
- Type mismatches - TypeConverter auto-fixes, SchemaValidator warns
- Null patterns - Validators flag >50% nulls as ERROR
- Temporal inconsistencies - TemporalValidator logs with 2-day tolerance

See `tests/test_preprocessing/` for comprehensive test suite.
