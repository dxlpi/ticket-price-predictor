# Data Preprocessing Pipeline - Work Plan

## Context

### Original Request
Build a data preprocessing pipeline for the ticket price prediction system that handles the variety of data sources and formats being collected.

### Current Architecture Analysis

**Data Sources Identified:**
1. **Ticketmaster API** (`api/` + `ingestion/events.py`) - Structured EventMetadata
2. **StubHub Scraper** (`scrapers/stubhub.py`) - Semi-structured HTML/JSON parsing
3. **VividSeats Scraper** (`scrapers/vividseats.py`) - Semi-structured HTML/JSON parsing
4. **Price Snapshots** (`ingestion/snapshots.py`) - Aggregated time-series data

**Current Data Schemas:**
- `EventMetadata`: event_id, event_type, event_datetime, artist_or_team, venue_id, venue_name, city, country, venue_capacity
- `TicketListing`: listing_id, event_id, source, timestamp, section, row, seat_from/to, quantity, face_value, listing_price, total_price, days_to_event
- `PriceSnapshot`: event_id, seat_zone, timestamp, price_min/avg/max, inventory_remaining, days_to_event
- `ScrapedEvent` / `ScrapedListing`: Raw scraper output schemas

**Current Validation:**
- `validation/quality.py`: Basic DataValidator for events and snapshots
- Pydantic model validators (e.g., price ordering in PriceSnapshot)
- Repository-level deduplication (ListingRepository hash-based)

**Gaps Identified:**
1. No centralized preprocessing pipeline - validation scattered across modules
2. No outlier detection for prices (extreme values pass through)
3. No standardization of text fields (artist names, venue names vary)
4. No handling of missing venue_capacity (common nullable field)
5. No price normalization across currencies (assumes USD)
6. No temporal data alignment (snapshots vs listings timestamps)
7. No data quality metrics/reporting
8. Feature extractors assume clean data (fill with defaults silently)

### Research Findings
- Data partitioned by date (listings: year/month/day, snapshots: year/month)
- Seat zone normalization exists (`normalization/seat_zones.py`) but is post-hoc
- Feature pipeline expects specific columns: `artist_or_team`, `event_datetime`, `city`, `venue_capacity`, `listing_price`
- ArtistStatsCache computes statistics from historical data at training time

---

## Work Objectives

### Core Objective
Create a modular, configurable data preprocessing pipeline that transforms raw collected data into clean, validated, ML-ready datasets.

### Deliverables
1. **Preprocessing Module** (`src/ticket_price_predictor/preprocessing/`)
   - Base preprocessor interface
   - Data cleaners (text normalization, outlier handling)
   - Data validators (extended from current validation)
   - Data transformers (type conversion, imputation)
   - Pipeline orchestrator

2. **Quality Report System**
   - Data quality metrics collection
   - Quality report generation
   - Threshold-based alerts

3. **Integration Points**
   - Hook into ingestion services (pre-storage)
   - Hook into training pipeline (post-load)
   - CLI commands for batch preprocessing

4. **Tests and Documentation**
   - Unit tests for each preprocessor
   - Integration tests for full pipeline
   - AGENTS.md for the preprocessing module

### Definition of Done
- [ ] All existing data passes through pipeline without data loss
- [ ] Outliers are detected and flagged (not silently dropped)
- [ ] Text fields normalized consistently
- [ ] Missing values handled with configurable strategies
- [ ] Quality metrics generated for each preprocessing run
- [ ] Tests achieve 90%+ coverage on preprocessing module
- [ ] Pipeline integrates with both ingestion and training flows

---

## Guardrails

### Must Have
- Configurable thresholds (not hardcoded magic numbers)
- Non-destructive by default (flag issues, don't drop data)
- Audit trail (log what was transformed and why)
- Backward compatible with existing Parquet schemas
- Idempotent operations (can re-run safely)

### Must NOT Have
- Silent data dropping without logging
- Hardcoded artist/venue whitelists
- Breaking changes to existing schemas
- External API calls during preprocessing
- Heavy dependencies (keep it lightweight)

---

## Task Flow and Dependencies

```
[1] Create preprocessing module structure
         |
    +----+----+
    |         |
   [2]       [3]
 Cleaners  Validators
    |         |
    +----+----+
         |
        [4]
    Transformers
         |
        [5]
  Pipeline Orchestrator
         |
    +----+----+
    |         |
   [6]       [7]
 Quality   Integration
 Reports    Points
    |         |
    +----+----+
         |
        [8]
    Tests & Docs
```

---

## Detailed TODOs

### Task 1: Create Preprocessing Module Structure
**File:** `src/ticket_price_predictor/preprocessing/__init__.py`
**Acceptance Criteria:**
- Create module directory with proper structure
- Define base `Preprocessor` abstract class with `process(df) -> ProcessingResult` interface
- Define `ProcessingResult` dataclass with: `data`, `issues`, `metrics`
- Export public API in `__init__.py`

**Subtasks:**
- 1.1: Create directory `src/ticket_price_predictor/preprocessing/`
- 1.2: Create `base.py` with `Preprocessor` ABC and `ProcessingResult`
- 1.3: Create `__init__.py` with exports
- 1.4: Create `config.py` for preprocessing configuration (thresholds, strategies)

---

### Task 2: Implement Data Cleaners
**Files:** `src/ticket_price_predictor/preprocessing/cleaners.py`
**Acceptance Criteria:**
- `TextNormalizer`: Standardize artist names, venue names (lowercase, strip whitespace, handle unicode)
- `PriceOutlierHandler`: Detect extreme prices using IQR or configurable bounds
- `DuplicateHandler`: Identify and flag duplicates beyond current hash-based dedup
- Each cleaner implements `Preprocessor` interface

**Subtasks:**
- 2.1: Implement `TextNormalizer` class
  - Normalize artist_or_team: strip, lowercase for matching, preserve original
  - Normalize venue_name: similar treatment
  - Normalize city: map common variations ("NYC" -> "New York")
- 2.2: Implement `PriceOutlierHandler` class
  - Configurable absolute bounds (default: $1 min, $50,000 max)
  - IQR-based detection with configurable multiplier (default: 1.5x IQR)
    - Lower bound: Q1 - (1.5 * IQR)
    - Upper bound: Q3 + (1.5 * IQR)
  - Flag but don't remove (add `is_price_outlier` column)
  - Store outlier reason in `outlier_reason` column (e.g., "below_iqr", "above_max")
- 2.3: Implement `DuplicateHandler` class
  - Time-window deduplication (same listing within N hours)
  - Cross-source deduplication (same seat, different source)

---

### Task 3: Implement Extended Validators
**Files:** `src/ticket_price_predictor/preprocessing/validators.py`
**Acceptance Criteria:**
- Extend existing `DataValidator` with additional checks
- `SchemaValidator`: Verify required columns and types
- `ReferentialValidator`: Check event_id exists in events table
- `TemporalValidator`: Ensure timestamp ordering, detect future dates
- Return structured validation issues (not just bool)

**Subtasks:**
- 3.1: Implement `SchemaValidator` class
  - Check required columns present
  - Verify column dtypes match expected
  - Detect unexpected null patterns
- 3.2: Implement `ReferentialValidator` class
  - Validate event_id references exist in events table
  - For **listings**: Validate `section` (raw string) can be mapped to valid `SeatZone` enum via `SeatZoneMapper`
  - For **snapshots**: Validate `seat_zone` contains valid `SeatZone` enum values
  - Cross-reference listing events with event metadata
  - **Note:** `TicketListing` has `section` (string), `PriceSnapshot` has `seat_zone` (enum) - different validation logic required
- 3.3: Implement `TemporalValidator` class
  - Detect timestamps in the future
  - Detect events that already passed (for training exclusion)
  - Validate days_to_event calculation matches timestamps
- 3.4: Create `ValidationIssue` dataclass for structured error reporting

---

### Task 4: Implement Data Transformers
**Files:** `src/ticket_price_predictor/preprocessing/transformers.py`
**Acceptance Criteria:**
- `MissingValueImputer`: Handle nulls with configurable strategies
- `TypeConverter`: Ensure consistent dtypes for ML
- `SeatZoneEnricher`: Pre-compute normalized seat zones
- `TemporalFeatureEnricher`: Add derived time features

**Subtasks:**
- 4.1: Implement `MissingValueImputer` class
  - **For EventMetadata preprocessing:**
    - `venue_capacity`: impute with median by city, fallback to global median (15,000)
  - **For TicketListing preprocessing:**
    - `face_value`: impute with listing_price * 0.5 (configurable ratio)
    - Note: If listings need `venue_capacity` for ML features, join from EventMetadata via `event_id` BEFORE imputation (see Task 4.5)
  - Configurable strategies: median, mean, mode, constant, drop
- 4.5: Implement `EventMetadataJoiner` class (NEW)
  - Join `venue_capacity` from EventMetadata to TicketListing via `event_id`
  - This transformer should run BEFORE `MissingValueImputer` when processing listings that need event-level fields
  - Fields to join: `venue_capacity`, optionally `event_type`
- 4.2: Implement `TypeConverter` class
  - Ensure datetime columns are timezone-aware UTC
  - Ensure price columns are float64
  - Ensure categorical columns are properly typed
- 4.3: Implement `SeatZoneEnricher` class
  - Use existing `SeatZoneMapper` from normalization module
  - Add `normalized_seat_zone` column to listings
- 4.4: Implement `TemporalFeatureEnricher` class
  - Add `hour_of_day` from timestamp
  - Add `days_since_onsale` if onsale_date available
  - Validate and recalculate `days_to_event`

---

### Task 5: Build Pipeline Orchestrator
**Files:** `src/ticket_price_predictor/preprocessing/pipeline.py`
**Acceptance Criteria:**
- `PreprocessingPipeline` class that chains preprocessors
- Configurable pipeline stages (skip/include)
- Aggregate results from all stages
- Support for both DataFrame and file-based processing

**Subtasks:**
- 5.1: Implement `PreprocessingPipeline` class
  - Constructor takes list of preprocessors
  - `process(df)` runs all preprocessors in order
  - Aggregate all issues and metrics
- 5.2: Implement `PipelineBuilder` for easy configuration
  - Preset pipelines: "listings", "events", "snapshots"
  - Custom pipeline construction
- 5.3: Add checkpoint support
  - Save intermediate results for debugging
  - Resume from checkpoint on failure
- 5.4: Add parallel processing option for large datasets

---

### Task 6: Build Quality Report System
**Files:** `src/ticket_price_predictor/preprocessing/quality.py`
**Acceptance Criteria:**
- `QualityMetrics` dataclass with comprehensive stats
- `QualityReporter` generates human-readable reports
- Configurable thresholds for warnings/errors
- Export to JSON for monitoring integration

**Subtasks:**
- 6.1: Define `QualityMetrics` dataclass
  - Row counts (input, output, dropped, flagged)
  - Column completeness percentages
  - Outlier counts by column
  - Validation error counts by type
- 6.2: Implement `QualityReporter` class
  - Generate text summary
  - Generate JSON export
  - Compare against baseline metrics
- 6.3: Implement threshold-based alerting
  - Configurable warning/error thresholds
  - Return alert level (OK, WARNING, ERROR)

---

### Task 7: Create Integration Points
**Files:** Multiple integration touchpoints
**Acceptance Criteria:**
- Hook into `ListingCollector` post-collection
- Hook into `ModelTrainer` pre-training
- CLI command for batch preprocessing
- Backward compatible (opt-in preprocessing)

**Subtasks:**
- 7.1: Add preprocessing hook to `ListingCollector`
  - Optional `preprocess=True` flag
  - Use listings pipeline preset
- 7.2: Add preprocessing hook to `ModelTrainer.train()`
  - Optional `preprocess=True` flag
  - Default to True for new training runs
- 7.3: Create CLI script `scripts/preprocess_data.py`
  - Accept input path, output path, config
  - Support dry-run mode (report only)
  - Support incremental processing (new data only):
    - Track processed partitions via watermark file (`.omc/state/preprocessing_watermark.json`)
    - Watermark stores: `{"listings": {"last_partition": "2025/01/15"}, "events": {"last_processed_ts": "2025-01-15T12:00:00Z"}}`
    - On incremental run, only process partitions newer than watermark
    - Update watermark after successful processing
- 7.4: Add preprocessing to existing `scripts/train_model.py`
  - Add `--preprocess` flag
  - Report quality metrics before training

---

### Task 8: Tests and Documentation
**Files:** `tests/test_preprocessing.py`, `src/ticket_price_predictor/preprocessing/AGENTS.md`
**Acceptance Criteria:**
- Unit tests for each preprocessor class
- Integration tests for full pipeline
- Edge case tests (empty data, all nulls, extreme values)
- AGENTS.md documenting module purpose and usage

**Subtasks:**
- 8.1: Write unit tests for cleaners
  - Test TextNormalizer with various inputs
  - Test PriceOutlierHandler with edge cases
  - Test DuplicateHandler accuracy
- 8.2: Write unit tests for validators
  - Test schema validation
  - Test referential integrity checks
  - Test temporal validation
- 8.3: Write unit tests for transformers
  - Test imputation strategies
  - Test type conversion edge cases
  - Test enrichment accuracy
- 8.4: Write integration tests
  - Full pipeline with real sample data
  - Pipeline with intentionally dirty data
  - Verify no data loss for valid records
- 8.5: Write AGENTS.md documentation
  - Module overview
  - Usage examples
  - Configuration reference

---

## Commit Strategy

| Commit | Tasks | Message |
|--------|-------|---------|
| 1 | 1.1-1.4 | `feat(preprocessing): add module structure and base classes` |
| 2 | 2.1-2.3 | `feat(preprocessing): implement data cleaners` |
| 3 | 3.1-3.4 | `feat(preprocessing): implement extended validators` |
| 4 | 4.1-4.5 | `feat(preprocessing): implement data transformers and joiner` |
| 5 | 5.1-5.4 | `feat(preprocessing): build pipeline orchestrator` |
| 6 | 6.1-6.3 | `feat(preprocessing): add quality reporting system` |
| 7 | 7.1-7.4 | `feat(preprocessing): integrate with ingestion and training` |
| 8 | 8.1-8.5 | `test(preprocessing): add comprehensive tests and docs` |

---

## Success Criteria

### Functional
- [ ] Pipeline processes all existing parquet files without error
- [ ] Outliers detected and flagged (verify outlier detection runs; actual % depends on data distribution)
- [ ] Text normalization produces consistent artist/venue keys
- [ ] Missing venue_capacity imputed for 100% of EventMetadata records
- [ ] Quality report generated with all metrics populated
- [ ] Quality report includes outlier statistics (count, percentage, by reason)

### Performance
- [ ] Pipeline processes 100K listings in <30 seconds
- [ ] Memory usage stays under 2GB for 1M records
- [ ] Incremental processing only touches new partitions

### Quality
- [ ] Test coverage >90% for preprocessing module
- [ ] No regressions in existing model performance (MAE within 5%)
- [ ] Zero silent data drops (all removals logged)

---

## File References

**Existing files to integrate with:**
- `/Users/heather/ticket-price-predictor/src/ticket_price_predictor/validation/quality.py` - Extend patterns
- `/Users/heather/ticket-price-predictor/src/ticket_price_predictor/normalization/seat_zones.py` - Reuse mapper
- `/Users/heather/ticket-price-predictor/src/ticket_price_predictor/ingestion/listings.py` - Integration point
- `/Users/heather/ticket-price-predictor/src/ticket_price_predictor/ml/training/trainer.py` - Integration point
- `/Users/heather/ticket-price-predictor/src/ticket_price_predictor/ml/features/pipeline.py` - Similar pattern

**New files to create:**
- `/Users/heather/ticket-price-predictor/src/ticket_price_predictor/preprocessing/__init__.py`
- `/Users/heather/ticket-price-predictor/src/ticket_price_predictor/preprocessing/base.py`
- `/Users/heather/ticket-price-predictor/src/ticket_price_predictor/preprocessing/config.py`
- `/Users/heather/ticket-price-predictor/src/ticket_price_predictor/preprocessing/cleaners.py`
- `/Users/heather/ticket-price-predictor/src/ticket_price_predictor/preprocessing/validators.py`
- `/Users/heather/ticket-price-predictor/src/ticket_price_predictor/preprocessing/transformers.py`
- `/Users/heather/ticket-price-predictor/src/ticket_price_predictor/preprocessing/pipeline.py`
- `/Users/heather/ticket-price-predictor/src/ticket_price_predictor/preprocessing/quality.py`
- `/Users/heather/ticket-price-predictor/src/ticket_price_predictor/preprocessing/AGENTS.md`
- `/Users/heather/ticket-price-predictor/scripts/preprocess_data.py`
- `/Users/heather/ticket-price-predictor/tests/test_preprocessing.py`

---

## Estimated Effort

| Task | Complexity | Estimate |
|------|------------|----------|
| Task 1: Module Structure | Low | 30 min |
| Task 2: Cleaners | Medium | 2 hours |
| Task 3: Validators | Medium | 2 hours |
| Task 4: Transformers (incl. 4.5 Joiner) | Medium | 2.5 hours |
| Task 5: Pipeline | Medium | 1.5 hours |
| Task 6: Quality Reports | Low | 1 hour |
| Task 7: Integration | Medium | 1.5 hours |
| Task 8: Tests & Docs | Medium | 2.5 hours |
| **Total** | | **~13.5 hours** |

---

*Plan generated by Prometheus (Planner Agent)*
*Ready for Critic review*
