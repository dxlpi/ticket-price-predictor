# Architecture

## System Overview

ML system for predicting secondary-market (resale) ticket prices at the seat-zone level. Ingests event metadata from Ticketmaster, scrapes actual resale prices from VividSeats/StubHub, enriches with popularity data from YouTube Music and Last.fm, and trains a LightGBM model with 67 raw features across 10 extractor domains (listing domain disabled; snapshot domain enabled by default). Default boosting: GBDT+Huber loss.

## Layer Diagram

                     EXTERNAL SERVICES
   +------------------+  +---------------+  +-------------+
   | Ticketmaster API  |  | YouTube Music |  | Last.fm API |
   | (event discovery) |  | (ytmusicapi)  |  | (httpx)     |
   +--------+---------+  +------+--------+  +------+------+
            |                    |                  |
            v                    v                  v
   +--------+---------+  +------+------------------+------+
   | api/              |  | popularity/                     |
   |  ticketmaster.py  |  |  youtube.py, lastfm.py          |
   |  setlistfm.py     |  |  aggregator.py, service.py      |
   +--------+---------+  |  cache.py                        |
            |             +------+---------------------------+
            v                    |
   +--------+---------+         |
   | schemas/          |         |
   |  listings.py      |<--------+
   |  snapshots.py     |
   +--------+---------+
            |
            v
   +--------+---------+
   | scrapers/         |         (Playwright / httpx)
   |  vividseats.py    |
   |  stubhub.py       |
   |  stubhub_playwright.py |
   +--------+---------+
            |
            v
   +--------+---------+
   | ingestion/        |         (orchestration)
   |  listings.py      |
   |  events.py        |
   |  snapshots.py     |
   +--------+---------+
            |
            v
   +--------+---------+
   | storage/          |         (Parquet I/O)
   |  parquet.py       |
   |  repository.py    |
   +--------+---------+
            |
            v
   +-----------+--------+  +--------------------+
   | normalization/      |  | preprocessing/     |
   |  seat_zones.py      |  |  cleaners.py       |
   +--------+------------+  |  validators.py     |
            |                |  transformers.py   |
            v                |  pipeline.py       |
   +--------+------------+  +--------+-----------+
   | validation/          |           |
   |  quality.py          |           |
   +--------+-------------+           |
            +-------+--------+--------+
                    |
                    v
   +----------------+--------------------------------------+
   | ml/                                                    |
   |  features/    (10 domain extractors, 63+ features)      |
   |  models/      (Ridge baseline, LightGBM, Quantile)     |
   |  training/    (split-first, leak-free pipeline)         |
   |  tuning/      (Optuna hyperparameter search)            |
   |  inference/   (PricePredictor + cold-start fallback)    |
   +--------------------------------------------------------+

## Dependency Rules

Code flows top-to-bottom. Each layer may only depend on layers above it:

| Layer | May Depend On | Must NOT Depend On |
|-------|---------------|-------------------|
| `schemas/` | pydantic, pyarrow (external only) | Any other project module |
| `api/` | `schemas/`, httpx | `scrapers/`, `storage/`, `ml/` |
| `popularity/` | `schemas/`, httpx, ytmusicapi | `ml/`, `storage/` |
| `scrapers/` | `schemas/`, playwright | `ml/`, `storage/` |
| `ingestion/` | `schemas/`, `scrapers/`, `storage/`, `normalization/` | `ml/` |
| `storage/` | `schemas/`, pyarrow | `ml/`, `scrapers/` |
| `normalization/` | `schemas/` | `ml/`, `storage/` |
| `preprocessing/` | `schemas/`, `normalization/` | `ml/` |
| `validation/` | `schemas/` | `ml/` |
| `ml/` | All layers above | — |
| `config.py` | pydantic-settings | Any module (imported by others) |

## Modification Policy

| Directory | Policy |
|-----------|--------|
| `schemas/` | **Careful**: Changes cascade to storage, scrapers, ingestion, ML. Add fields — don't remove or rename without checking all consumers. |
| `ml/features/` | **Moderate**: New extractors must implement `FeatureExtractor` ABC. Register in `pipeline.py`. Test with `tests/test_ml_features.py`. |
| `ml/training/` | **Careful**: Leak prevention logic lives here. Any change to `trainer.py` or `splitter.py` must preserve split-before-fit invariant. |
| `scrapers/` | **Moderate**: Anti-detection patterns are fragile. Test changes against live sites carefully. |
| `storage/` | **Careful**: Parquet schema changes affect all stored data. New fields need migration consideration. |
| `config.py` | **Moderate**: `MLConfig` is frozen dataclass — add fields with defaults to avoid breaking existing code. |

## Training Pipeline

1. Load raw listings from Parquet via `DataLoader`
2. Filter invalid prices (<$10), cap outliers at 95th percentile
3. Normalize city names (`geo_mapping._normalize_city`)
4. Normalize artist name aliases (e.g., "BTS - Bangtan Boys" → "BTS")
5. Split raw data temporally with artist stratification (`TimeBasedSplitter.split_raw()`)
6. Fit `FeaturePipeline` on training split only (two-stage: base extractors → post-extractors)
7. Transform train/val/test independently
8. Remove zero-variance features, log-transform price-based features
9. Log-transform target (`np.log1p`), train LightGBM with GBDT+Huber loss and early stopping
10. Evaluate on raw scale (`np.expm1` inverse transform)

## Model Versioning

Models stored as `data/models/lightgbm_v{N}.joblib` + `lightgbm_v{N}_metrics.json` pairs. No formal registry — version tracked by filename convention. Current production: v32.

## Inference Architecture

`PricePredictor` service with multi-tier fallback:
1. Primary: LightGBM prediction with fitted feature pipeline
2. Cold-start: `ColdStartHandler` → `PopularityService` tier → keyword matching → global defaults
3. Uncertainty: `QuantileLightGBMModel` provides 95% confidence intervals (2.5th/97.5th percentile)

## Data Pipeline

### Collection DAG
```
Ticketmaster API → EventIngestionService → events.parquet
                          ↓
VividSeats Scraper → ListingCollector → listings.parquet
                          ↓
SnapshotCollector → snapshots.parquet
```

Scheduling: EC2 t3.micro systemd timer runs hourly via `scripts/monitor_popular.py`.

### Data Lineage

| Source | Raw Storage | Processing | Training Input |
|--------|-------------|------------|----------------|
| Ticketmaster API | `data/events/events.parquet` | Event metadata enrichment | Event features (city, venue, date) |
| VividSeats scraper | `data/listings/year=*/month=*/day=*/listings.parquet` | Price filtering, city/artist normalization | All price-related features |
| YouTube Music | In-memory (cached to JSON) | `PopularityAggregator` weighted scoring | Popularity features |
| Last.fm | In-memory (cached to JSON) | `PopularityAggregator` weighted scoring | Popularity features |

### Quality Validation

- `validation/quality.py`: Data quality checks on raw scraped data
- `preprocessing/quality.py`: Pipeline quality metrics (completeness, consistency)
- Deduplication: MD5 hash of composite key in `ListingRepository`
- Hive partitioning: `year=YYYY/month=MM/day=DD/` for efficient time-range queries

## Key Files

| File | Purpose | Why It Matters |
|------|---------|----------------|
| `ml/training/trainer.py` | `ModelTrainer.train()` — orchestrates leak-free training | Central pipeline; split-before-fit invariant lives here |
| `ml/features/pipeline.py` | `FeaturePipeline` — two-stage feature extraction | All 10 extractors registered and orchestrated here |
| `ml/features/event_pricing.py` | `EventPricingFeatureExtractor` — strongest features (60%+ importance) | Bayesian-smoothed target encoding with fallback chains |
| `ml/training/splitter.py` | `TimeBasedSplitter` — temporal split with artist stratification | Leak prevention depends on correct splitting |
| `schemas/listings.py` | `TicketListing`, `ScrapedEvent` — core data models | Schema changes cascade everywhere |
| `storage/repository.py` | `EventRepository`, `ListingRepository` — Parquet CRUD | Deduplication and partitioning logic |
| `scrapers/vividseats.py` | Playwright VividSeats scraper | Primary data source; anti-detection patterns |
| `config.py` | `MLConfig` — all ML magic numbers (~40 constants) | Single source of truth for thresholds and defaults |
| `ml/inference/predictor.py` | `PricePredictor` — prediction service | Cold-start fallback chain |
| `normalization/seat_zones.py` | `SeatZoneMapper` — section→zone mapping | Business logic for 4 standard seat zones |
| `popularity/service.py` | `PopularityService` — YouTube + Last.fm facade | External API aggregation with caching |
