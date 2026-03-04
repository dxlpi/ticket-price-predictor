# Ticket Price Predictor

ML system for predicting secondary-market ticket prices at the seat-zone level using historical time-series data, demand signals, and event context.

## Project Structure

```
src/ticket_price_predictor/
├── api/                    # External API clients (Ticketmaster, Setlist.fm)
├── schemas/                # Pydantic data models
├── storage/                # Parquet storage layer (repositories)
├── ingestion/              # Data collection services
├── scrapers/               # VividSeats/StubHub web scrapers (Playwright)
├── normalization/          # Seat zone normalization
├── validation/             # Data quality checks
├── preprocessing/          # Data cleaning, validation, transformation pipeline
├── popularity/             # External popularity aggregation (YouTube Music, Last.fm)
├── synthetic/              # Synthetic data generation
└── ml/                     # Machine learning pipeline
    ├── features/           # Feature extractors (10 domains, 67 features with snapshot / 63 without)
    │   ├── performer.py    # Artist stats + artist×zone encoding
    │   ├── event.py        # Event/location features
    │   ├── seating.py      # Seat zone features
    │   ├── timeseries.py   # Time-to-event & momentum features
    │   ├── popularity.py   # External API popularity features
    │   ├── regional.py     # Regional price variation features
    │   ├── event_pricing.py # Event-level target encoding (strongest features)
    │   ├── geo_mapping.py  # City→country/region lookup helpers
    │   └── pipeline.py     # Feature pipeline orchestration
    ├── models/             # Baseline (Ridge) + LightGBM + Quantile LightGBM
    ├── training/           # Training pipeline (split-first, no leakage)
    │   ├── splitter.py     # TimeBasedSplitter with artist stratification
    │   ├── trainer.py      # ModelTrainer with leak-free flow
    │   └── evaluator.py    # Model evaluation metrics
    ├── tuning/             # Optuna hyperparameter tuning
    └── inference/          # Prediction service with cold-start handling
```

## Quick Commands

```bash
# Data collection
python scripts/collect_listings.py --artist "Bruno Mars" --max-events 3
python scripts/ingest_events.py --event-types concert --cities "Las Vegas"

# Model training
python scripts/train_model.py --model lightgbm --version v13
python scripts/train_model.py --from-study lightgbm_aggressive --version v14

# Hyperparameter tuning
python scripts/tune_model.py --n-trials 50

# Predictions
python scripts/predict.py --artist "BTS" --city "Tampa" --all-zones

# Preprocessing
python scripts/preprocess_data.py

# Quality checks
make check   # lint + typecheck + test
make test    # pytest only
```

## Key Patterns

- **Pydantic models** for all data schemas with Parquet serialization
- **Repository pattern** for data access (EventRepository, ListingRepository, SnapshotRepository)
- **Feature extractors** implement `FeatureExtractor` base class with `fit()` and `extract()`
- **Split-first training** — raw data is split temporally before feature extraction to prevent data leakage
- **Artist stratification** — each artist is split independently by time for balanced representation
- **ArtistStatsCache** computes popularity from historical data (no hardcoded lists)
- **RegionalStatsCache** computes city/country/global price stats with fallback chain
- **Preprocessing pipeline** with cleaners, validators, and transformers (PipelineBuilder)

## Data Flow

```
Ticketmaster API → EventMetadata → events.parquet (event discovery)
VividSeats/StubHub scrapers → TicketListing → listings.parquet (actual prices)
YouTube Music/Last.fm → PopularityService → popularity features
                          ↓
              Preprocessing Pipeline
                          ↓
              Split raw data (artist-stratified, temporal)
                          ↓
              Feature Pipeline (fit on train only)
                          ↓
              LightGBM Model → PricePrediction (with 95% CI)
```

## Training Pipeline

The training pipeline prevents data leakage by:
1. Filtering invalid prices (<$10) and capping outliers at 95th percentile
2. Normalizing city names for consistent regional grouping
3. Normalizing artist name aliases (e.g. "BTS - Bangtan Boys" → "BTS")
4. Splitting raw data temporally with artist stratification (`split_raw()`)
4. Fitting the feature pipeline on training data only (with Bayesian-smoothed regional stats)
5. Transforming train/val/test independently
6. Removing zero-variance features and log-transforming price-based features
7. Log-transforming target (`np.log1p`) for better skewed-price handling
8. Training LightGBM with DART boosting and early stopping (patience=200)

## Issue Tracking

When you encounter a major issue during any task (bugs with non-obvious root causes, feature engineering experiments that failed or succeeded unexpectedly, data quality problems, performance regressions, or architectural decisions with significant trade-offs), document it in `docs/issues/`. Use the next available number following the format `NNN-short-description.md` with sections: Problem, Impact, Root Cause, Solution, Outcome. Mark status as Open or Resolved and severity as Critical/High/Medium.

## Environment

- Python >=3.11, uv for package management
- Key deps: pydantic, pyarrow, lightgbm, scikit-learn, playwright
- API keys in `.env`: `TICKETMASTER_API_KEY`, `LASTFM_API_KEY`

## Current Model Performance (v30)

- **MAE**: $133.86
- **MAPE**: 40.0%
- **R²**: 0.56
- **RMSE**: $221.22
- **Dataset**: 136 events, 42 artists (35,476 listings)
- **Top features**: `event_section_median_price` (57.3%), `event_zone_median_price` (21.1%), `artist_regional_avg_price` (7.1%), `event_median_price` (6.8%), `artist_venue_price` (1.2%)
- **Key improvements over v29-section**: artist_venue_price interaction feature (+1.2% importance), momentum disabled by default, snapshot infrastructure (pending data), 58 features after zero-variance removal. MAE improved $14.41 (9.7%), R² improved 0.46→0.56
- **Feature pipeline**: 10 domains, 67 raw features (with snapshot) / 63 without snapshot; 58 active after zero-variance removal
