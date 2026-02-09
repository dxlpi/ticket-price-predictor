# Ticket Price Predictor

ML system for predicting ticket prices at the seat-zone level using historical time-series data.

## Project Structure

```
src/ticket_price_predictor/
├── api/                    # External API clients (Ticketmaster, Setlist.fm)
├── schemas/                # Pydantic data models
├── storage/                # Parquet storage layer (repositories)
├── ingestion/              # Data collection services
├── scrapers/               # VividSeats/StubHub web scrapers
├── normalization/          # Seat zone normalization
├── validation/             # Data quality checks
├── synthetic/              # Synthetic data generation
└── ml/                     # Machine learning pipeline
    ├── features/           # Feature extractors
    ├── models/             # Baseline + LightGBM models
    ├── training/           # Training pipeline
    └── inference/          # Prediction service
```

## Quick Commands

```bash
# Data collection
python scripts/collect_listings.py --artist "Bruno Mars" --max-events 3
python scripts/ingest_events.py --event-types concert --cities "Las Vegas"

# Model training
python scripts/train_model.py --model lightgbm --version v2

# Predictions
python scripts/predict.py --artist "BTS" --city "Tampa" --all-zones

# Quality checks
make check   # lint + typecheck + test
make test    # pytest only
```

## Key Patterns

- **Pydantic models** for all data schemas with Parquet serialization
- **Repository pattern** for data access (EventRepository, ListingRepository, SnapshotRepository)
- **Feature extractors** implement `FeatureExtractor` base class with `fit()` and `extract()`
- **Time-based splits** to avoid data leakage in ML training
- **ArtistStatsCache** computes popularity from historical data (no hardcoded lists)

## Data Flow

```
Ticketmaster API → EventMetadata → events.parquet
VividSeats scraper → TicketListing → listings.parquet (partitioned by date)
                          ↓
                  Feature Pipeline
                          ↓
                  LightGBM Model → PricePrediction
```

## Environment

- Python 3.11+
- Key deps: pydantic, pyarrow, lightgbm, scikit-learn, playwright
- API key: `TICKETMASTER_API_KEY` in `.env`

## Current Model Performance

- **MAE**: $94.60
- **MAPE**: 22.4%
- **Top feature**: `artist_avg_price` (61.7% importance)
