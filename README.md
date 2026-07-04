# Ticket Price Predictor

ML system for predicting secondary-market ticket prices at the seat-zone level using historical time-series data, demand signals, and event context.

## Overview

The system targets the U.S. market (concerts, sports, theater) and predicts resale ticket prices by combining data from multiple sources with a stacking ensemble of gradient-boosted models.

**Current Model Performance (v38, production)**: primary MAE **$28.88** (seen events) | overall MAE $80.92 | MAPE 41.6% | R² 0.63
See [`docs/model-card.md`](docs/model-card.md) for the full benchmark table and scope definition.

## Features

- Event discovery via Ticketmaster Discovery API
- Real-time price scraping from VividSeats & StubHub (Playwright-based, anti-detection)
- Artist popularity aggregation from YouTube Music and Last.fm
- 80+ engineered features across 11 domains (artist, event, seating, time-series, popularity, regional, event/zone/section target encoding, venue, interactions, listing structural, snapshot)
- StackingEnsembleV2 (LightGBM Huber + deeper LightGBM + residual model → Ridge meta-learner) with quantile bases and 95% confidence intervals
- Companion sale-probability (CVR) model — LightGBM classifier for sell-through ranking
- Leak-free training pipeline with artist-stratified temporal splits (split-before-fit)
- Data preprocessing pipeline with cleaning, validation, and transformation
- Standardized seat zone normalization for cross-venue learning
- FastAPI web serving layer for interactive price lookups
- 860+ automated tests across 49 modules

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ticketmaster API key](https://developer.ticketmaster.com/)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/ticket-price-predictor.git
cd ticket-price-predictor

# Install with uv
uv sync

# Copy environment template and add your API keys
cp .env.example .env
# Edit .env and set TICKETMASTER_API_KEY, LASTFM_API_KEY
```

### Verify Installation

```bash
# Run all checks (lint + typecheck + tests)
make check

# Or run individually:
make lint       # ruff check + format
make typecheck  # mypy
make test       # pytest
```

## Run the web app

1. Install deps: `pip install -e .`
2. Make sure a trained model exists at `data/models/lightgbm_combined-fixed.joblib` (the directory is gitignored — train one first via `make pipeline` + `python scripts/train_model.py`, or point `MODEL_PATH` at any other `*.joblib` artifact in `data/models/`).
3. Start the server: `make serve` (override the model path with `MODEL_PATH=path/to/your_model.joblib make serve`).
4. Open http://127.0.0.1:8000

The web app shows only events present in the loaded model's training corpus, and by
default only future-dated ones. The search box's free-text query (`q`) is a forgiving
substring match across artist + venue + city; use the dedicated city input for strict
city filtering when `q` produces noisy matches. The Tailwind CDN script is intended
for local prototyping only — do not deploy this UI to a public host as-is.

## Usage

### Data Collection

```bash
# Collect ticket listings for an artist
python scripts/collect_listings.py --artist "Bruno Mars" --max-events 3

# Ingest event metadata from Ticketmaster
python scripts/ingest_events.py --event-types concert --cities "Las Vegas" "New York"

# Run full pipeline
python scripts/run_pipeline.py --days-ahead 90 --event-types concert
```

### Model Training

```bash
# Train LightGBM model
python scripts/train_model.py --model lightgbm --version v13

# Train with Optuna hyperparameters
python scripts/train_model.py --from-study lightgbm_aggressive --version v14

# Train with preprocessing enabled
python scripts/train_model.py --model lightgbm --version v13 --preprocess

# Hyperparameter tuning
python scripts/tune_model.py --n-trials 50
```

### Predictions

```bash
# Predict prices for all zones
python scripts/predict.py --artist "BTS" --city "Tampa" --all-zones
```

## Project Structure

```
ticket-price-predictor/
├── src/ticket_price_predictor/
│   ├── api/                  # External API clients (Ticketmaster, Setlist.fm)
│   ├── schemas/              # Pydantic data models
│   ├── storage/              # Parquet storage layer (repositories)
│   ├── ingestion/            # Data collection services
│   ├── scrapers/             # VividSeats/StubHub web scrapers
│   ├── normalization/        # Seat zone normalization
│   ├── validation/           # Data quality checks
│   ├── preprocessing/        # Data cleaning & transformation pipeline
│   ├── popularity/           # Popularity aggregation (YouTube Music, Last.fm)
│   ├── synthetic/            # Synthetic data generation
│   ├── serving/              # FastAPI web serving layer
│   └── ml/                   # Machine learning pipeline
│       ├── features/         # Feature extractors (11 domains, 80+ features)
│       ├── models/           # Baseline, LightGBM, quantile, stacking ensemble, CVR
│       ├── training/         # Split-first training pipeline
│       ├── tuning/           # Optuna hyperparameter optimization
│       └── inference/        # Prediction service
├── scripts/                  # CLI entry points
├── tests/                    # 860+ automated tests
└── data/
    ├── raw/                  # Raw data (events, listings, snapshots)
    └── models/               # Trained model artifacts
```

## Data Flow

```
Ticketmaster API → Event metadata (discovery)
VividSeats/StubHub → Ticket listings (actual resale prices)
YouTube Music/Last.fm → Artist popularity signals
                    ↓
        Preprocessing Pipeline
                    ↓
        Split raw data (artist-stratified, temporal)
                    ↓
        Feature Pipeline (fit on train only)
                    ↓
        LightGBM Model → Price Prediction (with 95% CI)
```

## Feature Engineering (80+ features across 11 domains)

The strongest signal comes from event/zone/section-level target encoding; all group-level
statistics use Bayesian smoothing to prevent small-sample memorization.

| Domain | Description |
|--------|-------------|
| Event/zone/section pricing | Target-encoded event, zone, and section median prices — dominant features (~76% importance) |
| Artist | Historical avg/median price, event count, artist×zone and artist×region encoding |
| Regional | City/country/global price ratios and market strength (Bayesian-smoothed) |
| Event | City tier, day of week, season, venue capacity, market saturation |
| Seating | Zone encoding (floor to balcony), row number, price ratio, is_premium |
| Time-series | Days to event, urgency buckets |
| Popularity | YouTube Music / Last.fm integrated popularity score + availability flag |
| Venue | Venue avg/median/std price (Bayesian-smoothed), is_known |
| Interactions | Artist×zone, urgency×zone, artist×venue cross-domain terms |
| Listing structural | Structural signals parsed from section/listing names |
| Snapshot | Inventory change rate, zone price trend, listing count (temporal) |

## Training Pipeline

The training pipeline prevents data leakage by splitting before feature extraction (split-before-fit):

1. Filter invalid prices (<$10) and cap outliers at the 95th percentile
2. Normalize artist aliases and city names for consistent grouping
3. Split raw data temporally with artist stratification (`split_raw()`)
4. Fit the feature pipeline on training data only (Bayesian-smoothed group stats)
5. Transform train/val/test independently; remove zero-variance features
6. Log-transform the target (`np.log1p`) and price-based features
7. Train the stacking ensemble (LightGBM GBDT + Huber loss, early stopping patience=100) with temporally-sorted OOF folds for the meta-learner

## Configuration

Environment variables (set in `.env`):

| Variable | Required | Description |
|----------|----------|-------------|
| `TICKETMASTER_API_KEY` | Yes | Ticketmaster Discovery API key |
| `LASTFM_API_KEY` | No | Last.fm API key (for popularity features) |
| `DATA_DIR` | No | Data storage directory (default: `./data`) |

## Seat Zones

Standardized zones for cross-venue learning:

| Zone | Price Ratio | Encoding |
|------|-------------|----------|
| Floor/VIP | 100% | 3 |
| Lower Tier | 70% | 2 |
| Upper Tier | 45% | 1 |
| Balcony | 25% | 0 |

## Development

```bash
make format     # Auto-format code (ruff)
make lint       # Lint check
make typecheck  # mypy type checking
make test       # Run pytest
make check      # All of the above
```

## Roadmap

- [x] **M0**: Foundation (repo structure, API client, schemas)
- [x] **M1**: Data pipeline (batch ingestion, storage, validation)
- [x] **M2**: Feature engineering (80+ features, 11 domains)
- [x] **M3**: Model training (stacking ensemble, quantile regression, leak-free pipeline)
- [x] **M4**: Sale-probability (CVR) model + FastAPI web serving layer
- [ ] **M5**: Backtesting & production deployment

## License

Proprietary - All rights reserved
