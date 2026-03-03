# Ticket Price Predictor

ML system for predicting secondary-market ticket prices at the seat-zone level using historical time-series data, demand signals, and event context.

## Overview

The system targets the U.S. market (concerts, sports, theater) and predicts resale ticket prices by combining data from multiple sources with a LightGBM gradient boosting model.

**Current Model Performance**: MAE $94.60 | MAPE 22.4% | R² 0.66

## Features

- Event discovery via Ticketmaster Discovery API
- Real-time price scraping from VividSeats & StubHub (Playwright-based)
- Artist popularity aggregation from YouTube Music and Last.fm
- 44 engineered features across 7 domains (artist, event, seating, time-series, momentum, popularity, regional)
- LightGBM model with quantile regression variant for 95% confidence intervals
- Leak-free training pipeline with artist-stratified temporal splits
- Data preprocessing pipeline with cleaning, validation, and transformation
- Standardized seat zone normalization for cross-venue learning
- 301 automated tests

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
│   └── ml/                   # Machine learning pipeline
│       ├── features/         # 7 feature extractors (44 features)
│       ├── models/           # Baseline, LightGBM, Quantile LightGBM
│       ├── training/         # Split-first training pipeline
│       ├── tuning/           # Optuna hyperparameter optimization
│       └── inference/        # Prediction service
├── scripts/                  # CLI entry points
├── tests/                    # 301 automated tests
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

## Feature Engineering (44 features)

| Domain | Features | Description |
|--------|----------|-------------|
| Artist | 7 | Historical avg/median price, event count, premium ratio |
| Popularity | 6 | YouTube Music/Last.fm integrated popularity score |
| Regional | 7 | City/country/global price ratios, market strength |
| Event | 8 | City tier, day of week, season, venue capacity |
| Seating | 6 | Zone encoding (floor to balcony), row number, price ratio |
| Time-series | 6 | Days to event, urgency buckets |
| Momentum | 4 | 7d/30d price momentum, volatility |

## Training Pipeline

The training pipeline prevents data leakage by splitting before feature extraction:

1. Cap price outliers at 99th percentile
2. Split raw data temporally with artist stratification
3. Fit feature pipeline on training data only
4. Transform train/val/test independently
5. Train LightGBM with early stopping (patience=100)

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
- [x] **M2**: Feature engineering (44 features, 7 domains)
- [x] **M3**: Model training (LightGBM, quantile regression, leak-free pipeline)
- [ ] **M4**: Backtesting & validation
- [ ] **M5**: Deployment

## License

Proprietary - All rights reserved
