# Ticket Price Predictor

ML system for predicting ticket price changes at the seat-zone level using historical time-series data, demand signals, and event context.

## Overview

The system targets the U.S. market (concerts, sports, theater) and is designed to be legally safe, platform-agnostic, and extensible to global markets.

**Current Status**: M1 Data Pipeline complete

## Features

- Event metadata collection via Ticketmaster Discovery API
- Time-series price snapshot storage (Parquet format)
- Standardized seat zone normalization for cross-venue learning
- Data validation and quality checks
- Gradient boosting regression for price prediction (planned)

## Quick Start

### Prerequisites

- Python 3.11+
- [Ticketmaster API key](https://developer.ticketmaster.com/)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/ticket-price-predictor.git
cd ticket-price-predictor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with development dependencies
make install-dev

# Copy environment template and add your API key
cp .env.example .env
# Edit .env and set TICKETMASTER_API_KEY
```

### Initialize DVC (Data Version Control)

```bash
# Install DVC
pip install dvc

# Initialize DVC in the project
dvc init

# Track data directories
dvc add data/raw
dvc add data/processed
```

### Verify Installation

```bash
# Run all checks
make check

# Or run individually:
make lint       # Check code style
make typecheck  # Run mypy
make test       # Run tests
```

## Data Pipeline

### Run Full Pipeline

```bash
# Ingest events and collect snapshots
make pipeline
```

### Individual Steps

```bash
# Step 1: Ingest event metadata from Ticketmaster
make ingest-events

# Step 2: Collect price snapshots for tracked events
make collect-snapshots
```

### CLI Options

```bash
# Ingest events with options
python scripts/ingest_events.py \
    --days-ahead 90 \
    --event-types concert sports \
    --cities "Los Angeles" "New York" \
    --max-events 100

# Collect snapshots for specific events
python scripts/collect_snapshots.py --event-ids EVENT1 EVENT2

# Run full pipeline
python scripts/run_pipeline.py --days-ahead 90 --event-types concert
```

## Project Structure

```
ticket-price-predictor/
├── src/ticket_price_predictor/
│   ├── api/                  # External API clients
│   │   └── ticketmaster.py   # Ticketmaster Discovery API
│   ├── schemas/              # Data models
│   │   └── snapshots.py      # Event and price snapshot schemas
│   ├── storage/              # Data persistence
│   │   ├── parquet.py        # Parquet I/O utilities
│   │   └── repository.py     # Event and snapshot repositories
│   ├── ingestion/            # Data ingestion services
│   │   ├── events.py         # Event metadata ingestion
│   │   └── snapshots.py      # Price snapshot collection
│   ├── validation/           # Data quality checks
│   │   └── quality.py        # Validation rules
│   ├── normalization/        # Data normalization
│   │   └── seat_zones.py     # Seat zone mapping
│   └── config.py             # Settings management
├── scripts/
│   ├── ingest_events.py      # Event ingestion CLI
│   ├── collect_snapshots.py  # Snapshot collection CLI
│   └── run_pipeline.py       # Full pipeline runner
├── tests/                    # Test suite
├── data/
│   ├── raw/                  # Raw snapshot data (DVC tracked)
│   │   ├── events/           # Event metadata (Parquet)
│   │   └── snapshots/        # Price snapshots (partitioned Parquet)
│   └── processed/            # Processed features (DVC tracked)
└── fixtures/                 # Sample data for testing
```

## Configuration

Environment variables (set in `.env` or shell):

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TICKETMASTER_API_KEY` | Yes | - | API key from developer.ticketmaster.com |
| `DATA_DIR` | No | `./data` | Directory for data storage |
| `SNAPSHOT_INTERVAL_HOURS` | No | `6` | Hours between snapshot collections |

## Data Schemas

### EventMetadata

Event information from Ticketmaster:
- Event ID, type (concert/sports/theater), datetime
- Artist/team name
- Venue details (ID, name, city, capacity)

### PriceSnapshot

Point-in-time price observation:
- Event ID, seat zone, timestamp
- Price (min/avg/max)
- Inventory remaining
- Days until event

### SeatZone

Standardized zones for cross-venue learning:
- `floor_vip` - Floor / VIP sections
- `lower_tier` - Lower bowl
- `upper_tier` - Upper bowl
- `balcony` - Balcony / restricted view

## Data Storage

### Storage Layout

```
data/raw/
├── events/
│   └── events.parquet           # All event metadata
└── snapshots/
    └── year=2026/
        └── month=01/
            └── snapshots.parquet  # Partitioned by time
```

### Price Derivation

Since the Ticketmaster Discovery API only provides `priceRanges` (min/max) per event, zone-level prices are derived using configurable ratios:

| Zone | Price Ratio |
|------|-------------|
| Floor/VIP | 100% of max |
| Lower Tier | 70% of range |
| Upper Tier | 45% of range |
| Balcony | 25% of range |

## Development

```bash
# Format code
make format

# Run tests with coverage report
make test-cov

# Clean build artifacts
make clean
```

## Roadmap

- [x] **M0**: Foundation (repo structure, API client, schemas)
- [x] **M1**: Data pipeline (batch ingestion, storage, validation)
- [ ] **M2**: Feature engineering
- [ ] **M3**: Model training
- [ ] **M4**: Backtesting & validation
- [ ] **M5**: Deployment

## License

Proprietary - All rights reserved
