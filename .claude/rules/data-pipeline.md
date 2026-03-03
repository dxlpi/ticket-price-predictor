---
paths:
  - src/ticket_price_predictor/scrapers/**
  - src/ticket_price_predictor/ingestion/**
  - src/ticket_price_predictor/storage/**
---

# Data Pipeline Rules

## Idempotency

- `ListingRepository` uses MD5 hash deduplication on composite key `(event_id, section, row, seat_from, seat_to, source)`
- Re-running a scrape for the same event must not produce duplicate records
- Hash file stored as `.listing_hashes.txt` alongside Parquet files

## Schema Evolution

- All Pydantic models have `parquet_schema()` classmethods — these MUST stay in sync
- New fields: add with default values to avoid breaking existing Parquet files
- Removed fields: never remove — mark as deprecated with `None` default
- Type changes: create new field instead of changing existing field type

## Scraper Resilience

- VividSeats scraper uses Playwright with stealth mode — preserve anti-detection args
- Random delays (3-5 seconds) between page loads — do not reduce
- Scraper errors accumulate in `ListingCollectionResult.errors` — never silently swallow
- Individual listing parse failures `continue` without crashing the event loop

## Storage

- Hive partitioning: `year=YYYY/month=MM/day=DD/` for listings
- Snapshots partitioned: `year=YYYY/month=MM/`
- Events stored flat: `events/events.parquet`
- Models: `data/models/lightgbm_v{N}.joblib` + `_metrics.json` pairs
