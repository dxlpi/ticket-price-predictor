# Data Collection Patterns

## Rule
Three patterns are essential for robust data collection: (1) atomic parquet writes via `.parquet.tmp` + rename, (2) include `listing_price` in the dedup hash composite key so price changes are preserved as new records, and (3) always use `datetime.now(UTC)` — never bare `datetime.now()`.

## Why
Without atomic writes, a crash mid-write leaves a corrupt parquet file. Without price in the hash, silently deduplicating re-scraped listings at the same seat discards price change signal worth ~$6.79 MAE. Without UTC, naive datetimes cause silent timezone bugs in `days_to_event` calculations.

## Pattern

**Atomic write (parquet.py)**:
```python
tmp = path.with_suffix(".parquet.tmp")
pq.write_table(table, tmp, schema=schema)
tmp.rename(path)
```

**Dedup hash with price (repository.py)**:
```python
key_parts = [
    listing.listing_id,
    listing.event_id,
    listing.section,
    listing.row,
    f"{listing.listing_price:.2f}",  # price change = new record
]
```

**Timezone-aware datetime**:
```python
from datetime import UTC, datetime
timestamp = datetime.now(UTC)  # NOT datetime.now() or datetime.utcnow()
```

## Hash Migration Note
Changing the hash algorithm makes all existing `.listing_hashes.txt` stale. Delete it on deploy — the first run will re-collect all listings cleanly.
