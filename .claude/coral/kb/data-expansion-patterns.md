# Data Expansion: Schema Evolution & Event Type Threading
Promoted: 2026-03-23 | Updated: 2026-03-23

## Rule
When adding a new nullable field to TicketListing, thread it through 5 layers: (1) `parquet_schema()` with `nullable=True`, (2) `_dict_to_listing()` via `data.get("field")`, (3) `DataLoader._listings_to_dataframe()` record dict, (4) `create_listing_from_scraped()` from ScrapedEvent, (5) ML feature extractor encoding. Missing any layer silently zeros out the feature.

## Why
The `event_type` field was stored in parquet but produced constant 0 in ML features because `DataLoader._listings_to_dataframe()` didn't include it in the record dict — the DataFrame column simply never existed, so `EventFeatureExtractor.extract()` always hit the `fillna(0)` branch silently.

## Pattern
Right — all 5 layers:
```python
# 1. parquet_schema
pa.field("event_type", pa.string(), nullable=True)
# 2. _dict_to_listing
event_type=data.get("event_type")
# 3. DataLoader
"event_type": listing.event_type,
# 4. create_listing_from_scraped
event_type=event.event_type
# 5. EVENT_TYPE_MAP (lowercase keys match StrEnum values)
EVENT_TYPE_MAP = {"concert": 0, "sports": 1, "theater": 2, "comedy": 3}
```

Wrong — missing DataLoader step (was pre-existing gap):
```python
# record dict in _listings_to_dataframe had no event_type key
# → DataFrame column absent → fillna(0) for every row
```

Also: inference path must lowercase `event_type` to match training data:
```python
"event_type": event_type.lower()  # predictor.py
```
