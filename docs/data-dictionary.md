# Data Dictionary

## Core Schemas

### TicketListing (`schemas/listings.py:9-90`)

Primary record for a resale ticket listing.

| Field | Type | Description |
|-------|------|-------------|
| `listing_id` | str | Unique identifier (MD5 hash of composite key) |
| `event_id` | str | Foreign key to event |
| `source` | str | Data source (default: "stubhub"; also "vividseats") |
| `timestamp` | datetime | When the listing was collected (UTC) |
| `event_name` | str | Event display name |
| `artist_or_team` | str | Primary artist or team name |
| `venue_name` | str | Venue name |
| `city` | str | City where event takes place |
| `event_datetime` | datetime | Scheduled event date/time (UTC) |
| `section` | str | Raw section name from source (e.g., "Floor VIP", "Section 200") |
| `row` | str | Row identifier within section |
| `seat_from` | str \| None | Starting seat identifier (string, not int) |
| `seat_to` | str \| None | Ending seat identifier (string, not int) |
| `quantity` | int | Number of tickets in listing |
| `face_value` | float \| None | Original ticket price (before resale markup) |
| `listing_price` | float | Current asking price per ticket (USD) |
| `total_price` | float | Total price including fees |
| `currency` | str | Currency code (default: "USD") |
| `days_to_event` | int | Days until event date (stored field, computed at creation) |
| `markup_ratio` | float \| None | Computed: `listing_price / face_value` (None if no face_value) |
| `seat_description` | str | Computed: human-readable seat string (e.g., "Section 100, Row A, Seats 1-4") |

### ScrapedEvent (`schemas/listings.py:93-104`)

Event metadata discovered from scraping search results.

| Field | Type | Description |
|-------|------|-------------|
| `stubhub_event_id` | str | Event identifier from scraping source |
| `event_name` | str | Event display name |
| `artist_or_team` | str | Primary artist or team |
| `venue_name` | str | Venue name |
| `city` | str | City |
| `event_datetime` | datetime | Event date and time |
| `event_url` | str | Event page URL on source site |
| `min_price` | float \| None | Lowest listed ticket price |
| `ticket_count` | int \| None | Number of available listings |

### ScrapedListing (`schemas/listings.py:107-118`)

Raw listing data extracted from scraping before enrichment.

| Field | Type | Description |
|-------|------|-------------|
| `listing_id` | str | Listing identifier |
| `section` | str | Raw section name |
| `row` | str | Row identifier |
| `seat_from` | str \| None | Starting seat |
| `seat_to` | str \| None | Ending seat |
| `quantity` | int | Ticket count |
| `price_per_ticket` | float | Price per ticket |
| `total_price` | float | Total price |
| `face_value` | float \| None | Original face value |

### EventMetadata (`schemas/snapshots.py:28-56`)

Enriched event data from Ticketmaster API.

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | str | Ticketmaster event ID |
| `event_type` | EventType | Enum: CONCERT, SPORTS, THEATER |
| `event_datetime` | datetime | Event date and time |
| `artist_or_team` | str | Artist, team, or show name |
| `venue_id` | str | Venue identifier |
| `venue_name` | str | Venue name |
| `city` | str | City where venue is located |
| `country` | str | Country code (default: "US") |
| `venue_capacity` | int \| None | Venue seating capacity |

### PriceSnapshot (`schemas/snapshots.py:59-102`)

Point-in-time snapshot of ticket prices for a seat zone.

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | str | Event identifier |
| `seat_zone` | SeatZone | Normalized seat zone |
| `timestamp` | datetime | Snapshot capture time |
| `price_min` | float | Minimum observed price in zone (>= 0) |
| `price_avg` | float \| None | Average price in zone |
| `price_max` | float \| None | Maximum price in zone |
| `inventory_remaining` | int \| None | Tickets remaining in zone |
| `days_to_event` | int | Days until event date |

Validator: `price_min <= price_avg <= price_max` enforced via `model_validator`.

## Enums

### EventType (`schemas/snapshots.py:11-16`)

StrEnum: `"concert"`, `"sports"`, `"theater"`

### SeatZone (`schemas/snapshots.py:19-25`)

StrEnum with 4 standardized seat zone categories.

| Value | StrEnum Value | ML Encoding (`seating.py:24-28`) | Price Ratio (`seat_zones.py:11-15`) |
|-------|---------------|----------------------------------|--------------------------------------|
| `FLOOR_VIP` | `"floor_vip"` | 3 (highest) | 1.0 (100% of max price) |
| `LOWER_TIER` | `"lower_tier"` | 2 | 0.70 (70% of max price) |
| `UPPER_TIER` | `"upper_tier"` | 1 | 0.45 (45% of max price) |
| `BALCONY` | `"balcony"` | 0 (lowest) | 0.25 (25% of max price) |

ML encoding is ordinal by price tier (higher = more expensive). Price ratios are fractions of the max price range, used in `map_price_range_to_zones()` for weighted interpolation.

Mapping logic: `normalization/seat_zones.py` — keyword and regex matching on raw `section` strings via `SeatZoneMapper`.

## Derived Fields (computed during feature extraction)

| Field | Computed By | Formula/Logic |
|-------|------------|---------------|
| `days_to_event` | `schemas/listings.py` (stored) | `max(0, (event_datetime - timestamp).days)` — computed at listing creation |
| `markup_ratio` | `schemas/listings.py` (computed_field) | `listing_price / face_value` (None if face_value missing) |
| `artist_zone_median_price` | `ml/features/performer.py` | Bayesian-smoothed median price for artist×zone |
| `event_zone_median_price` | `ml/features/event_pricing.py` | Bayesian-smoothed median price for event×zone (strongest feature, 60% importance) |
| `popularity_score` | `ml/features/popularity.py` | Weighted log-normalized 0-100 score from YouTube + Last.fm |
| `city_median_price` | `ml/features/regional.py` | Bayesian-smoothed median price for city |
| `seat_number` | `ml/features/listing_structural.py` (v38) | Numeric seat number from `seat_from`; -1 sentinel for missing/wildcard |
| `seat_span` | `ml/features/listing_structural.py` (v38) | `seat_to - seat_from + 1`; capped at 50; defaults to 1 |
| `is_low_seat_number` | `ml/features/listing_structural.py` (v38) | 1 if `seat_number ≤ 5` (proxy for aisle-adjacent seats) |
| `is_unknown_seat` | `ml/features/listing_structural.py` (v38) | 1 if `seat_from` was missing or `*` wildcard |
| `row_bucket_encoded` | `ml/features/listing_structural.py` (v38) | Ordinal {front:0, mid:1, back:2, ga:3, unknown:4} from `_row_bucket(row)` |
| `event_section_row_median_price` | `ml/features/listing_structural.py` (v38) | Bayesian-smoothed `(event, section, row_bucket)` mean price (m=8); LOO formula smooths toward `(event, section)` prior — see plan math spec § B |
| `event_section_row_listing_count` | `ml/features/listing_structural.py` (v38) | `log1p(n)` where n = training listings in `(event, section, row_bucket)` |
| `row_bucket_section_count` | `ml/features/listing_structural.py` (v38) | Distinct sections per `(event, row_bucket)` from training data; capped at 50 |

## Stacking V2 Meta-Features (v38)

| Meta-Feature | Computed By | Formula |
|--------------|------------|---------|
| `q75_tail` | `ml/models/stacking_v2.py:_build_meta_features` (v38) | `q75_pred · sigmoid((huber_pred − log1p($310)) / 0.3)` — sigmoid-gated quantile contribution; Ridge learns the tail-blend weight automatically. Only added when `include_quantile_bases=True` AND both `lgb_huber` and `quantile_75` base learners are present. |

## Storage Layout

```
data/
├── events/
│   └── events.parquet                    # All event metadata
├── listings/
│   └── year=YYYY/month=MM/day=DD/
│       └── listings.parquet              # Daily listing snapshots
├── snapshots/
│   └── year=YYYY/month=MM/
│       └── snapshots.parquet             # Price snapshots
├── models/
│   ├── lightgbm_v{N}.joblib             # Serialized model
│   └── lightgbm_v{N}_metrics.json       # Training metrics
└── .cache/
    └── popularity/
        └── popularity_cache.json         # Cached popularity scores (TTL-based)
```
