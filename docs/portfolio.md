# Ticket Price Predictor

**End-to-end ML system for predicting secondary-market ticket prices at the seat-zone level.**

Built a production data pipeline and gradient boosting model that ingests live resale ticket data from multiple marketplaces, extracts 54 engineered features across 8 domains, and produces per-zone price predictions with 95% confidence intervals — all designed to help buyers identify optimal purchase timing.

## Motivation

This project started from my own experience buying resale concert tickets and the lack of a clear, quantitative basis for deciding: *"Should I wait two more days for a lower price, or buy now?"* Price-drop patterns vary significantly depending on the artist, seat location, city, and time-to-event. Simple signals like average price or current lowest price are not sufficient to forecast future prices.

The goal: develop a system that predicts a **stable buying window** — a purchase timing where ticket prices are low enough while a reasonable range of seats is still available.

## System Architecture

The system spans five layers with strict dependency enforcement (no upward imports):

```
Data Sources          Ticketmaster API | VividSeats Scraper | YouTube Music | Last.fm
                                       |
Schemas & Storage     Pydantic v2 models + Hive-partitioned Parquet (PyArrow)
                                       |
Preprocessing         11-stage composable pipeline (validate → normalize → enrich → flag)
                                       |
Feature Engineering   8 extractor domains, 54 features, two-stage pipeline
                                       |
ML Pipeline           LightGBM (DART) + Quantile regression → price + 95% CI
```

### Data Collection

- **Ticketmaster Discovery API** for event metadata (venue, date, capacity, event type)
- **VividSeats scraper** using Playwright with stealth mode — intercepts the internal listings API via a network response listener rather than DOM scraping, capturing full listing payloads (section, row, price, quantity) in a single network call
- **YouTube Music** (ytmusicapi) and **Last.fm** for artist popularity signals (subscriber counts, view counts, listener counts, play counts)
- **Automated hourly collection** on AWS EC2 (t3.micro, systemd timer) with rsync-based deployment and data sync

### Preprocessing Pipeline

An 11-stage composable pipeline where each stage returns structured results (data, issues, metrics):

1. Schema validation
2. Text normalization (artist/venue/city aliases)
3. Event metadata joining (venue capacity)
4. Type conversion
5. Seat zone enrichment (raw section → standardized zone)
6. Temporal feature enrichment
7. Missing value imputation
8. Price outlier flagging (IQR + absolute bounds)
9. Duplicate detection (6-hour time window)
10. Temporal validation
11. Referential integrity checks

Individual stage failures are logged without aborting the full pipeline. Checkpoint support enables resume from any stage.

### Feature Engineering (54 features, 8 domains)

Features are extracted in two stages — base extractors run on raw data, then interaction extractors run on the concatenated base features:

| Domain | Features | Key Signals |
|--------|----------|-------------|
| Event Pricing | 5 | Event/zone-level Bayesian target encoding (**60%+ importance**) |
| Performer | 8 | Artist historical avg/median price, event count, artist x zone median |
| Regional | 7 | City/country/global price stats with fallback chain |
| Event | 8 | City tier, day of week, season, venue capacity bucket |
| Popularity | 6 | YouTube + Last.fm normalized score and tier |
| Seating | 6 | Zone ordinal encoding, row number, zone price ratio |
| Time Series | 10 | Days to event, urgency buckets, 7d/30d momentum, volatility |
| Venue | 3 | Venue avg/median price (Bayesian smoothed) |
| Interactions | 6 | artist x zone, urgency x zone, popularity x zone |

All group-level statistics use **Bayesian smoothing** (`smoothed = (n * group + m * global) / (n + m)`) with calibrated smoothing factors to prevent small-sample memorization.

### Training Pipeline

The central invariant: **raw data is split temporally before any feature extraction** to prevent data leakage.

1. Filter invalid prices (< $10), cap outliers at 95th percentile
2. Normalize city names and artist name aliases
3. **Split raw data** — artist-stratified temporal split (70/15/15). Artists with < 10 samples go entirely to training
4. **Fit feature pipeline on training data only**, then transform each split independently
5. Remove zero-variance features, log-transform price-based features
6. Train LightGBM with DART boosting and early stopping (patience=200)
7. Evaluate on held-out test set with inverse log-transform back to dollar scale

### Inference

The prediction service produces per-zone price estimates with confidence intervals and a directional signal (UP / DOWN / STABLE). A cold-start fallback handles unseen artists using popularity tier lookups and configurable zone defaults.

## Results

### Model Performance (v28)

| Metric | Value |
|--------|-------|
| MAE | $150.08 |
| MAPE | 41.0% |
| R² | 0.53 |
| RMSE | $237.35 |
| Training samples | 19,099 |
| Test samples | 4,104 |
| Events | 81 |
| Artists | 23 |

### Performance by Price Quartile

| Quartile | Price Range | MAE |
|----------|-------------|-----|
| Q1 (Budget) | < $85 | ~$34 |
| Q2 (Mid) | $85 – $165 | ~$70 |
| Q3 (Premium) | $165 – $440 | ~$145 |
| Q4 (VIP) | > $440 | ~$337 |

The model performs well on the budget-to-premium range most relevant to the "buying window" use case. High-value VIP tickets remain difficult due to sparse data and wide price variance.

### Improvement Journey

| Version | MAE | What Changed |
|---------|-----|-------------|
| v18 | $216.88 | Baseline before systematic improvements |
| v21 | $141.95 | Fixed data leakage, zone mapping bug, artist normalization (**34.6% improvement**) |
| v28 | $150.08 | Added EventPricingFeatureExtractor, artist x zone encoding, larger dataset |

The v18 → v21 improvement came primarily from fixing a data leakage bug (feature pipeline was being fit on the full dataset before splitting) and a zone mapping bug (sections 400–499 misclassified). The slight v21 → v28 MAE increase reflects dataset growth — more events introduce harder, more diverse examples — not model regression.

## Key Technical Decisions

**Split-before-fit as an architectural invariant.** The original training function fit features on the full dataset before splitting — a textbook data leakage bug. Rather than just fixing the function, I restructured the entire training pipeline around the `ModelTrainer.train()` flow that enforces temporal splitting before any feature computation. The old function is deprecated with an explicit warning.

**Bayesian smoothing everywhere.** Every group-level statistic (artist, event, zone, city, venue) uses Bayesian smoothing rather than raw sample means. With only 2–3 listings per section or 10 events per artist, raw means memorize noise. Smoothing factors are calibrated per domain (event pricing = 20, artist = 50, regional = 75, venue = 200).

**Deduplication deliberately disabled.** Removing apparent duplicate listings degraded MAE by $6.79. Counter-intuitively, repeated listings at the same price reflect market consensus — a genuine price signal, not noise. This was discovered through systematic ablation studies.

**Network interception over DOM scraping.** The VividSeats scraper registers a Playwright response listener for the internal listings API endpoint, capturing the full JSON payload in a single network call. This is more reliable and faster than scrolling the page and parsing DOM elements.

**Frozen configuration as single source of truth.** All 40+ magic numbers (cold-start defaults, smoothing factors, tier multipliers, capacity buckets) live in a single frozen dataclass (`MLConfig`) rather than scattered across modules.

## Lessons from Ablation Studies

Several intuitive improvements actually hurt performance, documented in the project's issue tracker:

- **Deduplication hurts** (−$6.79 MAE): Repeated listings are a demand signal, not noise
- **Segment-aware outlier capping hurts** (−$6.07 MAE): Per-segment sample sizes too small for stable quantile estimates
- **Section-level target encoding hurts**: Too sparse (2–3 listings per section) — zone-level is the right granularity
- **Listing features (source, quantity) add noise** (~$2 MAE): Disabled after ablation showed no benefit

These findings reinforced that with a small dataset (81 events, 23 artists), simpler aggregation granularities outperform finer-grained ones.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| ML | LightGBM, scikit-learn, Optuna |
| Data | Pydantic v2, PyArrow, Parquet (Hive-partitioned) |
| Scraping | Playwright, playwright-stealth, httpx |
| Popularity | ytmusicapi, Last.fm API |
| Infrastructure | AWS EC2 (t3.micro), systemd timers, rsync |
| Quality | mypy (strict), ruff, pytest (~370 tests), pytest-asyncio |
| Package Management | uv |

## What's Next

- **More data** — the primary bottleneck. The automated EC2 pipeline is continuously expanding the dataset. More events and artists will improve the model's ability to generalize, especially for high-value tickets.
- **Temporal price trajectories** — modeling how prices evolve over time leading up to an event, rather than point-in-time prediction, to directly identify the buying window.
- **User-facing application** — a lightweight web interface where a user can enter an event and receive a buy/wait recommendation with a confidence level.
