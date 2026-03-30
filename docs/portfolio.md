# Ticket Price Predictor

**End-to-end ML system for ticket price prediction, listing value ranking, and sale probability estimation — demonstrating the core modeling patterns of commerce recommendation systems: CTR/CVR prediction, value-based ranking, and iterative offline experimentation.**

Built a production data pipeline and gradient boosting model that ingests live resale ticket data from multiple marketplaces, extracts 76 engineered features across 10 domains (70 active after zero-variance removal), and produces per-zone price predictions with 95% confidence intervals — all designed to help buyers identify optimal purchase timing.

## Motivation

This project started from my own experience buying resale concert tickets and the lack of a clear, quantitative basis for deciding: *"Should I wait two more days for a lower price, or buy now?"* Price-drop patterns vary significantly depending on the artist, seat location, city, and time-to-event. Simple signals like average price or current lowest price are not sufficient to forecast future prices. This decision — buy now or wait? — is structurally analogous to the conversion optimization problem in commerce recommendation: given a product's predicted fair value and the user's context, predict whether a purchase will occur and surface the best value options first.

The goal: develop a system that predicts a **stable buying window** — a purchase timing where ticket prices are low enough while a reasonable range of seats is still available.

## System Architecture

The system spans five layers with strict dependency enforcement (no upward imports):

```
Data Sources          Ticketmaster API | VividSeats Scraper (multi-category) | YouTube Music | Last.fm
                                       |
Schemas & Storage     Pydantic v2 models + Hive-partitioned Parquet (PyArrow)
                                       |
Preprocessing         11-stage composable pipeline (validate → normalize → enrich → flag)
                                       |
Feature Engineering   10 extractor domains, 68 features (62 active), two-stage pipeline
                                       |
ML Pipeline           LightGBM (GBDT/DART) + Quantile regression → price + 95% CI | ListingRanker → value score + ranking | SaleProbabilityModel → sale probability (CVR analogue)
```

### Data Collection

- **Ticketmaster Discovery API** for event metadata (venue, date, capacity, event type)
- **VividSeats scraper** using Playwright with stealth mode — intercepts the internal listings API via a network response listener rather than DOM scraping, capturing full listing payloads (section, row, price, quantity) in a single network call across multiple event categories (concerts, sports, theater, comedy)
- **YouTube Music** (ytmusicapi) and **Last.fm** for artist popularity signals (subscriber counts, view counts, listener counts, play counts)
- **Automated hourly collection** on AWS EC2 (t3.micro, systemd timer) with rsync-based deployment and data sync
- **Autonomous discovery pipeline** that dynamically generates the artist/event watchlist from Ticketmaster events, VividSeats multi-category browse, and Last.fm chart/tag endpoints — replacing manual curation with zero-maintenance artist discovery

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

### Feature Engineering (68 features, 10 domains; 62 active after zero-variance removal)

Features are extracted in two stages and serve dual purpose — informing both the price prediction model and the sale-probability (CVR) classifier:

| Domain | Features | Key Signals |
|--------|----------|-------------|
| Event Pricing | 6 | Event/zone/section-level Bayesian target encoding (**57%+ importance**) |
| Performer | 8 | Artist historical avg/median price, event count, artist x zone median |
| Regional | 7 | City/country/global price stats with fallback chain |
| Event | 9 | City tier, day of week, season, venue capacity, event type encoding |
| Popularity | 7 | YouTube + Last.fm normalized score, tier, and data availability flag |
| Seating | 6 | Zone ordinal encoding, row number, zone price ratio, is_premium |
| Time Series | 6 | Days to event, urgency buckets (momentum disabled by default) |
| Venue | 4 | Venue avg/median/std price (Bayesian smoothed), is_known |
| Interactions | 7 | artist x zone, urgency x zone, artist_venue_price, popularity x zone |
| Snapshot | 4 | Inventory change rate, zone price trend, count, price range |

Momentum features (4) and listing features (4) available but disabled by default after ablation studies showed no benefit.

All group-level statistics use **Bayesian smoothing** (`smoothed = (n * group + m * global) / (n + m)`) with calibrated smoothing factors to prevent small-sample memorization.

### Training Pipeline

The central invariant: **raw data is split temporally before any feature extraction** to prevent data leakage.

1. Filter invalid prices (< $10), cap outliers at 95th percentile
2. Normalize city names and artist name aliases
3. **Split raw data** — artist-stratified temporal split (70/15/15). Artists with < 10 samples go entirely to training
4. **Fit feature pipeline on training data only**, then transform each split independently
5. Remove zero-variance features, log-transform price-based features
6. Train LightGBM with GBDT (default) or DART boosting, optional Huber loss, and early stopping (patience=200)
7. Evaluate on held-out test set with inverse log-transform back to dollar scale

### Inference

The prediction service produces per-zone price estimates with confidence intervals and a directional signal (UP / DOWN / STABLE). A ListingRanker wraps the price predictor to score and rank listings by value score (predicted fair price / actual price) — surfacing the best-value listings first, analogous to product ranking by predicted conversion value. A SaleProbabilityModel (AUC 0.77) predicts whether a listing will sell within a 48-hour window using event-level inventory depletion from snapshot data as the conversion signal (CVR analogue). The original disappearance-based labels produced AUC 0.48 due to a cross-event timestamp contamination bug (99% sold ratio); redesigning label construction around aggregate inventory depletion yielded a healthy 20% positive rate and meaningful signal. A cold-start fallback handles unseen artists using popularity tier lookups and configurable zone defaults. At save time, the fitted feature pipeline and log-transformed column list are serialized as companion files alongside the model, ensuring inference replicates exact training-time feature statistics.

## Results

### Model Performance (v34)

| Metric | Value |
|--------|-------|
| MAE | $84.76 |
| MAPE | 52.8% |
| R² | 0.6155 |
| RMSE | $140.77 |
| Training samples | 69,771 |
| Test samples | 14,850 |
| Events | 771 |
| Artists | 500 |

### Performance by Price Quartile

| Quartile | MAE |
|----------|-----|
| Q1 (Budget) | $34.99 |
| Q2 (Mid) | $26.05 |
| Q3 (Premium) | $98.70 |
| Q4 (VIP) | $180.40 |

The model performs well on the budget-to-mid range most relevant to the "buying window" use case (Q1–Q2 MAE under $35). High-value VIP tickets remain the hardest segment but improved from ~$325 to $180 with the larger dataset.

### Improvement Journey

| Version | Hypothesis | Method | Result | Decision |
|---------|-----------|--------|--------|----------|
| v18 | Baseline | Initial LightGBM pipeline | MAE $216.88 | Baseline |
| v21 | Feature pipeline fitted before split causes leakage | Fixed leakage, zone mapping bug, artist normalization | MAE $141.95 (**34.6% improvement**) | Keep — made split-before-fit an architectural invariant |
| v28 | Section-level target encoding adds pricing granularity | Added EventPricingFeatureExtractor, artist x zone encoding | MAE $150.08 | Keep — higher MAE reflects 2x larger, harder dataset |
| v29 | Bayesian smoothing makes section encoding robust | Bayesian-smoothed section encoding, venue_price_std, expanded aliases | MAE $148.27 | Keep — section encoding now 55.6% importance |
| v30 | Log-transform bug inflates scale-independent features | Fixed log-transform allowlist, added pipeline serialization, artist_venue_price | MAE $133.86 (**10.9% improvement**) | Keep — eliminated train/serve skew |
| v32 | GBDT+Huber converges faster and generalizes better than DART+L2 | Switched default to GBDT+Huber | MAE $149.50, MAPE 37.8% | Keep — 50x faster training, better MAPE despite larger dataset |
| v34 | Model underfits due to insufficient artist diversity | Autonomous discovery pipeline, 43→500 artists | MAE $84.76 (**43.3% improvement**) | Keep — regional features jumped from 2.6% to 17.7% importance |
| CVR v3 | Disappearance-based CVR labels broken (AUC 0.48, 99% sold ratio) | Redesigned labels using inventory depletion from snapshot data; fixed early stopping (logloss over AUC) | AUC **0.77**, ECE 0.15 | Keep — top features (days_to_event, event_listing_count) match commerce CVR intuition |

**Methodology**: Each version represents a hypothesis about the root cause of prediction error, tested with offline metrics and accepted/rejected based on quantitative evidence. Improvements that hurt MAE were reverted regardless of theoretical appeal.

The v18 → v21 improvement came primarily from fixing a data leakage bug (feature pipeline was being fit on the full dataset before splitting) and a zone mapping bug (sections 400–499 misclassified). The v21 → v28 MAE increase reflected dataset growth (81 vs ~40 events). The v29 → v30 improvement ($148.27 → $133.86) came from fixing a log-transform bug that was incorrectly transforming scale-independent features (std, ratio, cv), adding pipeline serialization to eliminate train/serve skew, and introducing the artist_venue_price interaction feature. The v30 → v32 MAE increase ($133.86 → $149.50) reflects the dataset nearly doubling in diversity (136 → 147 events) with MAPE actually improving (40.0% → 37.8%), indicating better generalization despite a harder prediction task. The v32 → v34 improvement ($149.50 → $84.76) is the largest single-version gain, driven entirely by data volume — the autonomous discovery pipeline expanded coverage from 43 to 500 artists, giving the model enough data to generalize across artists and regions. Artist regional features jumped from 2.6% to 17.7% importance, confirming that regional pricing patterns only emerge with sufficient geographic diversity.

## Key Technical Decisions

**Split-before-fit as an architectural invariant.** The original training function fit features on the full dataset before splitting — a textbook data leakage bug. Rather than just fixing the function, I restructured the entire training pipeline around the `ModelTrainer.train()` flow that enforces temporal splitting before any feature computation. The old function is deprecated with an explicit warning.

**Bayesian smoothing everywhere.** Every group-level statistic (artist, event, zone, city, venue) uses Bayesian smoothing rather than raw sample means. With only 2–3 listings per section or 10 events per artist, raw means memorize noise. Smoothing factors are calibrated per domain (event pricing = 20, artist = 50, regional = 75, venue = 200).

**Deduplication deliberately disabled.** Removing apparent duplicate listings degraded MAE by $6.79. Counter-intuitively, repeated listings at the same price reflect market consensus — a genuine price signal, not noise. This was discovered through systematic ablation studies.

**Network interception over DOM scraping.** The VividSeats scraper registers a Playwright response listener for the internal listings API endpoint, capturing the full JSON payload in a single network call. This is more reliable and faster than scrolling the page and parsing DOM elements.

**Frozen configuration as single source of truth.** All 40+ magic numbers (cold-start defaults, smoothing factors, tier multipliers, capacity buckets) live in a single frozen dataclass (`MLConfig`) rather than scattered across modules.

## Offline Experiment Results

The following experiments were run, measured against offline MAE/MAPE, and decisions made based on evidence rather than intuition:

Several intuitive improvements actually hurt performance, documented in the project's issue tracker:

- **Deduplication hurts** (−$6.79 MAE): Repeated listings are a demand signal, not noise
- **Segment-aware outlier capping hurts** (−$6.07 MAE): Per-segment sample sizes too small for stable quantile estimates
- **Section-level target encoding**: Initially hurt performance (v28, sparse data), but helps when Bayesian-smoothed with zone prior and gated behind flag (v29+, 57%+ feature importance)
- **Listing features (source, quantity) add noise** (~$2 MAE): Disabled after ablation showed no benefit

These findings reinforced that with a small dataset (81 events, 23 artists), simpler aggregation granularities outperform finer-grained ones.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| ML | LightGBM, scikit-learn, Optuna |
| Classification | LightGBM binary (sale probability, AUC 0.77) |
| Data | Pydantic v2, PyArrow, Parquet (Hive-partitioned) |
| Scraping | Playwright, playwright-stealth, httpx |
| Popularity | ytmusicapi, Last.fm API |
| Infrastructure | AWS EC2 (t3.micro), systemd timers, rsync |
| Quality | mypy (strict), ruff, pytest (656 tests), pytest-asyncio |
| Package Management | uv |

## What's Next

Remaining priorities: temporal price trajectories, cross-validation tuning, and a user-facing recommendation interface.
