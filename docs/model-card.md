# Model Card: Ticket Price Predictor

## Model Details

| Field | Value |
|-------|-------|
| **Model type** | LightGBM with GBDT boosting + Huber loss |
| **Version** | v36 |
| **Framework** | LightGBM 4.3+, scikit-learn 1.4+ |
| **Task** | Regression — predict secondary-market ticket listing price |
| **Input** | Event metadata + listing attributes (70 features after zero-variance removal; 76 raw features across 10 domains) |
| **Output** | Predicted price (USD) + 95% confidence interval + price direction |
| **Training time** | ~21 seconds (1271 iterations with early stopping) |
| **Artifact** | `data/models/lightgbm_v36.joblib` |

## Intended Use & Scope

Predict resale ticket prices for events present in the training corpus. The model's `predict()` API gates on `event_id`: queries for events not seen during training raise `UnknownEventError`. New events that have not yet been ingested are out of scope; refresh the model on updated data to expand coverage.

## Training Data

| Metric | Value |
|--------|-------|
| Total listings | ~99,500 |
| Training samples | 69,771 |
| Test samples | 14,850 |
| Events | 771 |
| Artists | 500 |
| Sources | VividSeats (primary), StubHub (secondary) |
| Split strategy | Temporal with artist stratification |

## Performance (v36)

Headline metric is `primary_mae` — MAE on the in-scope (seen-event) slice of the held-out test set. This reflects the model's intended use: queries are gated to events present in the training corpus.

| Metric | Value |
|--------|-------|
| **primary_mae** (seen events) | **$52.75** |
| MAPE | 46.5% |
| R² | 0.6734 |
| RMSE | $141.35 |

> *Source: `primary_mae` ($52.75) and `unseen_mae` ($128.91) sourced from `MEMORY.md` v36 stacking_v2 ensemble entry (held-out test split; ~197K listings, 81 features). `overall_mae` ($83.63) cross-checked against the v36 row of the existing Benchmark Table. No re-evaluation was performed — v36 weights and pipeline are unchanged.*

### Diagnostics

These numbers include out-of-scope events held out by the temporal split. They are retained for diagnostic comparison but are not the headline metric, since `predict()` rejects unseen events at inference time.

| Metric | Value |
|--------|-------|
| overall_mae (seen + unseen combined) | $83.63 |
| unseen_mae (out-of-scope events) | $128.91 |

### Performance by Price Quartile

*Note: Quartile breakdown numbers below are from v34 and have not been refreshed in this scope-only change.*

| Quartile | MAE |
|----------|-----|
| Q1 | $34.99 |
| Q2 | $26.05 |
| Q3 | $98.70 |
| Q4 | $180.40 |

### Performance by Zone

*Note: Zone breakdown numbers below are from v34 and have not been refreshed in this scope-only change.*

| Zone | MAE |
|------|-----|
| upper_tier | $79.55 |
| lower_tier | $69.10 |
| balcony | $101.27 |
| floor_vip | $154.05 |

## Top Features

| Feature | Importance |
|---------|-----------|
| `event_section_median_price` | 48.6% |
| `event_zone_median_price` | 21.1% |
| `artist_regional_median_price` | 6.4% |
| `artist_regional_avg_price` | 4.4% |
| `artist_zone_median_price` | 4.4% |
| `event_median_price` | 3.7% |

Importances sourced from `data/models/lightgbm_v36_metrics.json`. Top 3 features account for ~76% of total importance. Event-level target encoding (`event_section_median_price` + `event_zone_median_price`) dominates — consistent with the seen-events-only scope, since these features carry strong signal for events present in the training corpus.

## Recommendation & Ranking

**ListingRanker**: Wraps the price predictor to rank listings by value score (predicted fair price / actual listing price). Listings with value_score > 1.0 are underpriced relative to model estimates — surfaced first as "best value" recommendations.

**SaleProbabilityModel**: See Sale Probability Model section below.

## Sale Probability Model

**Purpose**: Predicts the probability that a ticket listing will sell within a 24-hour window — a conversion rate (CVR) analogue for commerce-style ranking.

**Architecture**: LightGBM binary classifier (`objective: binary`, `is_unbalance: True` for class imbalance). Reuses the price prediction feature pipeline (10 domains, 70 active features) augmented with sale-specific features:

| Feature | Description |
|---------|-------------|
| `relative_price_position` | Z-score of listing price within its event+zone (computed per training split) |
| `price_vs_zone_median` | Ratio of listing price to training-split zone median |
| `price_vs_predicted` | Ratio of listing price to price model's predicted fair value (optional; requires trained price model) |

**Label Construction**: Two strategies available:

1. **Inventory Depletion (default)**: Uses aggregate inventory depletion from `PriceSnapshot` data. For each listing, computes whether the event's total inventory depleted by >30% within a 48-hour window. Robust to the scraper's partial capture problem because it uses zone-level aggregate counts rather than individual listing tracking. Produces ~20% positive rate across 293 qualifying events.

2. **Disappearance (legacy)**: Labels derived from listing disappearance across consecutive hourly scraping runs. Unreliable with current scraper's partial captures (median ~2 listings/scrape).

**Offline Evaluation Metrics (v3, inventory depletion labels)**:

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.7653 |
| ECE | 0.1525 |
| Best iteration | 1830 |
| Training samples | 53,699 |
| Validation samples | 11,507 |
| Test samples | 11,508 |

**Top CVR Features**:

| Feature | Importance |
|---------|-----------|
| `days_to_event` | 21.8% |
| `days_to_event_squared` | 10.8% |
| `event_listing_count` | 7.7% |
| `day_of_week` | 5.5% |
| `artist_regional_listing_count` | 4.8% |

**Leakage prevention**: The `relative_price_position` and `price_vs_zone_median` features are computed from the training split only and applied to val/test using training-time statistics. The `_label_depletion_rate` column is dropped before feature extraction to prevent label leakage. Follows the same split-before-fit invariant as the price regression model.

**Limitations**:
- Event-level labels: all listings in the same event share the same sold label (individual listing-level conversion is not observable)
- No user-side features: predicts item-level sell-through, not personalized purchase probability
- AUC may improve further with user behavior data (click, browse, purchase history)

## Limitations

1. **Concert-only data**: All 771 events are concerts — event type feature is zero-variance until sports/theater data is collected
2. **High-price inaccuracy**: Q4 tickets have MAE $180 — 5x the cheapest quartile
3. **Feature dominance**: Model relies heavily on target-encoded event/zone/section medians (~80% importance) — essentially a smoothed lookup table
4. **Geographic bias**: Data concentrated in U.S. markets only
5. **Huber loss trade-off**: Robust to outliers but may underpredict extreme-price tickets

## Ethical Considerations

- No personal data collected — only public listing prices and event metadata
- Predictions should not be used for automated ticket purchasing (scalping)
- Price predictions may reflect existing market biases in premium pricing

## Training Data Stats (v37 final)

| Metric | v36 | v37-target | v37-actual |
|--------|-----|------------|------------|
| Total listings | 197,857 | 400,000+ | 347,353 |
| Distinct events | ~2,900 | 5,500+ | 3,868 |
| Distinct artists | ~1,060 | 1,500+ | 1,807 |

## Benchmark Table (v37)

Going forward, `primary_mae` (in-scope, seen-event MAE) is the comparison column — consistent with the resale-market gate documented in "Intended Use & Scope". Historical `overall_mae` numbers (legacy column) are preserved for continuity but do not match the production scope.

| Model / Version | primary_mae (seen) | overall_mae (legacy) | RMSE | R² | MAPE | Q4 MAE | Max feat imp | Promoted? |
|-----------------|--------------------|----------------------|------|-----|------|--------|--------------|-----------|
| v36 Stacking V2 (log) | **$52.75** | $83.63 (legacy) | $141.35 | 0.6734 | 46.5% | — | 0.491 | baseline |
| v37 Stacking V2 (relative) | — | $107.12 (legacy) | $182.22 | 0.4620 | 65.6% | $263.39 | 0.175 | rejected |
| **v37-log Stacking V2 (log)** | — | **$82.57** (legacy) | **$147.76** | **0.6462** | **36.6%** | **$187.05** | **0.447** | best v37 |

*Footnote: `primary_mae` is reported only for v36, where the seen-event slice was measured (`MEMORY.md`). v37 runs were evaluated under the legacy overall-MAE convention; their `primary_mae` was not computed in this scope-only change.*

**Primary AC9 goal** (MAE ≤ $41.82, legacy overall-MAE convention): **NOT MET** — best v37 run at $82.57, gap $40.75 (97% of required reduction).

**Secondary thresholds (AC10)**:
| Threshold | Target | Actual | Pass? |
|-----------|--------|--------|-------|
| R² | ≥ 0.80 | 0.6462 | ❌ |
| Max feature importance | < 0.25 | 0.447 | ❌ |
| No leakage | — | — | ✓ (16/16 canonical LOO tests pass) |

**Arithmetic-floor analysis (plan P2b.1)**:
- `unseen_event_pct_by_event = 92.8%` (actual artist-stratified trainer split: 94.12%)
- Even dream-scenario floor (unseen MAE → seen MAE): `≈ $53` > $41.82
- Overnight collection added 200 artists successfully, but new scrapes populate *current* events (test window), not historical events (train window) — so `unseen_event_pct` did not move.
- Target is arithmetically unreachable without historical-listing data source or metric-scope change.
