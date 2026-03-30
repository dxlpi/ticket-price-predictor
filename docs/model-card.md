# Model Card: Ticket Price Predictor

## Model Details

| Field | Value |
|-------|-------|
| **Model type** | LightGBM with GBDT boosting + Huber loss |
| **Version** | v34 |
| **Framework** | LightGBM 4.3+, scikit-learn 1.4+ |
| **Task** | Regression — predict secondary-market ticket listing price |
| **Input** | Event metadata + listing attributes (70 features after zero-variance removal; 76 raw features across 10 domains) |
| **Output** | Predicted price (USD) + 95% confidence interval + price direction |
| **Training time** | ~21 seconds (1271 iterations with early stopping) |
| **Artifact** | `data/models/lightgbm_v34.joblib` |

## Intended Use

Predict resale ticket prices for U.S. concerts, sports, and theater events at the seat-zone level. Designed for price estimation before purchase, not real-time trading.

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

## Performance (v34)

| Metric | Value |
|--------|-------|
| MAE | $84.76 |
| MAPE | 52.8% |
| R² | 0.6155 |
| RMSE | $140.77 |

### Performance by Price Quartile

| Quartile | MAE |
|----------|-----|
| Q1 | $34.99 |
| Q2 | $26.05 |
| Q3 | $98.70 |
| Q4 | $180.40 |

### Performance by Zone

| Zone | MAE |
|------|-----|
| upper_tier | $79.55 |
| lower_tier | $69.10 |
| balcony | $101.27 |
| floor_vip | $154.05 |

## Top Features

| Feature | Importance |
|---------|-----------|
| `event_section_median_price` | 45.8% |
| `event_zone_median_price` | 21.7% |
| `artist_regional_median_price` | 12.9% |
| `artist_regional_avg_price` | 4.8% |
| `event_median_price` | 2.5% |

Top 3 features account for ~80% of total importance. Artist regional features gained significance with the larger, more diverse dataset (500 artists vs 43 in v32).

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
