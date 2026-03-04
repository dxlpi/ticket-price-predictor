# Model Card: Ticket Price Predictor

## Model Details

| Field | Value |
|-------|-------|
| **Model type** | LightGBM with DART boosting |
| **Version** | v28 |
| **Framework** | LightGBM 4.3+, scikit-learn 1.4+ |
| **Task** | Regression — predict secondary-market ticket listing price |
| **Input** | Event metadata + listing attributes (54 features after zero-variance removal; v30 pipeline: 67 raw features across 10 domains) |
| **Output** | Predicted price (USD) + 95% confidence interval + price direction |
| **Training time** | ~47 seconds |
| **Artifact** | `data/models/lightgbm_v28.joblib` |

## Intended Use

Predict resale ticket prices for U.S. concerts, sports, and theater events at the seat-zone level. Designed for price estimation before purchase, not real-time trading.

## Training Data

| Metric | Value |
|--------|-------|
| Total listings | ~27,000 |
| Training samples | 19,099 |
| Test samples | 4,104 |
| Events | 81 |
| Artists | 23 |
| Date range | 2026-01-30 to 2026-03-02 (32 days) |
| Sources | VividSeats (primary), StubHub (secondary) |
| Split strategy | Temporal with artist stratification |

## Performance (v28)

| Metric | Value |
|--------|-------|
| MAE | $150.08 |
| MAPE | 41.0% |
| R² | 0.53 |
| RMSE | $237.35 |

### Performance by Price Quartile

| Quartile | Price Range | Approximate MAE |
|----------|------------|-----------------|
| Q1 | <$85 | ~$34 |
| Q2 | $85-$165 | ~$70 |
| Q3 | $165-$440 | ~$145 |
| Q4 | >$440 | ~$337 |

## Top Features

| Feature | Importance |
|---------|-----------|
| `event_zone_median_price` | 60.2% |
| `event_median_price` | 15.4% |
| `venue_median_price` | 6.2% |

Top 3 features account for ~82% of total importance.

## Limitations

1. **Small dataset**: 81 events across 23 artists limits generalization to unseen artists/venues
2. **High-price inaccuracy**: Q4 tickets (>$440) have MAE ~$337 — 10x the cheapest quartile
3. **Feature dominance**: Model relies heavily on target-encoded event/zone medians (82% importance) — essentially a smoothed lookup table
4. **Geographic bias**: Data concentrated in U.S. markets only
5. **Temporal scope**: 32 days of data; seasonal patterns not captured
6. **Feature pipeline not serialized**: Fitted caches (artist stats, regional stats) not saved with model artifact — inference uses unfitted defaults

## Ethical Considerations

- No personal data collected — only public listing prices and event metadata
- Predictions should not be used for automated ticket purchasing (scalping)
- Price predictions may reflect existing market biases in premium pricing
