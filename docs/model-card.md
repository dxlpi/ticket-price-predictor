# Model Card: Ticket Price Predictor

## Model Details

| Field | Value |
|-------|-------|
| **Model type** | LightGBM with GBDT boosting + Huber loss |
| **Version** | v32 |
| **Framework** | LightGBM 4.3+, scikit-learn 1.4+ |
| **Task** | Regression — predict secondary-market ticket listing price |
| **Input** | Event metadata + listing attributes (62 features after zero-variance removal; 68 raw features across 10 domains) |
| **Output** | Predicted price (USD) + 95% confidence interval + price direction |
| **Training time** | ~3 seconds (145 iterations with early stopping) |
| **Artifact** | `data/models/lightgbm_v32.joblib` |

## Intended Use

Predict resale ticket prices for U.S. concerts, sports, and theater events at the seat-zone level. Designed for price estimation before purchase, not real-time trading.

## Training Data

| Metric | Value |
|--------|-------|
| Total listings | ~38,100 |
| Training samples | 26,568 |
| Test samples | 5,717 |
| Events | 147 |
| Artists | 43 |
| Sources | VividSeats (primary), StubHub (secondary) |
| Split strategy | Temporal with artist stratification |

## Performance (v32)

| Metric | Value |
|--------|-------|
| MAE | $149.50 |
| MAPE | 37.8% |
| R² | 0.5971 |
| RMSE | $236.36 |

### Performance by Price Quartile

| Quartile | Approximate MAE |
|----------|-----------------|
| Q1 | ~$34 |
| Q2 | ~$96 |
| Q3 | ~$144 |
| Q4 | ~$325 |

## Top Features

| Feature | Importance |
|---------|-----------|
| `event_section_median_price` | 65.6% |
| `event_zone_median_price` | 14.0% |
| `event_median_price` | 8.8% |
| `artist_regional_avg_price` | 2.6% |
| `event_zone_price_ratio` | 2.3% |

Top 3 features account for ~88% of total importance.

## Limitations

1. **Small dataset**: 147 events across 43 artists limits generalization to unseen artists/venues
2. **High-price inaccuracy**: Q4 tickets have MAE ~$325 — 10x the cheapest quartile
3. **Feature dominance**: Model relies heavily on target-encoded event/zone/section medians (~88% importance) — essentially a smoothed lookup table
4. **Geographic bias**: Data concentrated in U.S. markets only
5. **Huber loss trade-off**: Robust to outliers but may underpredict extreme-price tickets

## Ethical Considerations

- No personal data collected — only public listing prices and event metadata
- Predictions should not be used for automated ticket purchasing (scalping)
- Price predictions may reflect existing market biases in premium pricing
