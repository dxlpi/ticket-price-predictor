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
