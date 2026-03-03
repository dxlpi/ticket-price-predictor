---
paths:
  - src/ticket_price_predictor/ml/**
---

# ML Pipeline Rules

## Leak Prevention

1. **Split-before-fit**: `ModelTrainer.train()` must call `splitter.split_raw()` BEFORE `pipeline.fit()`. Never reverse this order.
2. **Fit on train only**: `pipeline.fit(train_df)` — never pass val or test data to fit.
3. **Transform independently**: Each split gets `pipeline.transform()` separately after fit.
4. **Target encoding isolation**: `EventPricingFeatureExtractor` uses Bayesian smoothing (factor=20) to prevent memorization of event-level prices.

## Feature Extraction Contract

Every feature extractor must:
- Inherit from `FeatureExtractor` base class
- Implement `fit(df: pd.DataFrame) -> None` (learn statistics from training data)
- Implement `extract(df: pd.DataFrame) -> pd.DataFrame` (produce feature columns)
- Implement `feature_names -> list[str]` property (list all output column names)
- Be registered in `FeaturePipeline.__init__()` (base extractor or post-extractor)

## Bayesian Smoothing

All group-level statistics must use Bayesian smoothing:
```
smoothed = (n * group_stat + m * global_stat) / (n + m)
```

Current smoothing factors by module:
| Module | Factor | Rationale |
|--------|--------|-----------|
| `event_pricing.py` | 20 | Events have few listings per zone |
| `artist_stats.py` | 50 | Artists have moderate sample sizes |
| `regional.py` | 75 | Cities have larger, more stable samples |
| `venue.py` | 200 | Venues have the largest samples |

## Target Transform

- Target: `np.log1p(listing_price)` for training
- Predictions: `np.expm1(prediction)` to convert back to dollar scale
- Price-based features are also log-transformed for alignment with log target
