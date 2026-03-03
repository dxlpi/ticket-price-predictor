# 005: Listing-Level Features Add Noise

**Status:** Resolved (features disabled)
**Severity:** Medium
**Area:** ML Features (`ml/features/listing.py`)

## Problem

Listing-level features (ticket source, quantity available) were expected to provide useful signal about pricing. Ablation testing showed they actually degraded model performance, adding ~$2 MAE.

## Impact

- Including listing features increased prediction error
- Additional features increased model complexity without improving accuracy

## Root Cause

Listing-level attributes like source marketplace and quantity are weakly correlated with price after controlling for event, zone, and artist features. They introduced noise that the model attempted to fit, slightly overfitting to training patterns that didn't generalize.

## Solution

Disabled listing features in the training pipeline via configuration:
```python
pipeline_kwargs={'include_listing': False}
```

The feature extractor code was preserved but excluded from the active pipeline.

## Outcome

- ~$2 MAE improvement by removing 4 noisy features
- Established ablation testing as a standard practice for feature validation
- Best config documented: `include_listing=False`
