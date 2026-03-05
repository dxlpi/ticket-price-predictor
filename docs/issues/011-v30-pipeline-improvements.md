# 011: v30 Pipeline Improvements — Serialization, Log-Transform Fix, Enhanced Evaluation

**Status:** Resolved
**Severity:** High
**Area:** ML Pipeline, Training, Inference

## Problem

Three separate issues were limiting model performance and inference reliability:

1. **Pipeline serialization gap**: Fitted feature pipeline was not saved alongside the model, causing inference to recreate an unfitted pipeline that lost training-time statistics (Bayesian-smoothed medians, regional averages). This introduced silent train/serve skew.

2. **Log-transform bug**: The heuristic for log-transforming price-based features matched any column containing "price", "avg", or "median" — incorrectly transforming scale-independent derived statistics (`venue_price_std`, `zone_price_ratio`, `artist_price_cv`) that should remain in their natural scale.

3. **Evaluation blindness**: Only aggregate metrics (MAE, RMSE, R², MAPE) were reported, hiding the fact that Q4 tickets (>$316) had 12x worse MAE than Q1 (<$72) and that per-zone accuracy varied significantly.

## Solution

### Phase 1: Pipeline Serialization
- Added `save()`/`load()` to `FeaturePipeline` using joblib
- Added `__getstate__`/`__setstate__` to `PopularityFeatureExtractor` for pickle compatibility (unpicklable `YTMusic()` session)
- `ModelTrainer.save()` writes companion files: `{stem}_pipeline.joblib` + `{stem}_meta.json`
- `PricePredictor.from_path()` loads companion files when present, falls back gracefully for old models
- Guard: `expm1` applied only when `_log_transformed_cols is not None` (not `[]`)

### Phase 2: Log-Transform Allowlist
- Defined `_LOG_EXCLUDE_SUFFIXES = ("_std", "_cv", "_ratio", "_count", "_change", "_rate")` in `trainer.py`
- Imported in `objective.py` (single source of truth, no duplication)
- Filters out scale-independent features from log-transform

### Phase 3: Enhanced Evaluation
- `compute_metrics()` returns per-quartile MAE (Q1–Q4) and per-zone MAE
- `TrainingMetrics` schema extended with `quartile_mae` and `zone_mae` dict fields
- `print_metrics()` displays breakdown tables

### Additional
- GBDT + Huber loss option (`--loss huber` CLI flag)
- TemporalGroupCV infrastructure for Optuna (`--cv-tuning` flag)
- QuantileLightGBM predictions clipped after `expm1` to prevent negative prices

## Impact

| Metric | v29-section | v30 | Change |
|--------|------------|-----|--------|
| MAE | $148.27 | $133.86 | **-$14.41 (-9.7%)** |
| MAPE | 47.4% | 40.0% | **-7.4pp** |
| R² | 0.46 | 0.56 | **+0.10** |
| RMSE | $242.70 | $221.22 | **-$21.48** |

## Date
2026-03-04
