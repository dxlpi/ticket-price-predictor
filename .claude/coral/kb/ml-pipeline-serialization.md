# ML Pipeline Serialization: Companion File Pattern

## Rule
When saving a trained model, also serialize the fitted `FeaturePipeline` and a `_meta.json` with `log_transformed_cols` as companion files. At inference time, load both before predicting. Guard all log-transform and expm1 operations on `_log_transformed_cols is not None` (not `[]`) to distinguish old models (no meta file) from new models with no price features.

## Why
Without saving the fitted pipeline, inference recreates an unfitted pipeline and loses training-time statistics (e.g. Bayesian-smoothed event medians, regional averages). This produces silent mispredictions. Without `_meta.json`, the inference path cannot replicate the feature log-transforms applied during training, causing a train/serve skew in log-space predictions.

## Pattern
```python
# trainer.py — save() naming convention
pipeline_path = output_dir / f"{model_type}_{version}_pipeline.joblib"
meta_path     = output_dir / f"{model_type}_{version}_meta.json"

# predictor.py — from_path() loading
stem = model_path.stem  # "lightgbm_v31"
pipeline_path = model_path.parent / f"{stem}_pipeline.joblib"
meta_path     = model_path.parent / f"{stem}_meta.json"
fitted_pipeline     = FeaturePipeline.load(pipeline_path) if pipeline_path.exists() else None
log_transformed_cols = json.loads(meta_path.read_text()).get("log_transformed_cols") if meta_path.exists() else None

# predict() — guard expm1 strictly on `is not None`, not truthiness
if self._log_transformed_cols is not None:
    preds = np.clip(np.expm1(preds), 0, None)
# QuantileLightGBM — clip all three outputs
if self._log_transformed_cols is not None:
    median = np.clip(np.expm1(median), 0, None)
    lower  = np.clip(np.expm1(lower),  0, None)
    upper  = np.clip(np.expm1(upper),  0, None)
```
`PopularityFeatureExtractor` holds an unpicklable `YTMusic()` HTTP session in `_service`. Add `__getstate__`/`__setstate__` to nullify `_service` on pickle while preserving `_artist_cache` contents.
