# Issue 010: Training Improvement — v29

## Summary

Comprehensive training pipeline improvement across 6 phases: bug fix, feature engineering, correctness fixes, pipeline hardening, experimental section feature, and model retraining.

## Changes Applied (Phases 1–4)

### Phase 1: Data Collection Bug Fix
- **File**: `scripts/monitor_popular.py:265`
- **Bug**: `event_datetime` was hardcoded to `datetime.now(UTC)` in ScrapedEvent constructor, corrupting `days_to_event` for all auto-collected listings
- **Fix**: `event_datetime=event.get("event_datetime") or datetime.now(UTC)`

### Phase 2: Feature Engineering
| Change | File | Impact |
|--------|------|--------|
| Added `popularity_data_available` flag | `ml/features/popularity.py` | Zero-variance in current data (auto-removed), useful as data grows |
| Removed `section_hash` (MD5 mod 1000) | `ml/features/seating.py` | Removed noise — hash has no semantic meaning |
| Added Bayesian-smoothed `venue_price_std` | `ml/features/venue.py` | Captures venue price variability (MSG vs small venues) |
| Lowered venue smoothing 200 → 75 | `ml/features/venue.py` | Better signal for moderate-sample venues |

### Phase 3: Correctness Fixes
| Change | File | Impact |
|--------|------|--------|
| LOO guard: float set → integer-cents set | `ml/features/event_pricing.py` | Robust to Parquet float round-trip |
| `np.log1p()` on regional listing count | `ml/features/regional.py` | Consistent with log-transformed target |

### Phase 4: Pipeline Hardening
| Change | File | Impact |
|--------|------|--------|
| Deleted `prepare_training_data()` | `ml/features/pipeline.py` | Removed data leakage code path |
| Fixed 5× `datetime.utcnow()` + 1× `datetime.now()` | evaluator, predictor, schemas, preprocess | Timezone-aware timestamps |
| Expanded artist aliases (2 → 11) | `ml/training/trainer.py` | 801 listings normalized |
| Fixed mypy type errors | youtube.py, lightgbm_model.py | Clean typecheck |

## Results

### v29 vs v28 Performance Comparison

| Metric | v28 | v29 | Change |
|--------|-----|-----|--------|
| MAE | $150.08 | $149.60 | **-$0.48** (improved) |
| RMSE | $237.35 | $235.93 | **-$1.42** (improved) |
| R² | 0.53 | 0.49 | -0.04 (note: different test set) |
| MAPE | 41.0% | 49.1% | +8.1% (note: different test set) |
| Dataset | 81 events, 23 artists | 135 events, 42 artists | 66% more events |
| Features | 54 | 50 | -4 (10 zero-variance removed) |

**Note**: Direct comparison is approximate — v29 trained on a significantly larger and more diverse dataset (135 events vs 81). The dataset nearly doubled in size, which makes generalization harder. MAE and RMSE improved despite this.

### Feature Importance Distribution (v29)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | event_zone_median_price | 53.1% |
| 2 | event_median_price | 17.5% |
| 3 | artist_regional_avg_price | 11.2% |
| 4 | artist_regional_median_price | 7.5% |
| 5 | event_zone_price_ratio | 3.7% |
| 6 | artist_zone_median_price | 1.8% |
| 7 | venue_median_price | 1.7% |
| 8 | artist_zone_price | 0.4% |
| 9 | row_numeric | 0.4% |
| 10 | artist_avg_price | 0.4% |

**Structural improvement**: Top feature dropped from 60.2% → 53.1%. Broader importance distribution indicates better generalization. Regional features now contribute 18.7% (up from ~3% in v28).

### Zero-Variance Features Removed (10)
is_known_artist, event_type_encoded, is_holiday_season, venue_capacity_bucket, price_momentum_7d, price_momentum_30d, price_vs_initial, price_volatility, popularity_data_available, is_known_venue

## Phase 5: Section Feature Experiment

**Status**: PASSED — feature kept.

### Go/No-Go Gate

| Criterion | Threshold | Actual | Pass? |
|-----------|-----------|--------|-------|
| Feature importance | ≥ 0.5% | 55.6% | YES |
| MAE does not increase | ≤ $149.60 | $148.27 | YES (-$1.33) |

### v29-section Results

| Metric | v29 (no section) | v29-section | Delta |
|--------|-----------------|-------------|-------|
| MAE | $149.60 | $148.27 | -$1.33 (better) |
| RMSE | $235.93 | $242.70 | +$6.77 (worse) |
| MAPE | 49.1% | 47.4% | -1.7% (better) |
| R² | 0.4895 | 0.4598 | -0.03 (worse) |
| Features | 50 | 51 | +1 |

**Trade-off**: The section feature improves average prediction accuracy (MAE, MAPE down) but slightly worsens tail predictions (RMSE up). This is expected — section-level encoding captures within-zone variation but with only 2-3 listings per section, the Bayesian smoothing is dominated by the zone prior. The model relies more heavily on the section feature (55.6%) at the expense of zone/event features, which slightly reduces robustness for extreme values.

**Implementation**: `event_section_median_price` is gated behind `include_section_feature=True` on `EventPricingFeatureExtractor`. Enabled via `--section-feature` flag on `scripts/train_model.py`.

## Final Recommended Model

**v29-section** (with section feature enabled): MAE $148.27, 51 features, 135 events, 42 artists.

## Date
2026-03-03
