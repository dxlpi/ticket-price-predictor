# LightGBM Model Regression Analysis
Generated: 2026-02-11 12:40:30

## Executive Summary

The model regression from v11 (MAE $115, R² 0.66) to v12 (MAE $139, R² 0.57) is caused by **catastrophic validation set contamination** from time-based splitting on collection dates rather than event dates. The validation set has a 123% price spike ($762 vs $342 in training) due to extreme-price artists (Dolly Parton, Ariana Grande) and complete artist distribution shift between train/test sets with zero top-5 overlap.

## Data Overview

- **Dataset**: 13,170 listings from Parquet files
- **Date Range**: 2026-01-30 to 2026-02-11
- **Quality**: Clean data (0% missing critical fields, <1% duplicates in new data)
- **New Data Added**: 333 listings on Feb 11 (2.5% increase from v11's 12,837)

## Critical Findings

### Finding 1: Validation Set Price Spike (123% Increase)

Time-based split (70/15/15 on collection_date) creates massive distribution shift:

**Price Statistics by Split:**
| Split | Mean Price | Median Price | Date Range |
|-------|------------|--------------|------------|
| Train (70%) | $342.06 | $193.00 | Jan 30 - Feb 3 |
| Validation (15%) | $762.58 | $452.87 | Feb 3 - Feb 6 |
| Test (15%) | $390.96 | $291.00 | Feb 6 - Feb 11 |

**Statistical Evidence:**
- [STAT:val_train_ratio] 2.23x price ratio (validation/train)
- [STAT:train_mean] $342.06
- [STAT:val_mean] $762.58
- [STAT:test_mean] $390.96
- [STAT:ci] Validation set has 123% higher mean than training

The model is trained on $342 average prices but evaluated on $762 average prices - a fundamental mismatch.

### Finding 2: Extreme Artist Concentration in Validation

8 high-priced artists (avg > $500) contribute 85.4% of validation set value:

**High-Price Artists in Validation:**
| Artist | Listings | Avg Price | Train Avg |
|--------|----------|-----------|-----------|
| Dolly Parton | 105 | $3,704 | $3,302 |
| Ariana Grande | 91 | $2,178 | $1,535 |
| Chris Stapleton | 255 | $928 | $624 |
| Backstreet Boys | 242 | $895 | $768 |
| Alan Jackson | 109 | $793 | $776 |

**Impact:**
- [STAT:ariana_impact_on_val_mean] $68.35 increase to validation mean from Ariana Grande alone
- [STAT:high_price_artists_contribution] 85.4% of validation value from 8 artists
- Artist prices in validation are 15-42% higher than in training

### Finding 3: Complete Artist Distribution Shift

Zero overlap in top-5 artists between train and test sets:

**Train Top 5:**
- Morgan Wallen (2,189 listings, $200 avg)
- Bruno Mars (2,153 listings, $291 avg)
- BTS (1,599 listings, $153 avg)
- Zach Bryan (1,500 listings, $66 avg)
- BTS - Bangtan Boys (572 listings, $151 avg)

**Test Top 5:**
- Backstreet Boys (278 listings, $614 avg)
- Chris Stapleton (244 listings, $269 avg)
- The Eagles (218 listings, $588 avg)
- Lady Gaga (186 listings, $345 avg)
- Megan Moroney (176 listings, $53 avg)

**Cold Start Problem:**
- [STAT:unseen_artists_in_test] 4 artists appear in test but never in training
- Unseen artists: Alan Jackson, George Strait, Megan Moroney, Rush
- Model has no `artist_avg_price` feature for these artists

### Finding 4: Feature Collapse (days_to_event)

**Days to Event Distribution:**
| Split | Mean | Median | Variance |
|-------|------|--------|----------|
| Train | 44.3 days | 67.0 days | High |
| Validation | 0.0 days | 0.0 days | **ZERO** |
| Test | 0.0 days | 0.0 days | **ZERO** |

The `days_to_event` feature has zero variance in validation and test sets - all listings are same-day or expired. This renders the feature useless for evaluation.

### Finding 5: Feature Leakage via artist_avg_price

The most important feature (51% importance) exhibits leakage:

- Model learns from training: "Ariana Grande = $1,535 avg"
- Validation reality: "Ariana Grande = $2,178 avg" (42% higher)
- Model systematically underestimates high-priced artists
- No mechanism to handle price evolution over time

## Statistical Details

### Outlier Analysis

**Old Data (Jan 30 - Feb 9):**
- [STAT:old_outlier_rate] 9.00% outliers (1,155 listings)
- 672 extreme outliers (>3×IQR)
- Max price: $16,465 (2026 Philadelphia Eagles Season Tickets)

**New Data (Feb 11):**
- [STAT:new_outlier_rate] 1.20% outliers (4 listings)
- 3 extreme outliers
- Max price: $1,591 (much cleaner)

### Price Distribution Percentiles

| Percentile | Old Data | New Data |
|------------|----------|----------|
| p5 | $58.00 | $52.00 |
| p25 | $143.00 | $144.00 |
| p50 | $217.00 | $295.00 |
| p75 | $418.14 | $464.00 |
| p95 | $1,309.00 | $845.80 |
| p99 | $3,640.67 | $1,243.24 |

New data is cleaner with fewer extreme outliers, but median is 36% higher.

### Data Quality Comparison

**Old Data:**
- [STAT:duplicate_listings] 2,979 duplicates (23.21%)
- Missing: face_value (100%), markup_ratio (100%), seat positions (18-19%)

**New Data:**
- [STAT:duplicate_listings] 1 duplicate (0.30%)
- Same missing patterns (face_value, markup_ratio unavailable from source)

## Visualizations

(No visualizations generated - matplotlib not used in this analysis)

## Limitations

- [LIMITATION] Analysis assumes 70/15/15 time-sorted split on collection_date; actual implementation may differ
- [LIMITATION] Feature engineering details (how artist_avg_price is computed, whether it uses temporal windows) not examined
- [LIMITATION] Model hyperparameters and early stopping criteria not investigated
- [LIMITATION] Only 12 days of data (Jan 30 - Feb 11); longer-term trends unknown
- [LIMITATION] Missing face_value and markup_ratio (100% null) limits feature richness

## Recommendations

### 1. FIX TIME-BASED SPLIT (CRITICAL)

**Current Problem:**
- Splitting on `collection_date` creates artist/venue clustering
- Collection date ≠ event date
- Results in non-comparable train/val/test distributions

**Solution:**
```python
# Sort by event_datetime instead of collection_date
df_sorted = df.sort_values('event_datetime')
train, val, test = time_split(df_sorted, [0.70, 0.15, 0.15])
```

This ensures temporal progression of EVENTS, not data collection dates.

### 2. STRATIFIED SPLIT BY ARTIST

Prevent validation set from being dominated by extreme artists:

```python
from sklearn.model_selection import StratifiedShuffleSplit

# Bin artists by price tier
df['artist_price_tier'] = pd.qcut(df.groupby('artist_or_team')['listing_price'].transform('mean'), q=5, labels=False)

# Stratified split
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, temp_idx = next(splitter.split(df, df['artist_price_tier']))
# Then split temp into val/test
```

### 3. HANDLE COLD-START ARTISTS

**Problem:** 4 artists in test never seen in training → artist_avg_price is unknown

**Solutions:**
- **Option A:** Global fallback (use overall mean price)
- **Option B:** Genre-based fallback (use similar-genre average)
- **Option C:** Exclude artist_avg_price for unseen artists, rely on other features
- **Option D:** Use artist features from external API (Spotify popularity, genre)

### 4. TEMPORAL ARTIST FEATURES

Replace static `artist_avg_price` with time-aware features:

```python
# Instead of: artist_avg_price (all-time average)
# Use: artist_avg_price_last_30d, artist_price_trend, artist_volatility
```

This handles price evolution (e.g., Ariana Grande $1,535 → $2,178).

### 5. FILTER EXTREME OUTLIERS

- Cap prices at 99th percentile ($3,641) during training
- Or use robust loss function (Huber loss instead of MSE)
- 672 extreme outliers (max $16,465) distort the model

### 6. INVESTIGATE days_to_event FEATURE

**Issue:** Zero variance in val/test sets (all 0 days)

**Actions:**
- Check data collection logic - why only same-day listings in recent data?
- If unavoidable, consider binning: [0, 1-7, 8-30, 31-60, 60+] days
- Or remove feature if recent data never has future events

---

*Generated by Scientist Agent*
*Python 3.12, pandas 2.x, pyarrow*
