# 001: Data Leakage in Training Pipeline

**Status:** Resolved
**Severity:** Critical
**Area:** ML Training (`ml/training/`)

## Problem

The original training function `prepare_training_data()` fitted the feature pipeline on the entire dataset before splitting into train/val/test. This meant validation and test features were computed using statistics (means, medians, encodings) derived from data the model should never have seen — a textbook case of data leakage.

## Impact

- Model evaluation metrics were artificially inflated
- Real-world prediction accuracy was significantly worse than reported
- Target-encoded features (event/zone medians, artist stats) were the most affected since they directly encode price information

## Root Cause

The pipeline followed a common but incorrect pattern:
1. Load all data
2. Extract features (fitting encoders on everything)
3. Split into train/val/test

This allowed future price information to leak into training features via group-level statistics.

## Solution

Implemented a **split-before-fit** architecture:

1. Split raw DataFrames temporally with artist stratification (`TimeBasedSplitter.split_raw()`)
2. Fit the feature pipeline on training data only (`pipeline.fit(train_df)`)
3. Transform each split independently (`pipeline.transform(val_df)`, etc.)

Key design decisions:
- `prepare_training_data()` was deprecated and replaced with `ModelTrainer.train()`
- `TimeBasedSplitter` splits each artist's events independently by time for balanced representation
- All group-level statistics (ArtistStatsCache, RegionalStatsCache) use Bayesian smoothing to prevent small-sample memorization

## Outcome

- Training pipeline is now provably leak-free
- `leakage-guardian` agent enforces the split-before-fit invariant on every code change
- Model metrics reflect true generalization performance
