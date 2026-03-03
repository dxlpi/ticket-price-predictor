# 007: Segment-Aware Outlier Capping Regression

**Status:** Resolved (reverted)
**Severity:** Medium
**Area:** ML Training (`ml/training/data_loader.py`)

## Problem

As an attempt to improve high-value ticket predictions (Issue #006), segment-aware outlier capping was implemented — capping outliers at the 95th percentile within each price segment rather than globally. This degraded performance by $6.07 MAE.

## Impact

- MAE increased by ~$6.07 compared to global 95th percentile capping
- Tail error inflation: per-segment caps were too aggressive for smaller segments

## Root Cause

Per-segment capping with small segment sizes produced unstable cap values. A segment with only 20-30 samples had a noisy 95th percentile estimate, leading to either too-aggressive or too-lenient capping. The global cap, while less targeted, was more stable due to the larger sample size.

## Solution

Reverted to global 95th percentile capping. The simpler approach was more robust given the current dataset size.

## Outcome

- Reinforced the principle: with small datasets, simpler statistical approaches outperform segment-level ones
- Connected to the broader dataset size bottleneck (Issue #006)
