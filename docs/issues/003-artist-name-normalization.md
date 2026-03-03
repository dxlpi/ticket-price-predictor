# 003: Artist Name Normalization

**Status:** Resolved
**Severity:** Medium
**Area:** ML Training (`ml/training/`), Features (`ml/features/performer.py`)

## Problem

The same artist appeared under multiple name variants in the data (e.g., "BTS", "BTS - Bangtan Boys", "Bangtan Boys"). The training pipeline treated these as separate artists, fragmenting their historical data and weakening artist-level features.

## Impact

- Artist-level statistics (median price, event count, popularity) were split across aliases
- Bayesian smoothing over-regularized toward global means due to artificially small per-artist sample sizes
- Artist stratification during train/val/test splitting was inconsistent — the same artist could appear in multiple splits under different names

## Root Cause

External data sources (Ticketmaster, VividSeats, StubHub) use inconsistent artist naming. No normalization was applied before computing artist-level aggregates or splitting data.

## Solution

Added artist name normalization as an early preprocessing step, applied before both splitting and feature extraction:

- Built an alias mapping for known variants
- Normalization runs before `TimeBasedSplitter.split_raw()` so artist stratification sees unified names
- ArtistStatsCache receives normalized names, producing stronger per-artist statistics

## Outcome

- Per-artist sample sizes increased, improving Bayesian-smoothed statistics
- Artist stratification during splitting became consistent
- Contributed to improved model accuracy in v21+
