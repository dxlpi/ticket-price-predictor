# 009: City Name Normalization Inconsistencies

**Status:** Resolved
**Severity:** Medium
**Area:** ML Features (`ml/features/geo_mapping.py`, `ml/features/regional.py`)

## Problem

City names from different data sources were inconsistent (e.g., "Las Vegas" vs "Las Vegas, NV" vs "North Las Vegas"). This fragmented regional statistics, weakening the RegionalStatsCache and city-level features.

## Impact

- RegionalStatsCache treated name variants as separate cities
- City-tier classification was inconsistent
- Regional price statistics were diluted across aliases
- Bayesian smoothing was overly aggressive due to artificially small per-city sample sizes

## Root Cause

Multiple data sources (Ticketmaster API, VividSeats scraper, StubHub scraper) format city names differently. No normalization was applied before computing regional aggregates.

## Solution

Implemented `geo_mapping._normalize_city()` as an early normalization step:
- Standardizes city name formats across sources
- Maps city names to country and region for geographic features
- Applied before splitting so regional statistics are consistent across train/val/test

## Outcome

- Regional features became more reliable
- RegionalStatsCache produces stronger per-city statistics with larger effective sample sizes
- City-tier classification is now consistent across data sources
