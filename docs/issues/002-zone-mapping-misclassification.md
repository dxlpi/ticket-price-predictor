# 002: Zone Mapping Misclassification (Sections 400-499)

**Status:** Resolved
**Severity:** High
**Area:** Normalization (`normalization/seat_zones.py`)

## Problem

Sections numbered 400-499 were being incorrectly classified in the seat zone mapping. This caused tickets in those sections to receive wrong zone labels, which propagated through all zone-level features (zone median price, artist-zone encoding, zone price ratios).

## Impact

- Zone-level features were noisy for affected sections
- Model predictions for events with many 400-level sections were unreliable
- Since `event_zone_median_price` is the strongest feature (60% importance), misclassified zones had outsized impact on accuracy

## Root Cause

The zone mapping logic did not correctly handle the 400-series section numbering convention used by certain venue layouts, leading to these sections being assigned to the wrong zone category.

## Solution

Fixed the zone classification rules in the seat zone normalization module to correctly map sections 400-499 to their appropriate zones based on venue section numbering conventions.

## Outcome

- Contributed to the v21 improvement (MAE dropped from $216.88 to $141.95, a 34.6% improvement)
- Zone-level features became more reliable across all venues
