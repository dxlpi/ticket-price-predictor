# 004: Deduplication Hurts Model Performance

**Status:** Resolved (kept duplicates)
**Severity:** Medium
**Area:** ML Training, Data Quality

## Problem

During a model improvement investigation, deduplication of training data was tested as a potential improvement. The hypothesis was that duplicate listings (same event, zone, price, timestamp) were noise. In reality, removing them degraded performance by $6.79 MAE.

## Impact

- Naive deduplication removed real pricing signal
- MAE increased by ~$6.79 when duplicates were removed

## Root Cause

What appeared to be "duplicates" were actually distinct listings from different sellers at the same price point. Multiple sellers listing at the same price for the same zone is a strong demand signal — it indicates price consensus. Removing these data points destroyed that signal.

## Solution

Kept all listings without deduplication. Documented this as a key finding to prevent future regressions.

## Outcome

- Established that in secondary-market ticket data, apparent duplicates carry real information
- This finding informed the broader principle: be cautious about aggressive data cleaning in market data where repetition itself is a signal
