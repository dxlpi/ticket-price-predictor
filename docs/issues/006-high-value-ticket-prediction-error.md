# 006: High-Value Ticket Prediction Error (Q4 Quartile)

**Status:** Open (known limitation)
**Severity:** High
**Area:** ML Model Performance

## Problem

Tickets in the top price quartile (Q4, $440+) have dramatically higher prediction errors — MAE of $337, roughly 10x the error of the cheapest quartile. The model struggles to accurately predict premium ticket prices.

## Impact

- Predictions for VIP, front-row, and premium seating are unreliable
- Overall MAE is heavily influenced by Q4 errors
- Users relying on predictions for high-value tickets may make poor purchasing decisions

## Root Cause

Multiple contributing factors:
1. **Price distribution skew**: Premium tickets have high variance — floor seats at the same event can range from $500 to $5,000+ depending on exact position and seller
2. **Sparse data**: High-value listings are less common, so the model has fewer examples to learn from
3. **Log-transform compression**: The `np.log1p` target transform compresses high-value differences, reducing the model's ability to distinguish between $800 and $1,500
4. **95th percentile cap**: Outlier capping removes the most extreme prices, but the remaining Q4 range is still very wide

## Attempted Solutions

- **Segment-aware outlier capping**: Tried capping outliers per price segment instead of globally. Result: hurt performance by $6.07 MAE (tail error inflation)
- **p99 cap** (instead of p95): Allowed more extreme values through. Result: hurt performance (extreme outliers the model can't predict inflate MAE)
- **Section-level target encoding**: Too sparse (only 2-3 listings per section), added noise

## Current Status

This remains an open limitation. The most promising path forward is collecting more data (currently 81 events, 23 artists) to give the model more high-value examples to learn from. Documented in the model card as a known limitation.

## Outcome

- Documented as a known limitation in `docs/model-card.md`
- Established that Q4 prediction is the primary accuracy bottleneck
- Dataset size (not model architecture) is the primary constraint
