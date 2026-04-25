# Relative Pricing Features — Target Leak Pattern
Promoted: 2026-03-23 | Updated: 2026-03-23

## Rule
Never use `listing_price` as an input to compute a feature when `listing_price` is the prediction target. `(listing_price - group_median) / group_std` is a direct target leak — the model inverts it to recover the price trivially. Similarly, `price_min`/`price_max` for small zone groups memorize individual listing prices.

## Why
This caused R² to jump from 0.60 to 0.97 (impossibly good on unseen data), which revealed the leak. The model learned to invert the deviation formula rather than learning genuine pricing patterns.

## Pattern
Safe (no leak — uses only group-level stats, not the row's own price):
```python
# Use hierarchy-level statistics that don't depend on the target row
features["event_zone_median_price"] = zone_group["listing_price"].median()  # group stat
features["zone_price_ratio"] = zone_median / event_median  # ratio of group stats
```

Leaky (avoid):
```python
# Uses listing_price (target) to compute the feature for that same row
features["price_deviation"] = (listing_price - group_median) / group_std
features["price_min"] = small_group["listing_price"].min()  # memorizes individual prices
```

The leakage-guardian agent checks for this pattern. Any feature derivation touching `listing_price` for the same row being predicted is suspect.
