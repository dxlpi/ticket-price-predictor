# Log-Transform Allowlist for Price Features

## Rule
When log-transforming price-based features to align with a log-transformed target, use an explicit allowlist that excludes columns whose names contain "price"/"avg"/"median" but represent derived statistics (`_std`, `_cv`, `_ratio`, `_count`, `_change`, `_rate`). These suffixes indicate scale-independent values where log-transform is statistically inappropriate.

## Why
A broad heuristic that matches any column containing "price" will incorrectly transform `venue_price_std` (standard deviation, in $), `zone_price_ratio` (dimensionless), and similar derived features. Log-transforming a ratio or a std that already lies in [0,1] or small ranges produces distorted features that don't align with the log-space target and can hurt model performance.

## Pattern
```python
# trainer.py — module-level constant (single source of truth)
_LOG_EXCLUDE_SUFFIXES = ("_std", "_cv", "_ratio", "_count", "_change", "_rate")

# Apply to both trainer.py and objective.py (import from trainer)
price_cols = [
    c for c in X_train.columns
    if ("price" in c.lower() or "avg" in c.lower() or "median" in c.lower())
    and not any(c.lower().endswith(suffix) for suffix in _LOG_EXCLUDE_SUFFIXES)
]
for c in price_cols:
    X_train[c] = np.log1p(X_train[c].clip(lower=0))
    X_val[c]   = np.log1p(X_val[c].clip(lower=0))
    X_test[c]  = np.log1p(X_test[c].clip(lower=0))
```
Import `_LOG_EXCLUDE_SUFFIXES` in `objective.py` from `trainer` to keep the definition DRY — never duplicate the tuple.
