# Validation Rules

## BLOCKING (must pass before merge)

- [ ] No data leakage: feature pipeline fitted on training data only
- [ ] Split-before-fit: `split_raw()` called before `pipeline.fit()`
- [ ] Pydantic model changes have matching `parquet_schema()` updates
- [ ] No upward imports (lower layers must not import from `ml/`)
- [ ] All changed source files have corresponding tests
- [ ] `make check` passes (lint + typecheck + test)

## STRONG (should be addressed)

- [ ] New feature extractors implement `FeatureExtractor` ABC (`fit()`, `extract()`, `feature_names`)
- [ ] Bayesian smoothing used for group-level statistics (no raw means of small groups)
- [ ] Scraper changes preserve anti-detection patterns (random delays, stealth mode)
- [ ] Price filtering maintained: <$10 invalid, 95th percentile cap
- [ ] Artist name normalization applied before splitting
- [ ] City name normalization applied before splitting

## MINOR (nice to have)

- [ ] Conventional commit message format
- [ ] New code follows existing naming conventions (snake_case functions, PascalCase classes)
- [ ] No `datetime.utcnow()` — use `datetime.now(UTC)` instead
- [ ] Type annotations present on all new functions
