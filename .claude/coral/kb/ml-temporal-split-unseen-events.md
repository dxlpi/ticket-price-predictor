# Temporal Split: Seen vs Unseen Event MAE Gap
Promoted: 2026-04-03

## Rule
With artist-stratified temporal splitting, the test set contains a mix of seen events (earlier listings in training, later in test) and unseen events (entirely new events). For this project, seen events get $52.75 MAE while unseen events get $128.91 MAE — a 2.4x gap. The validation set has 82% event overlap with training vs 63% for test, explaining the persistent $20 val→test MAE gap.

## Why
Event-level target encoding (section median, zone median) accounts for ~66% of feature importance. For seen events, these features encode actual event pricing. For unseen events, they're Bayesian-smoothed toward training priors, providing much weaker signal. No amount of hyperparameter tuning or model complexity can overcome this fundamental data limitation — all configs converge to the same test MAE ($86-87).

## Pattern
Before setting aggressive MAE targets, analyze the seen/unseen event composition:
```python
train_events = set(raw.train_df['event_id'].unique())
test_seen = test_df['event_id'].isin(train_events)
# Compute MAE separately for seen vs unseen
# Set targets based on the weighted average, not just validation performance
```
