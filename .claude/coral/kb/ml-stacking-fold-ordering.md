# Stacking Temporal CV: Positional K-Fold Needs Globally Sorted Rows
Promoted: 2026-04-16

## Rule
When expanding-window temporal CV builds fold indices via `np.arange(n_samples)` positional slicing, the input rows MUST be globally sorted by time. Upstream splitters that concatenate stratified sub-groups (e.g. `split_raw()` grouping by artist, sorting within each, then `pd.concat`) produce rows ordered `(group, time_within_group)` — NOT globally temporal. Running positional k-fold on that ordering silently trains on one group's future to predict another group's past.

## Why
Stratified splits preserve per-group representation but destroy global time order. Any CV scheme that indexes by position assumes time order. Mismatched ordering is a leakage pattern that is invisible in per-fold metrics (folds look valid, each sample appears in exactly one val fold) but cross-group temporal invariants are silently violated, producing mildly optimistic base predictions and biasing the meta-learner.

## Pattern
Wrong (positional k-fold on artist-grouped rows):
```python
def fit(self, X_train, y_train, ...):
    n = len(X_train)
    fold_size = n // (self._n_folds + 1)
    for i in range(self._n_folds):
        train_idx = np.arange(fold_size * (i + 1))
        val_idx = np.arange(fold_size * (i + 1), fold_size * (i + 2))
        # train_idx rows may include future timestamps from artist A
        # while val_idx rows are past timestamps from artist B
```

Right (sort by explicit timestamps before positional indexing):
```python
def fit(self, X_train, y_train, *, timestamps=None, ...):
    if timestamps is not None:
        ts = pd.Series(timestamps).reset_index(drop=True)
        if len(ts) != len(X_train):
            raise ValueError(...)
        sort_idx = ts.argsort(kind="stable").to_numpy()
        X_train = X_train.iloc[sort_idx].reset_index(drop=True)
        y_train = pd.Series(y_train).iloc[sort_idx].reset_index(drop=True)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)[sort_idx]
    # now np.arange() positional slicing aligns with true temporal order
```

Caller must pass row-aligned timestamps: `model.fit(X_train, y_train, timestamps=train_df["timestamp"])`.

Use `kind="stable"` so ties (identical timestamps) retain input order and folds remain deterministic.
