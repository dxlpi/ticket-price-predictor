# merge_asof Edge Cases in Temporal Joins

## Rule
When using `pd.merge_asof(direction="backward")` followed by a separate group-aggregate merge, rows where merge_asof finds no backward match still receive valid aggregate data. Computed deltas between NaN-filled matched values and valid earliest values produce nonsensical features. Always zero out computed features where the merge key (`_snap_ts`) is NaN. Also guard against empty right DataFrames — they cause `MergeError: incompatible merge keys` due to dtype mismatches on empty typed columns.

## Why
Without the no-match guard, listings timestamped before all snapshots get features like `inventory_change_rate = -200` instead of 0.0. Without the empty-DataFrame guard, training crashes when no snapshot data exists.

## Pattern

**Right:**
```python
# Guard empty right DataFrame
if snapshot_df.empty:
    for col in snap_output_cols:
        result[col] = 0.0
    return result

# After merge_asof + aggregate merge + feature computation:
no_match = merged["_snap_ts"].isna()
for col in snap_output_cols:
    merged.loc[no_match, col] = 0.0  # zero out before general fillna
    merged[col] = merged[col].fillna(0.0)
```

**Wrong:**
```python
# Only doing general fillna — misses rows where merge_asof had no match
# but group-aggregate merge produced valid (but misleading) values
for col in snap_output_cols:
    merged[col] = merged[col].fillna(0.0)
```

---

## Rule 2 — Global sort on `on` column, not multi-column sort

`pd.merge_asof` requires the `on` column (e.g. `timestamp`) to be **globally sorted** across the entire DataFrame. Sorting by `["event_id", "timestamp"]` does NOT satisfy this — it sorts within each event group but not globally, causing `ValueError: left keys must be sorted`.

## Why
The `by=` parameter handles per-group matching at merge time. The `on=` column must be monotonically increasing across all rows.

## Pattern

```python
# Right: sort globally by the on-column only
df = df.sort_values("timestamp").reset_index(drop=True)
event_snapshots = event_snapshots.sort_values("timestamp").reset_index(drop=True)
merged = pd.merge_asof(df, event_snapshots, by="event_id", on="timestamp", direction="backward")

# Wrong: multi-column sort breaks the global monotonicity requirement
df = df.sort_values(["event_id", "timestamp"])  # NOT globally sorted on timestamp
```

---

## Rule 3 — Deduplicate lookup before left-merging back

When using `merge_asof` to look up values (e.g. `inv_at_future`) and then joining that result back to the original DataFrame by `(event_id, timestamp)`, multiple listings sharing the same scrape timestamp cause M:M row inflation unless the lookup is deduplicated first.

## Why
`merge_asof` produces one output row per left-side row. When you then left-merge this back onto a DataFrame where multiple rows share the same join key `(event_id, timestamp)`, each lookup row fans out to all matching left rows — multiplying the DataFrame.

## Pattern

```python
# Right: deduplicate lookup on join keys before merging back
future_lookup = (
    merged_future[["event_id", "timestamp", "inv_at_future"]]
    .drop_duplicates(subset=["event_id", "timestamp"])
)
result = merged_obs.merge(future_lookup, on=["event_id", "timestamp"], how="left")

# Wrong: merging without dedup causes row count inflation
result = merged_obs.merge(
    merged_future[["event_id", "timestamp", "inv_at_future"]],
    on=["event_id", "timestamp"], how="left"
)
```
