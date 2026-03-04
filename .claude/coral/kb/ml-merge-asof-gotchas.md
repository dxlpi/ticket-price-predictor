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
