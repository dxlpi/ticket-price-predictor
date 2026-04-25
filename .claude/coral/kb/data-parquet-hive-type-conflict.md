# Parquet Hive Partition Type Conflict
Promoted: 2026-03-27 | Updated: 2026-03-27
## Rule
When reading Hive-partitioned Parquet directories, use `pyarrow.dataset.dataset(path, format="parquet", partitioning="hive")` instead of `pq.read_table(path)`. The latter fails with `ArrowTypeError: Unable to merge: Field year has incompatible types: int32 vs dictionary<values=int32, ...>` when partition column types are inconsistent across fragments (common after large dataset growth).
## Why
`pq.read_table()` tries to merge schemas eagerly across all fragments and fails on heterogeneous partition column types. The dataset API defers schema merging and handles mixed types gracefully. Hit during v34 training when dataset grew to 99K listings across many date partitions.
## Pattern
```python
# Wrong — fails with ArrowTypeError on heterogeneous partition types
table = pq.read_table(path)

# Right — handles mixed partition column types across fragments
import pyarrow.dataset as ds
table = ds.dataset(path, format="parquet", partitioning="hive").to_table()
```
