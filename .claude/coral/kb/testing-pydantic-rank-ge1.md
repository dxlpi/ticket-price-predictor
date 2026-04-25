# Pydantic ge=1 Rank Field — Collect Raw Dicts Before Building
Promoted: 2026-03-27 | Updated: 2026-03-27
## Rule
When building Pydantic models with a `rank: int = Field(..., ge=1)` field, never assign rank during collection (rank is unknown until the full list is sorted). Instead, collect raw dicts, sort them, then construct Pydantic models with the final rank in one pass.
## Why
Assigning a placeholder rank of 0 during collection violates `ge=1` and raises `ValidationError`. Assigning rank=1 to all items before sorting is also wrong. The rank is only knowable after sorting.
## Pattern
```python
# Wrong — rank=0 violates ge=1
pending = []
for item in items:
    pending.append(RankedListing(**data, rank=0))  # ValidationError
pending.sort(key=lambda x: x.value_score, reverse=True)

# Right — collect raw dicts, sort, then build with correct rank
pending: list[tuple[float, dict]] = []
for item in items:
    pending.append((value_score, data_dict))
pending.sort(key=lambda x: x[0], reverse=True)
result = [RankedListing(**data, rank=rank) for rank, (_, data) in enumerate(pending, start=1)]
```
