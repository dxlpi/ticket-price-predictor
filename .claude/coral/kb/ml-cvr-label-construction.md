# CVR Label Construction Patterns

Promoted: 2026-03-30 | Updated: 2026-03-30

## Rule

Use event-level inventory depletion from `PriceSnapshot` data (not individual listing disappearance) for CVR label construction. Disappearance-based labels are unreliable because the scraper captures partial inventory (median ~2 listings/scrape). Per-event timestamps must be used — global timestamps inflate sold ratio to ~99% by letting future listings across all events "prove" a listing was never sold.

## Why

Using global timestamps in `SaleLabelBuilder` caused sold ratio of 98.93% and AUC 0.48 (worse than random). The fix: group by `event_id` and compute timestamps, cutoffs, and the `existing` set per-event independently. For the inventory depletion strategy, aggregate snapshots to event-level (sum `inventory_remaining` across zones) because raw listings lack `seat_zone` at label time.

## Pattern

**Per-event timestamp isolation (disappearance strategy):**
```python
# Right: process each event independently
for _event_id, event_group in df.groupby("event_id"):
    event_timestamps = sorted(event_group["timestamp"].unique())
    event_max_ts = event_group["timestamp"].max()
    event_cutoff = event_max_ts - timedelta(hours=window_hours)
    event_existing = set(zip(event_group["listing_id"], event_group["timestamp"]))
    # ... disappearance logic using event_timestamps and event_existing

# Wrong: global timestamps contaminate cross-event label computation
all_timestamps = sorted(df["timestamp"].unique())  # mixes all events → ~99% sold
```

**Inventory depletion labels (`_label_` prefix convention):**
```python
# Label columns prefixed with _label_ must be dropped before feature extraction
result["_label_depletion_rate"] = depletion_rate
result["sold"] = (result["_label_depletion_rate"] > threshold).astype(int)

# In trainer — drop immediately after label building, before split and feature pipeline
if "_label_depletion_rate" in labeled_df.columns:
    labeled_df = labeled_df.drop(columns=["_label_depletion_rate"])
```

**Event-level aggregation (required because listings lack `seat_zone` at label time):**
```python
# Drop null inventory before aggregation (Ticketmaster snapshots may have None)
snapshots = snapshots.dropna(subset=["inventory_remaining"])
event_snapshots = (
    snapshots.groupby(["event_id", "timestamp"])["inventory_remaining"]
    .sum()
    .reset_index(name="total_inventory")
)
```

**Depletion threshold 0.3 → ~20% sold ratio** on this dataset (within healthy 10–70% range for binary classification).
