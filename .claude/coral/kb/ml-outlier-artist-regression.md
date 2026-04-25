# ML Regression from Outlier Artist Concentration
Promoted: 2026-03-23 | Updated: 2026-03-23

## Rule
When dataset artist diversity shrinks (fewer unique artists/events) while outlier-price artists dominate, model MAE regresses even with similar total listing counts. A single artist at >20% share corrupts artist-level feature statistics (artist_regional_avg_price, artist_venue_price) that are the 2nd–4th most important features.

## Why
v33 regressed +$11.52 MAE vs v32: Lady Gaga was 22.6% of listings (mean $767, max $89k), Ariana Grande 4.1% (mean $1,598, $36k outlier). Festival/multi-day passes (BottleRock, Stagecoach) attributed to real artist names inflated their `artist_avg_price`. `snapshot_inventory_change_rate` dropped to zero-variance from lack of temporal diversity.

## Pattern
Monitor before retraining:
```python
# Artist concentration check — flag if any artist >15% of listings
artist_counts = df.groupby("artist_or_team").size()
concentration = artist_counts / len(df)
if concentration.max() > 0.15:
    print(f"WARNING: {concentration.idxmax()} is {concentration.max():.1%} of dataset")
```

Watch for: festival passes attributed to headliners, residency-heavy artists, and artists with extreme outlier tickets (>$10k). Consider per-artist outlier capping before training.
