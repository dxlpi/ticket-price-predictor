# Snapshot Data Is Collected But Not Used in Training

## Rule
The SnapshotRepository (`storage/repository.py:157-284`) actively collects zone-level price snapshots via the EC2 hourly scraper, but the training pipeline (`data_loader.py`, `trainer.py`) only loads individual listings. MomentumFeatureExtractor (`timeseries.py:62-146`) is instantiated in the feature pipeline but returns hardcoded defaults (0.0, 1.0) because no snapshot data is ever passed to it. This means momentum features are always zero-variance and auto-removed.

## Why
Without this knowledge, you'll waste time debugging why momentum features have no signal, or assume temporal data doesn't exist. The infrastructure is complete — the gap is purely in the DataLoader → FeaturePipeline integration.

## Pattern
**Wrong**: Assuming momentum features need new data collection to work.
**Right**: The data exists in `data/raw/snapshots/`. DataLoader needs to load snapshots via SnapshotRepository, join them to listings on (event_id, seat_zone, timestamp proximity), and pre-compute momentum columns before passing to the feature pipeline. The join requires mapping raw section names to normalized zones via SeatZoneMapper.

Current data: ~1,591 snapshots, 91 events, 8 timestamps over ~20 hours (as of 2026-03-03). Sparse but sufficient for pipeline validation.
