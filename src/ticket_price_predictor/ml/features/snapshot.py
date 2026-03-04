"""Snapshot-based temporal feature extraction."""

import pandas as pd

from ticket_price_predictor.ml.features.base import FeatureExtractor

# Pre-joined column names written by _join_snapshot_features in trainer.py
_SNAP_INV_CHANGE = "_snap_inventory_change_rate"
_SNAP_PRICE_TREND = "_snap_zone_price_trend"
_SNAP_COUNT = "_snap_count"
_SNAP_PRICE_RANGE = "_snap_price_range"


class SnapshotFeatureExtractor(FeatureExtractor):
    """Extract temporal features from pre-joined price snapshot data.

    This extractor is stateless w.r.t. snapshot lookups — it only reads
    ``_snap_*`` columns pre-joined by ``ModelTrainer._join_snapshot_features``
    (called after split_raw, per split independently). This prevents any
    leakage from the extractor directly accessing raw snapshot data.

    Global defaults are learned from training data in ``fit()`` and used as
    fallbacks for rows with no snapshot match.
    """

    def __init__(self) -> None:
        """Initialize with zero defaults (overwritten in fit)."""
        self._default_inv_change: float = 0.0
        self._default_price_trend: float = 0.0
        self._default_count: float = 0.0
        self._default_price_range: float = 0.0

    @property
    def feature_names(self) -> list[str]:
        """Return list of feature names."""
        return [
            "snapshot_inventory_change_rate",
            "snapshot_zone_price_trend",
            "snapshot_count",
            "snapshot_price_range",
        ]

    def fit(self, df: pd.DataFrame) -> "SnapshotFeatureExtractor":
        """Compute global defaults from training data.

        Args:
            df: Training DataFrame (must have _snap_* columns if snapshots
                were joined; otherwise defaults remain 0.0)

        Returns:
            self
        """

        def _mean(col: str) -> float:
            if col not in df.columns:
                return 0.0
            vals = df[col].dropna()
            return float(vals.mean()) if len(vals) > 0 else 0.0

        self._default_inv_change = _mean(_SNAP_INV_CHANGE)
        self._default_price_trend = _mean(_SNAP_PRICE_TREND)
        self._default_count = _mean(_SNAP_COUNT)
        self._default_price_range = _mean(_SNAP_PRICE_RANGE)
        return self

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract snapshot features from pre-joined DataFrame.

        Reads ``_snap_*`` columns written by the trainer's snapshot join.
        Falls back to training-mean defaults when columns are absent or NaN.

        Args:
            df: DataFrame (raw listings with pre-joined _snap_* columns)

        Returns:
            DataFrame with 4 snapshot feature columns
        """
        result = pd.DataFrame(index=df.index)

        mappings = [
            ("snapshot_inventory_change_rate", _SNAP_INV_CHANGE, self._default_inv_change),
            ("snapshot_zone_price_trend", _SNAP_PRICE_TREND, self._default_price_trend),
            ("snapshot_count", _SNAP_COUNT, self._default_count),
            ("snapshot_price_range", _SNAP_PRICE_RANGE, self._default_price_range),
        ]
        for out_col, in_col, default in mappings:
            if in_col in df.columns:
                result[out_col] = df[in_col].fillna(default)
            else:
                result[out_col] = default

        return result
