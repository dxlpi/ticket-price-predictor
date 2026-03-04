"""Tests for SnapshotFeatureExtractor."""

import math

import numpy as np
import pandas as pd
import pytest

from ticket_price_predictor.ml.features.snapshot import SnapshotFeatureExtractor


def _make_enriched_df(n: int = 3) -> pd.DataFrame:
    """DataFrame with pre-joined _snap_* columns (as written by trainer)."""
    base: dict[str, list[float]] = {
        "listing_price": [100.0, 200.0, 150.0],
        "_snap_inventory_change_rate": [-5.0, -2.0, 0.0],
        "_snap_zone_price_trend": [0.05, -0.02, 0.0],
        "_snap_count": [np.log1p(3), np.log1p(5), np.log1p(1)],
        "_snap_price_range": [0.4, 0.2, 0.0],
    }
    repeats = math.ceil(n / 3)
    return pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(n)],
            **{k: (v * repeats)[:n] for k, v in base.items()},
        }
    )


def _make_unenriched_df(n: int = 3) -> pd.DataFrame:
    """DataFrame WITHOUT _snap_* columns (no snapshot data joined)."""
    return pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(n)],
            "listing_price": [100.0, 200.0, 150.0],
        }
    )


class TestSnapshotFeatureExtractor:
    def test_feature_names(self):
        ext = SnapshotFeatureExtractor()
        assert ext.feature_names == [
            "snapshot_inventory_change_rate",
            "snapshot_zone_price_trend",
            "snapshot_count",
            "snapshot_price_range",
        ]

    def test_feature_count(self):
        ext = SnapshotFeatureExtractor()
        assert len(ext.feature_names) == 4

    def test_extract_with_snap_columns(self):
        ext = SnapshotFeatureExtractor()
        df = _make_enriched_df()
        ext.fit(df)
        result = ext.extract(df)

        assert list(result.columns) == ext.feature_names
        assert len(result) == 3
        assert result["snapshot_inventory_change_rate"].iloc[0] == pytest.approx(-5.0)
        assert result["snapshot_zone_price_trend"].iloc[0] == pytest.approx(0.05)
        assert result["snapshot_count"].iloc[0] == pytest.approx(np.log1p(3))
        assert result["snapshot_price_range"].iloc[0] == pytest.approx(0.4)

    def test_extract_without_snap_columns_uses_defaults(self):
        """When _snap_* columns are absent, returns global defaults (0.0 unfitted)."""
        ext = SnapshotFeatureExtractor()
        df = _make_unenriched_df()
        result = ext.extract(df)

        assert list(result.columns) == ext.feature_names
        assert len(result) == 3
        # Unfitted extractor returns 0.0 defaults
        assert (result["snapshot_inventory_change_rate"] == 0.0).all()
        assert (result["snapshot_zone_price_trend"] == 0.0).all()
        assert (result["snapshot_count"] == 0.0).all()
        assert (result["snapshot_price_range"] == 0.0).all()

    def test_fit_computes_global_defaults(self):
        """fit() stores training means as defaults for missing rows."""
        ext = SnapshotFeatureExtractor()
        train_df = _make_enriched_df()
        ext.fit(train_df)

        # Defaults should be training means
        expected_inv = float(train_df["_snap_inventory_change_rate"].mean())
        assert ext._default_inv_change == pytest.approx(expected_inv)

        expected_trend = float(train_df["_snap_zone_price_trend"].mean())
        assert ext._default_price_trend == pytest.approx(expected_trend)

    def test_extract_fills_nan_with_defaults(self):
        """NaN _snap_* values are filled with training defaults."""
        ext = SnapshotFeatureExtractor()
        train_df = _make_enriched_df()
        ext.fit(train_df)

        # One row has no snapshot match (NaN)
        test_df = pd.DataFrame(
            {
                "_snap_inventory_change_rate": [-3.0, float("nan")],
                "_snap_zone_price_trend": [0.1, float("nan")],
                "_snap_count": [np.log1p(2), float("nan")],
                "_snap_price_range": [0.3, float("nan")],
            }
        )
        result = ext.extract(test_df)

        # Non-NaN row preserved
        assert result["snapshot_inventory_change_rate"].iloc[0] == pytest.approx(-3.0)
        # NaN row uses training default
        assert result["snapshot_inventory_change_rate"].iloc[1] == pytest.approx(
            ext._default_inv_change
        )

    def test_fit_on_empty_data_keeps_zero_defaults(self):
        """fit() on DataFrame without _snap_* columns keeps 0.0 defaults."""
        ext = SnapshotFeatureExtractor()
        ext.fit(_make_unenriched_df())

        assert ext._default_inv_change == pytest.approx(0.0)
        assert ext._default_price_trend == pytest.approx(0.0)
        assert ext._default_count == pytest.approx(0.0)
        assert ext._default_price_range == pytest.approx(0.0)

    def test_extract_returns_correct_row_count(self):
        ext = SnapshotFeatureExtractor()
        df = _make_enriched_df(n=10)
        result = ext.extract(df)
        assert len(result) == 10
