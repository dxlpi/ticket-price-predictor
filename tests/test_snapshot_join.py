"""Integration tests for _join_snapshot_features (temporal snapshot join).

These tests verify the merge_asof temporal matching, delta feature computation,
and row-order preservation of the private module-level function in trainer.py.
"""

import numpy as np
import pandas as pd
import pytest

from ticket_price_predictor.ml.training.trainer import _join_snapshot_features
from ticket_price_predictor.normalization.seat_zones import SeatZoneMapper


def _ts(s: str) -> pd.Timestamp:
    """Parse a UTC timestamp string."""
    return pd.Timestamp(s, tz="UTC")


@pytest.fixture
def zone_mapper() -> SeatZoneMapper:
    return SeatZoneMapper()


def _make_listings(
    timestamps: list[str],
    sections: list[str] | None = None,
    event_ids: list[str] | None = None,
) -> pd.DataFrame:
    n = len(timestamps)
    return pd.DataFrame(
        {
            "event_id": event_ids or ["e1"] * n,
            "section": sections or ["Section 100"] * n,
            "timestamp": [_ts(t) for t in timestamps],
            "listing_price": [100.0] * n,
        }
    )


def _make_snapshots(
    event_id: str,
    seat_zone: str,
    timestamps: list[str],
    prices: list[float],
    inventories: list[int],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": [event_id] * len(timestamps),
            "seat_zone": [seat_zone] * len(timestamps),
            "timestamp": [_ts(t) for t in timestamps],
            "price_avg": prices,
            "price_min": [p * 0.8 for p in prices],
            "price_max": [p * 1.2 for p in prices],
            "inventory_remaining": inventories,
        }
    )


class TestJoinSnapshotFeatures:
    def test_backward_merge_matches_latest_prior_snapshot(
        self, zone_mapper: SeatZoneMapper
    ) -> None:
        """Listing gets the latest snapshot with timestamp <= listing.timestamp."""
        # Two snapshots: T1=08:00 (price=100), T2=09:00 (price=110)
        # Listing at T3=11:00 → should match T2 snapshot
        listings = _make_listings(["2024-06-01 11:00:00+00:00"])
        snapshots = _make_snapshots(
            "e1",
            "lower_tier",
            ["2024-06-01 08:00:00+00:00", "2024-06-01 09:00:00+00:00"],
            [100.0, 110.0],
            [200, 180],
        )
        result = _join_snapshot_features(listings, snapshots, zone_mapper)
        # price_trend = (110 - 100) / 100 = 0.10 (matched T2, earliest T1)
        assert result["_snap_zone_price_trend"].iloc[0] == pytest.approx(0.10)

    def test_no_snapshot_match_returns_zero_defaults(self, zone_mapper: SeatZoneMapper) -> None:
        """Listing with no prior snapshot gets 0.0 for all _snap_* columns."""
        # Listing is at T=06:00, snapshot is at T=08:00 (future) → no match
        listings = _make_listings(["2024-06-01 06:00:00+00:00"])
        snapshots = _make_snapshots(
            "e1",
            "lower_tier",
            ["2024-06-01 08:00:00+00:00"],
            [100.0],
            [200],
        )
        result = _join_snapshot_features(listings, snapshots, zone_mapper)
        assert result["_snap_inventory_change_rate"].iloc[0] == pytest.approx(0.0)
        assert result["_snap_zone_price_trend"].iloc[0] == pytest.approx(0.0)
        assert result["_snap_count"].iloc[0] == pytest.approx(0.0)
        assert result["_snap_price_range"].iloc[0] == pytest.approx(0.0)

    def test_inventory_change_rate_computation(self, zone_mapper: SeatZoneMapper) -> None:
        """inventory_change_rate = (matched_inv - earliest_inv) / delta_hours."""
        # Earliest: T=08:00, inv=200. Matched: T=10:00 (2h later), inv=160.
        # rate = (160 - 200) / 2h = -20 tickets/hour
        listings = _make_listings(["2024-06-01 11:00:00+00:00"])
        snapshots = _make_snapshots(
            "e1",
            "lower_tier",
            ["2024-06-01 08:00:00+00:00", "2024-06-01 10:00:00+00:00"],
            [100.0, 100.0],
            [200, 160],
        )
        result = _join_snapshot_features(listings, snapshots, zone_mapper)
        assert result["_snap_inventory_change_rate"].iloc[0] == pytest.approx(-20.0)

    def test_price_trend_computation(self, zone_mapper: SeatZoneMapper) -> None:
        """price_trend = (matched_price - earliest_price) / earliest_price."""
        # Earliest: 100.0. Matched: 125.0. Trend = (125 - 100) / 100 = 0.25
        listings = _make_listings(["2024-06-01 12:00:00+00:00"])
        snapshots = _make_snapshots(
            "e1",
            "lower_tier",
            ["2024-06-01 08:00:00+00:00", "2024-06-01 10:00:00+00:00"],
            [100.0, 125.0],
            [200, 200],
        )
        result = _join_snapshot_features(listings, snapshots, zone_mapper)
        assert result["_snap_zone_price_trend"].iloc[0] == pytest.approx(0.25)

    def test_original_row_order_preserved(self, zone_mapper: SeatZoneMapper) -> None:
        """Output row order matches the original listings index, not merge sort order."""
        # Listings in reverse chronological order
        listings = _make_listings(
            [
                "2024-06-01 12:00:00+00:00",  # row 0
                "2024-06-01 08:00:00+00:00",  # row 1
                "2024-06-01 10:00:00+00:00",  # row 2
            ]
        )
        snapshots = _make_snapshots(
            "e1",
            "lower_tier",
            ["2024-06-01 07:00:00+00:00"],
            [100.0],
            [200],
        )
        result = _join_snapshot_features(listings, snapshots, zone_mapper)
        assert len(result) == 3
        assert list(result.index) == list(listings.index)
        # All three rows are after T=07:00 so they all get a snapshot match
        assert (result["_snap_count"] > 0).all()

    def test_empty_snapshot_df_returns_all_zeros(self, zone_mapper: SeatZoneMapper) -> None:
        """When snapshot_df is empty, all _snap_* columns default to 0.0."""
        listings = _make_listings(["2024-06-01 10:00:00+00:00"])
        empty_snaps = pd.DataFrame(
            columns=[
                "event_id",
                "seat_zone",
                "timestamp",
                "price_avg",
                "price_min",
                "price_max",
                "inventory_remaining",
            ]
        )
        result = _join_snapshot_features(listings, empty_snaps, zone_mapper)
        assert result["_snap_inventory_change_rate"].iloc[0] == pytest.approx(0.0)
        assert result["_snap_zone_price_trend"].iloc[0] == pytest.approx(0.0)
        assert result["_snap_count"].iloc[0] == pytest.approx(0.0)
        assert result["_snap_price_range"].iloc[0] == pytest.approx(0.0)

    def test_snapshot_count_is_log1p_of_total(self, zone_mapper: SeatZoneMapper) -> None:
        """_snap_count = log1p(total snapshot count for this event-zone)."""
        listings = _make_listings(["2024-06-01 12:00:00+00:00"])
        snapshots = _make_snapshots(
            "e1",
            "lower_tier",
            [
                "2024-06-01 08:00:00+00:00",
                "2024-06-01 09:00:00+00:00",
                "2024-06-01 10:00:00+00:00",
            ],
            [100.0, 105.0, 110.0],
            [200, 190, 180],
        )
        result = _join_snapshot_features(listings, snapshots, zone_mapper)
        assert result["_snap_count"].iloc[0] == pytest.approx(np.log1p(3))

    def test_snap_price_range_computation(self, zone_mapper: SeatZoneMapper) -> None:
        """_snap_price_range = (max - min) / avg at the matched snapshot."""
        listings = _make_listings(["2024-06-01 10:00:00+00:00"])
        snapshots = pd.DataFrame(
            {
                "event_id": ["e1"],
                "seat_zone": ["lower_tier"],
                "timestamp": [_ts("2024-06-01 09:00:00+00:00")],
                "price_avg": [100.0],
                "price_min": [80.0],
                "price_max": [120.0],
                "inventory_remaining": [200],
            }
        )
        result = _join_snapshot_features(listings, snapshots, zone_mapper)
        # price_range = (120 - 80) / 100 = 0.40
        assert result["_snap_price_range"].iloc[0] == pytest.approx(0.40)

    def test_row_count_preserved(self, zone_mapper: SeatZoneMapper) -> None:
        """Output has exactly the same number of rows as input listings."""
        listings = _make_listings(["2024-06-01 10:00:00+00:00", "2024-06-01 11:00:00+00:00"])
        snapshots = _make_snapshots(
            "e1",
            "lower_tier",
            ["2024-06-01 09:00:00+00:00"],
            [100.0],
            [200],
        )
        result = _join_snapshot_features(listings, snapshots, zone_mapper)
        assert len(result) == 2

    def test_different_zones_matched_independently(self, zone_mapper: SeatZoneMapper) -> None:
        """Listings in different zones are matched to their respective zone snapshots."""
        listings = pd.DataFrame(
            {
                "event_id": ["e1", "e1"],
                "section": ["Section 100", "Upper Deck"],  # lower_tier and upper_tier
                "timestamp": [
                    _ts("2024-06-01 10:00:00+00:00"),
                    _ts("2024-06-01 10:00:00+00:00"),
                ],
                "listing_price": [200.0, 80.0],
            }
        )
        # Lower tier snapshot: price 200, inv 50
        # Upper tier snapshot: price 80, inv 150
        snapshots = pd.DataFrame(
            {
                "event_id": ["e1", "e1"],
                "seat_zone": ["lower_tier", "upper_tier"],
                "timestamp": [
                    _ts("2024-06-01 09:00:00+00:00"),
                    _ts("2024-06-01 09:00:00+00:00"),
                ],
                "price_avg": [200.0, 80.0],
                "price_min": [160.0, 64.0],
                "price_max": [240.0, 96.0],
                "inventory_remaining": [50, 150],
            }
        )
        result = _join_snapshot_features(listings, snapshots, zone_mapper)
        assert len(result) == 2
        # Both zones got a match (count > 0)
        assert (result["_snap_count"] > 0).all()
