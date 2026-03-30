"""Tests for SaleLabelBuilder and InventoryDepletionLabeler."""

import warnings
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from ticket_price_predictor.ml.training.label_builder import (
    InventoryDepletionLabeler,
    SaleLabelBuilder,
)


def _make_listings(
    timestamps_by_id: dict[str, list[datetime]],
    event_dt: datetime | None = None,
) -> pd.DataFrame:
    """Build a listings DataFrame from {listing_id: [timestamps]} mapping."""
    if event_dt is None:
        event_dt = datetime(2025, 12, 31, 20, 0, tzinfo=UTC)
    rows = []
    for lid, timestamps in timestamps_by_id.items():
        for ts in timestamps:
            rows.append(
                {
                    "listing_id": lid,
                    "event_id": "evt_001",
                    "timestamp": ts,
                    "event_datetime": event_dt,
                    "listing_price": 100.0,
                    "section": "100",
                    "row": "5",
                    "seat_zone": "lower_tier",
                }
            )
    return pd.DataFrame(rows)


BASE_TIME = datetime(2025, 6, 1, 10, 0, tzinfo=UTC)


class TestVerifyListingIdStability:
    def setup_method(self):
        self.builder = SaleLabelBuilder()

    def test_stable_listing_id_detection(self):
        """listing_id appearing in 2+ timestamps is stable."""
        df = _make_listings(
            {
                "L001": [BASE_TIME, BASE_TIME + timedelta(hours=1), BASE_TIME + timedelta(hours=2)],
                "L002": [BASE_TIME],  # unstable — single scrape
            }
        )
        stats = self.builder.verify_listing_id_stability(df)
        assert stats["stable_ids"] == 1
        assert stats["unstable_ids"] == 1
        assert stats["stability_ratio"] == pytest.approx(0.5)

    def test_all_stable(self):
        """All listings with 2+ timestamps are all stable."""
        df = _make_listings(
            {
                "L001": [BASE_TIME, BASE_TIME + timedelta(hours=1)],
                "L002": [BASE_TIME, BASE_TIME + timedelta(hours=1)],
            }
        )
        stats = self.builder.verify_listing_id_stability(df)
        assert stats["stable_ids"] == 2
        assert stats["unstable_ids"] == 0
        assert stats["stability_ratio"] == pytest.approx(1.0)

    def test_all_unstable_warns(self):
        """All single-scrape listings triggers a low-stability warning."""
        df = _make_listings(
            {
                "L001": [BASE_TIME],
                "L002": [BASE_TIME],
            }
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stats = self.builder.verify_listing_id_stability(df)

        assert stats["stability_ratio"] == pytest.approx(0.0)
        assert any("stability" in str(w.message).lower() for w in caught)

    def test_missing_columns_raises(self):
        """Missing required columns raises ValueError."""
        df = pd.DataFrame({"listing_id": ["L001"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            self.builder.verify_listing_id_stability(df)

    def test_empty_dataframe(self):
        """Empty DataFrame returns all-zero stats."""
        df = pd.DataFrame(columns=["listing_id", "event_id", "timestamp"])
        stats = self.builder.verify_listing_id_stability(df)
        assert stats["total_listings"] == 0
        assert stats["stable_ids"] == 0
        assert stats["unstable_ids"] == 0
        assert stats["stability_ratio"] == pytest.approx(0.0)

    def test_total_is_stable_plus_unstable(self):
        """total_listings == stable_ids + unstable_ids."""
        df = _make_listings(
            {
                "L001": [BASE_TIME, BASE_TIME + timedelta(hours=1)],
                "L002": [BASE_TIME],
                "L003": [BASE_TIME, BASE_TIME + timedelta(hours=2)],
            }
        )
        stats = self.builder.verify_listing_id_stability(df)
        assert stats["total_listings"] == stats["stable_ids"] + stats["unstable_ids"]

    def test_stability_ratio_type_is_float(self):
        """stability_ratio is a float."""
        df = _make_listings({"L001": [BASE_TIME, BASE_TIME + timedelta(hours=1)]})
        stats = self.builder.verify_listing_id_stability(df)
        assert isinstance(stats["stability_ratio"], float)


class TestBuildLabels:
    def setup_method(self):
        self.builder = SaleLabelBuilder()

    def test_sold_label_requires_consecutive_absences(self):
        """Listing absent from 2+ consecutive scrapes within window is labeled sold=1."""
        # L001 appears at t0, t1 then disappears — sold
        # L002 appears throughout — not sold
        timestamps = [BASE_TIME + timedelta(hours=i) for i in range(5)]
        df = _make_listings(
            {
                "L001": timestamps[:2],  # present at t0, t1; absent t2, t3, t4
                "L002": timestamps,  # present throughout
            }
        )
        result = self.builder.build_labels(df, window_hours=4, min_absent_scrapes=2)

        sold_map = dict(zip(result["listing_id"], result["sold"], strict=True))
        assert sold_map["L001"] == 1
        assert sold_map["L002"] == 0

    def test_transient_absence_not_sold(self):
        """Listing that disappears for one scrape then reappears is NOT sold."""
        # t0, t1, t3 present (absent only at t2)
        timestamps = [BASE_TIME + timedelta(hours=i) for i in range(5)]
        df = _make_listings(
            {
                "L001": [timestamps[0], timestamps[1], timestamps[3], timestamps[4]],
            }
        )
        result = self.builder.build_labels(df, window_hours=6, min_absent_scrapes=2)

        # L001 was seen at t3 and t4 after absence at t2 — not sold
        if "L001" in result["listing_id"].values:
            row = result[result["listing_id"] == "L001"].iloc[0]
            assert row["sold"] == 0

    def test_sold_column_present(self):
        """Output always has a 'sold' column."""
        timestamps = [BASE_TIME + timedelta(hours=i) for i in range(3)]
        df = _make_listings({"L001": timestamps})
        result = self.builder.build_labels(df)
        assert "sold" in result.columns

    def test_hours_until_disappearance_column_present(self):
        """Output always has 'hours_until_disappearance' column."""
        timestamps = [BASE_TIME + timedelta(hours=i) for i in range(3)]
        df = _make_listings({"L001": timestamps})
        result = self.builder.build_labels(df)
        assert "hours_until_disappearance" in result.columns

    def test_hours_until_disappearance_positive_for_sold(self):
        """hours_until_disappearance is positive for sold listings."""
        timestamps = [BASE_TIME + timedelta(hours=i) for i in range(5)]
        df = _make_listings({"L001": timestamps[:2], "L002": timestamps})
        result = self.builder.build_labels(df, window_hours=4, min_absent_scrapes=2)

        sold_rows = result[result["sold"] == 1]
        assert len(sold_rows) > 0
        assert (sold_rows["hours_until_disappearance"] > 0).all()

    def test_hours_until_disappearance_null_for_not_sold(self):
        """hours_until_disappearance is NaN for listings labeled not sold."""
        timestamps = [BASE_TIME + timedelta(hours=i) for i in range(5)]
        df = _make_listings({"L001": timestamps})  # always present, not sold
        result = self.builder.build_labels(df, window_hours=2, min_absent_scrapes=2)

        not_sold = result[result["sold"] == 0]
        if len(not_sold) > 0:
            assert not_sold["hours_until_disappearance"].isna().all()

    def test_post_event_listings_excluded(self):
        """Listings with timestamp >= event_datetime are excluded."""
        event_dt = BASE_TIME + timedelta(hours=1)
        # L001 has one scrape before event, one after
        df = _make_listings(
            {"L001": [BASE_TIME, BASE_TIME + timedelta(hours=2)]},
            event_dt=event_dt,
        )
        result = self.builder.build_labels(df)
        # All remaining rows must be before event_dt
        if len(result) > 0:
            assert (result["timestamp"] < result["event_datetime"]).all()

    def test_empty_dataframe_returns_correct_columns(self):
        """Empty input returns empty output with correct columns."""
        df = pd.DataFrame(columns=["listing_id", "event_id", "timestamp", "event_datetime"])
        result = self.builder.build_labels(df)
        assert "sold" in result.columns
        assert "hours_until_disappearance" in result.columns
        assert len(result) == 0

    def test_missing_columns_raises(self):
        """Missing required columns raises ValueError."""
        df = pd.DataFrame({"listing_id": ["L001"], "event_id": ["e"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            self.builder.build_labels(df)

    def test_single_scrape_listings_not_sold(self):
        """Listings with only one scrape cannot be labeled sold (insufficient future data)."""
        df = _make_listings({"L001": [BASE_TIME]})
        result = self.builder.build_labels(df, window_hours=1, min_absent_scrapes=2)
        # Either excluded by cutoff or labeled 0
        if len(result) > 0:
            assert result["sold"].iloc[0] == 0

    def test_min_absent_scrapes_one_requires_only_single_absence(self):
        """min_absent_scrapes=1 labels sold after a single missing scrape."""
        timestamps = [BASE_TIME + timedelta(hours=i) for i in range(4)]
        df = _make_listings(
            {
                "L001": timestamps[:1],  # present only at t0; absent at t1, t2, t3
                "L002": timestamps,
            }
        )
        result = self.builder.build_labels(df, window_hours=3, min_absent_scrapes=1, min_scrapes=4)

        sold_map = dict(zip(result["listing_id"], result["sold"], strict=True))
        # L001 absent from t1 onward — should be sold
        assert sold_map.get("L001", 0) == 1

    def test_sold_labels_are_integer(self):
        """sold column dtype is int (not bool or float)."""
        timestamps = [BASE_TIME + timedelta(hours=i) for i in range(4)]
        df = _make_listings({"L001": timestamps[:2], "L002": timestamps})
        result = self.builder.build_labels(df, window_hours=3, min_absent_scrapes=2)
        if len(result) > 0:
            assert result["sold"].dtype in (int, "int64", "int32")

    def test_window_hours_respected(self):
        """Absences outside window_hours are not counted."""
        # L001 present at t0-t4, absent starting at t5
        # With window=4h from t0: sees t1,t2,t3,t4 — L001 present in all → not sold at t0
        # If window were 6h: would see t5 (absent) too → might be sold
        timestamps = [BASE_TIME + timedelta(hours=i) for i in range(10)]
        df = _make_listings(
            {
                "L001": timestamps[:5],  # present t0-t4, absent t5+
                "L002": timestamps,
            }
        )
        result = self.builder.build_labels(df, window_hours=4, min_absent_scrapes=2)
        # At t0, L001 is present throughout the 4h window (t1,t2,t3,t4) → not sold
        rows = result[result["listing_id"] == "L001"]
        if len(rows) > 0:
            # t0 observation: L001 present at t1,t2,t3,t4 → zero absences → not sold
            assert rows.iloc[0]["sold"] == 0


# ---------------------------------------------------------------------------
# AC8: Per-event timestamp fix and density guard tests
# ---------------------------------------------------------------------------


def _make_multi_event_listings(
    event_timestamps: dict[str, list[datetime]],
    listing_id_prefix: str = "L",
    event_dt: datetime | None = None,
) -> pd.DataFrame:
    """Build listings DataFrame with multiple events.

    Args:
        event_timestamps: {event_id: [timestamps]} — one listing per timestamp per event
    """
    if event_dt is None:
        event_dt = datetime(2025, 12, 31, 20, 0, tzinfo=UTC)
    rows = []
    for event_id, timestamps in event_timestamps.items():
        for i, ts in enumerate(timestamps):
            rows.append(
                {
                    "listing_id": f"{listing_id_prefix}_{event_id}_{i:02d}",
                    "event_id": event_id,
                    "timestamp": ts,
                    "event_datetime": event_dt,
                    "listing_price": 100.0,
                    "section": "100",
                    "row": "5",
                    "seat_zone": "lower_tier",
                }
            )
    return pd.DataFrame(rows)


class TestSaleLabelBuilderPerEventTimestamps:
    """Verify per-event timestamp isolation (AC1 fix)."""

    def setup_method(self) -> None:
        self.builder = SaleLabelBuilder()

    def test_per_event_isolation_no_cross_contamination(self) -> None:
        """Events with disjoint timestamp sets must not cross-contaminate labels.

        Before the fix: Event A's listing would be checked against Event B's
        timestamps (global set), causing near-universal sold=1. After the fix:
        each event only checks its own timestamps.
        """
        # Event A: dense scraping — L_A_00 present at all 6 timestamps → not sold
        ts_a = [BASE_TIME + timedelta(hours=i) for i in range(6)]
        # Event B: same 6 timestamps, offset by 100 hours — no overlap with Event A
        ts_b = [BASE_TIME + timedelta(hours=100 + i) for i in range(6)]
        event_dt = datetime(2025, 12, 31, 20, 0, tzinfo=UTC)

        rows = []
        # Event A: one listing present at all 6 timestamps
        for ts in ts_a:
            rows.append(
                {
                    "listing_id": "L_A",
                    "event_id": "evt_A",
                    "timestamp": ts,
                    "event_datetime": event_dt,
                    "listing_price": 100.0,
                    "section": "100",
                    "row": "5",
                    "seat_zone": "lower_tier",
                }
            )
        # Event B: one listing present at all 6 timestamps
        for ts in ts_b:
            rows.append(
                {
                    "listing_id": "L_B",
                    "event_id": "evt_B",
                    "timestamp": ts,
                    "event_datetime": event_dt,
                    "listing_price": 100.0,
                    "section": "100",
                    "row": "5",
                    "seat_zone": "lower_tier",
                }
            )
        df = pd.DataFrame(rows)

        result = self.builder.build_labels(df, window_hours=5, min_absent_scrapes=2, min_scrapes=3)

        # L_A is present throughout all of Event A's timestamps → not sold
        a_rows = result[result["listing_id"] == "L_A"]
        assert len(a_rows) > 0, "Event A listing should be in result"
        assert (a_rows["sold"] == 0).all(), "L_A present throughout — not sold"

        # L_B is present throughout all of Event B's timestamps → not sold
        b_rows = result[result["listing_id"] == "L_B"]
        assert len(b_rows) > 0, "Event B listing should be in result"
        assert (b_rows["sold"] == 0).all(), "L_B present throughout — not sold"

    def test_per_event_cutoff_not_global(self) -> None:
        """Per-event cutoff is based on each event's max timestamp.

        Before the fix: global cutoff could exclude all listings from events
        with earlier max timestamps. After fix: each event uses its own cutoff.
        """
        # Event A: 6 timestamps starting at BASE_TIME
        ts_a = [BASE_TIME + timedelta(hours=i) for i in range(6)]
        # Event B: 6 timestamps starting 48h later (higher max timestamp)
        ts_b = [BASE_TIME + timedelta(hours=48 + i) for i in range(6)]
        event_dt = datetime(2025, 12, 31, 20, 0, tzinfo=UTC)

        rows = []
        for ts in ts_a:
            rows.append(
                {
                    "listing_id": "L_A",
                    "event_id": "evt_A",
                    "timestamp": ts,
                    "event_datetime": event_dt,
                    "listing_price": 100.0,
                    "section": "100",
                    "row": "5",
                    "seat_zone": "lower_tier",
                }
            )
        for ts in ts_b:
            rows.append(
                {
                    "listing_id": "L_B",
                    "event_id": "evt_B",
                    "timestamp": ts,
                    "event_datetime": event_dt,
                    "listing_price": 100.0,
                    "section": "100",
                    "row": "5",
                    "seat_zone": "lower_tier",
                }
            )
        df = pd.DataFrame(rows)

        # window=5h: global cutoff would be ts_b[-1] - 5h = ts_a[-1] + 43h
        # That would exclude ALL of Event A's timestamps.
        # Per-event cutoff: Event A uses its own max → ts_a[-1] - 5h = ts_a[0]
        result = self.builder.build_labels(df, window_hours=5, min_absent_scrapes=2, min_scrapes=3)

        # Event A should have at least some results (per-event cutoff preserves it)
        a_rows = result[result["event_id"] == "evt_A"]
        assert len(a_rows) > 0, "Event A should have labels — per-event cutoff, not global"

    def test_sold_label_still_works_single_event(self) -> None:
        """Per-event fix is backward-compatible with single-event data."""
        timestamps = [BASE_TIME + timedelta(hours=i) for i in range(6)]
        df = _make_listings({"L001": timestamps[:2], "L002": timestamps})
        result = self.builder.build_labels(df, window_hours=4, min_absent_scrapes=2, min_scrapes=3)

        sold_map = {row["listing_id"]: row["sold"] for _, row in result.iterrows()}
        assert sold_map.get("L001") == 1, "L001 absent at t2,t3,t4 — sold"
        assert sold_map.get("L002") == 0, "L002 present throughout — not sold"


class TestSaleLabelBuilderDensityGuard:
    """Verify min_scrapes density guard (AC2)."""

    def setup_method(self) -> None:
        self.builder = SaleLabelBuilder()

    def test_sparse_event_excluded(self) -> None:
        """Events with fewer than min_scrapes timestamps are excluded from output."""
        # Only 3 timestamps — below min_scrapes=5
        ts = [BASE_TIME + timedelta(hours=i) for i in range(3)]
        df = _make_listings({"L001": ts[:2], "L002": ts})
        result = self.builder.build_labels(df, min_scrapes=5)
        assert len(result) == 0, "Sparse event should produce no labels"

    def test_dense_event_included(self) -> None:
        """Events with >= min_scrapes timestamps are included."""
        ts = [BASE_TIME + timedelta(hours=i) for i in range(6)]
        df = _make_listings({"L001": ts[:2], "L002": ts})
        # Use window_hours=2 so the per-event cutoff (max_ts - 2h) doesn't exclude
        # all listings (timestamps span only 5h, well inside a 24h default window)
        result = self.builder.build_labels(df, window_hours=2, min_scrapes=5)
        assert len(result) > 0, "Dense event should produce labels"

    def test_min_scrapes_default_is_five(self) -> None:
        """Default min_scrapes=5 excludes events with 4 or fewer scrapes."""
        ts_4 = [BASE_TIME + timedelta(hours=i) for i in range(4)]
        df = _make_listings({"L001": ts_4})
        result = self.builder.build_labels(df)  # default min_scrapes=5
        assert len(result) == 0, "4 scrapes < default min_scrapes=5 — excluded"

    def test_density_guard_per_event(self) -> None:
        """Density guard applies per-event: dense events included, sparse excluded."""
        event_dt = datetime(2025, 12, 31, 20, 0, tzinfo=UTC)

        # Dense event: 6 timestamps
        ts_dense = [BASE_TIME + timedelta(hours=i) for i in range(6)]
        # Sparse event: 2 timestamps
        ts_sparse = [BASE_TIME + timedelta(hours=i) for i in range(2)]

        rows = []
        for ts in ts_dense:
            rows.append(
                {
                    "listing_id": "dense_L",
                    "event_id": "evt_dense",
                    "timestamp": ts,
                    "event_datetime": event_dt,
                    "listing_price": 100.0,
                    "section": "100",
                    "row": "5",
                    "seat_zone": "lower_tier",
                }
            )
        for ts in ts_sparse:
            rows.append(
                {
                    "listing_id": "sparse_L",
                    "event_id": "evt_sparse",
                    "timestamp": ts,
                    "event_datetime": event_dt,
                    "listing_price": 100.0,
                    "section": "100",
                    "row": "5",
                    "seat_zone": "lower_tier",
                }
            )
        df = pd.DataFrame(rows)
        # Use window_hours=2 so per-event cutoff (max_ts - 2h) doesn't exclude all
        # dense-event listings (timestamps span only 5h)
        result = self.builder.build_labels(df, window_hours=2, min_scrapes=5)

        # Dense event should have results
        assert "evt_dense" in result["event_id"].values, "Dense event should be included"
        # Sparse event should be excluded
        assert "evt_sparse" not in result["event_id"].values, "Sparse event should be excluded"


# ---------------------------------------------------------------------------
# Helpers for InventoryDepletionLabeler tests
# ---------------------------------------------------------------------------

EVENT_DT = datetime(2025, 12, 31, 20, 0, tzinfo=UTC)


def _make_depletion_listings(
    event_id: str,
    timestamps: list[datetime],
    event_dt: datetime | None = None,
) -> pd.DataFrame:
    """Build listings DataFrame for depletion labeler tests."""
    if event_dt is None:
        event_dt = EVENT_DT
    rows = [
        {
            "listing_id": f"{event_id}_L{i:02d}",
            "event_id": event_id,
            "timestamp": ts,
            "event_datetime": event_dt,
            "listing_price": 100.0,
        }
        for i, ts in enumerate(timestamps)
    ]
    return pd.DataFrame(rows)


def _make_snapshots(
    event_id: str,
    timestamps_inventory: list[tuple[datetime, int | None]],
    seat_zone: str = "lower_tier",
) -> pd.DataFrame:
    """Build snapshots DataFrame for depletion labeler tests."""
    rows = [
        {
            "event_id": event_id,
            "seat_zone": seat_zone,
            "timestamp": ts,
            "inventory_remaining": inv,
        }
        for ts, inv in timestamps_inventory
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# AC7: InventoryDepletionLabeler tests
# ---------------------------------------------------------------------------


class TestInventoryDepletionLabeler:
    """Full test suite for InventoryDepletionLabeler."""

    def setup_method(self) -> None:
        self.labeler = InventoryDepletionLabeler(
            window_hours=10, depletion_threshold=0.3, min_snapshots=3
        )

    def _listing_ts(self, offset_hours: int = 0) -> datetime:
        return BASE_TIME + timedelta(hours=offset_hours)

    def _snap_ts(self, offset_hours: int = 0) -> datetime:
        return BASE_TIME + timedelta(hours=offset_hours)

    def test_basic_depletion_sold(self) -> None:
        """Zone depleting >30% within window → sold=1."""
        # Inventory drops from 100 to 60 (40% depletion > 0.3 threshold)
        snaps = _make_snapshots(
            "E1",
            [
                (self._snap_ts(0), 100),
                (self._snap_ts(5), 80),
                (self._snap_ts(10), 60),
                (self._snap_ts(15), 50),
            ],
        )
        listings = _make_depletion_listings(
            "E1",
            [self._listing_ts(0), self._listing_ts(1)],
        )
        result = self.labeler.build_labels(listings, snaps)
        assert len(result) > 0
        assert (result["sold"] == 1).all(), "40% depletion should label all listings sold=1"

    def test_stable_inventory_not_sold(self) -> None:
        """Stable inventory → sold=0."""
        # Inventory stays at 100 (0% depletion < 0.3 threshold)
        snaps = _make_snapshots(
            "E1",
            [
                (self._snap_ts(0), 100),
                (self._snap_ts(5), 100),
                (self._snap_ts(10), 102),  # slight increase
            ],
        )
        listings = _make_depletion_listings("E1", [self._listing_ts(1)])
        result = self.labeler.build_labels(listings, snaps)
        assert len(result) > 0
        assert (result["sold"] == 0).all(), "Stable inventory → not sold"

    def test_increasing_inventory_not_sold(self) -> None:
        """Inventory increase (relisting) yields negative depletion → sold=0."""
        # Inventory goes UP (relisting scenario)
        snaps = _make_snapshots(
            "E1",
            [
                (self._snap_ts(0), 50),
                (self._snap_ts(5), 70),
                (self._snap_ts(10), 90),
            ],
        )
        listings = _make_depletion_listings("E1", [self._listing_ts(1)])
        result = self.labeler.build_labels(listings, snaps)
        assert len(result) > 0
        assert (result["sold"] == 0).all(), "Inventory increase → depletion negative → not sold"
        assert (result["_label_depletion_rate"] < 0).all()

    def test_density_filter_excludes_sparse_events(self) -> None:
        """Events with fewer than min_snapshots snapshot timestamps are excluded."""
        # Only 2 snapshot timestamps — below min_snapshots=3
        snaps = _make_snapshots(
            "E1",
            [
                (self._snap_ts(0), 100),
                (self._snap_ts(5), 60),
            ],
        )
        listings = _make_depletion_listings("E1", [self._listing_ts(1)])
        result = self.labeler.build_labels(listings, snaps)
        assert len(result) == 0, "Sparse event should be excluded by min_snapshots filter"

    def test_empty_listings_returns_empty(self) -> None:
        """Empty listings DataFrame → empty result."""
        snaps = _make_snapshots(
            "E1", [(self._snap_ts(0), 100), (self._snap_ts(5), 80), (self._snap_ts(10), 60)]
        )
        listings = pd.DataFrame(columns=["event_id", "timestamp", "event_datetime"])
        result = self.labeler.build_labels(listings, snaps)
        assert len(result) == 0
        assert "sold" in result.columns
        assert "_label_depletion_rate" in result.columns

    def test_empty_snapshots_returns_empty(self) -> None:
        """Empty snapshots DataFrame → empty result."""
        snaps = pd.DataFrame(columns=["event_id", "timestamp", "inventory_remaining"])
        listings = _make_depletion_listings("E1", [self._listing_ts(1)])
        result = self.labeler.build_labels(listings, snaps)
        assert len(result) == 0

    def test_missing_listing_columns_raises(self) -> None:
        """Missing required listing columns raises ValueError."""
        snaps = _make_snapshots(
            "E1", [(self._snap_ts(0), 100), (self._snap_ts(5), 80), (self._snap_ts(10), 60)]
        )
        bad_listings = pd.DataFrame({"event_id": ["E1"]})
        with pytest.raises(ValueError, match="Missing required listing columns"):
            self.labeler.build_labels(bad_listings, snaps)

    def test_missing_snapshot_columns_raises(self) -> None:
        """Missing required snapshot columns raises ValueError."""
        bad_snaps = pd.DataFrame({"event_id": ["E1"]})
        listings = _make_depletion_listings("E1", [self._listing_ts(1)])
        with pytest.raises(ValueError, match="Missing required snapshot columns"):
            self.labeler.build_labels(listings, bad_snaps)

    def test_threshold_tuning(self) -> None:
        """Higher threshold requires greater depletion for sold=1."""
        # 40% depletion: above 0.3, below 0.5
        snaps = _make_snapshots(
            "E1",
            [
                (self._snap_ts(0), 100),
                (self._snap_ts(5), 80),
                (self._snap_ts(10), 60),
            ],
        )
        listings = _make_depletion_listings("E1", [self._listing_ts(1)])

        low_labeler = InventoryDepletionLabeler(
            window_hours=10, depletion_threshold=0.3, min_snapshots=3
        )
        high_labeler = InventoryDepletionLabeler(
            window_hours=10, depletion_threshold=0.5, min_snapshots=3
        )

        result_low = low_labeler.build_labels(listings, snaps)
        result_high = high_labeler.build_labels(listings, snaps)

        if len(result_low) > 0:
            assert (result_low["sold"] == 1).all(), "40% > 0.3 threshold → sold=1"
        if len(result_high) > 0:
            assert (result_high["sold"] == 0).all(), "40% < 0.5 threshold → sold=0"

    def test_temporal_alignment_uses_prior_snapshot(self) -> None:
        """merge_asof uses the PRIOR snapshot (backward direction)."""
        # Inventory at t0=100, t5=50, t10=20
        # Listing at t3: prior snapshot is t0 (inv=100), future (t3+10=t13) → t10 (inv=20)
        # depletion = (100-20)/100 = 0.80 > 0.3 → sold=1
        snaps = _make_snapshots(
            "E1",
            [
                (self._snap_ts(0), 100),
                (self._snap_ts(5), 50),
                (self._snap_ts(10), 20),
            ],
        )
        listings = _make_depletion_listings("E1", [self._listing_ts(3)])
        result = self.labeler.build_labels(listings, snaps)
        assert len(result) > 0
        # depletion_rate should be (100-20)/100 = 0.80
        assert result["_label_depletion_rate"].iloc[0] == pytest.approx(0.80)
        assert result["sold"].iloc[0] == 1

    def test_no_matching_snapshots_excluded(self) -> None:
        """Listings without a matching snapshot (different event) are dropped."""
        # Snapshots only for event E2, listings for event E1 → no match
        snaps = _make_snapshots(
            "E2",
            [
                (self._snap_ts(0), 100),
                (self._snap_ts(5), 60),
                (self._snap_ts(10), 30),
            ],
        )
        listings = _make_depletion_listings("E1", [self._listing_ts(1)])
        result = self.labeler.build_labels(listings, snaps)
        assert len(result) == 0, "E1 listings with no E1 snapshots should be dropped"

    def test_null_inventory_dropped_before_aggregation(self) -> None:
        """Rows with null inventory_remaining are excluded before sum aggregation."""
        # Mix of zones: one has null inventory (Ticketmaster-style), one has real inventory
        rows = [
            {
                "event_id": "E1",
                "seat_zone": "lower_tier",
                "timestamp": self._snap_ts(0),
                "inventory_remaining": 100,
            },
            {
                "event_id": "E1",
                "seat_zone": "upper_tier",
                "timestamp": self._snap_ts(0),
                "inventory_remaining": None,
            },
            {
                "event_id": "E1",
                "seat_zone": "lower_tier",
                "timestamp": self._snap_ts(5),
                "inventory_remaining": 70,
            },
            {
                "event_id": "E1",
                "seat_zone": "upper_tier",
                "timestamp": self._snap_ts(5),
                "inventory_remaining": None,
            },
            {
                "event_id": "E1",
                "seat_zone": "lower_tier",
                "timestamp": self._snap_ts(10),
                "inventory_remaining": 50,
            },
            {
                "event_id": "E1",
                "seat_zone": "upper_tier",
                "timestamp": self._snap_ts(10),
                "inventory_remaining": None,
            },
        ]
        snaps = pd.DataFrame(rows)
        snaps["inventory_remaining"] = snaps["inventory_remaining"].astype(float)
        listings = _make_depletion_listings("E1", [self._listing_ts(1)])

        # Should not raise; null rows silently dropped before aggregation
        result = self.labeler.build_labels(listings, snaps)
        assert len(result) > 0, "Should produce results despite null inventory rows"
        # depletion computed only on lower_tier (100→50 = 50% depletion > 0.3)
        assert result["sold"].iloc[0] == 1

    def test_label_depletion_rate_column_present(self) -> None:
        """_label_depletion_rate column is present in output."""
        snaps = _make_snapshots(
            "E1",
            [
                (self._snap_ts(0), 100),
                (self._snap_ts(5), 70),
                (self._snap_ts(10), 50),
            ],
        )
        listings = _make_depletion_listings("E1", [self._listing_ts(1)])
        result = self.labeler.build_labels(listings, snaps)
        if len(result) > 0:
            assert "_label_depletion_rate" in result.columns

    def test_zero_inventory_at_obs_labeled_not_sold(self) -> None:
        """Division by zero guard: inv_obs=0 → depletion_rate=0.0 → sold=0."""
        snaps = _make_snapshots(
            "E1",
            [
                (self._snap_ts(0), 0),
                (self._snap_ts(5), 0),
                (self._snap_ts(10), 0),
            ],
        )
        listings = _make_depletion_listings("E1", [self._listing_ts(1)])
        result = self.labeler.build_labels(listings, snaps)
        if len(result) > 0:
            assert (result["sold"] == 0).all(), "Zero inventory at obs → depletion=0 → not sold"
            assert (result["_label_depletion_rate"] == 0.0).all()

    def test_multiple_listings_same_timestamp_no_row_inflation(self) -> None:
        """Multiple listings at the same (event_id, timestamp) must not inflate row count.

        This guards against M:M join when multiple listings share a scrape timestamp.
        Without deduplication in future_lookup, row count would multiply incorrectly.
        """
        snaps = _make_snapshots(
            "E1",
            [
                (self._snap_ts(0), 100),
                (self._snap_ts(5), 70),
                (self._snap_ts(10), 50),
            ],
        )
        # 3 listings, ALL at the same timestamp (common: multiple listings per scrape)
        same_ts = self._listing_ts(1)
        rows = [
            {
                "listing_id": f"L{i:02d}",
                "event_id": "E1",
                "timestamp": same_ts,
                "event_datetime": EVENT_DT,
                "listing_price": 100.0,
            }
            for i in range(3)
        ]
        listings = pd.DataFrame(rows)
        result = self.labeler.build_labels(listings, snaps)
        # Output must have exactly 3 rows — no M:M inflation
        assert len(result) == 3, f"Expected 3 rows (one per listing), got {len(result)}"
