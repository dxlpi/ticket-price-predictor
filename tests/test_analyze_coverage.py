"""Tests for scripts/analyze_coverage.py — pure compute_coverage() function."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from analyze_coverage import compute_coverage


def _make_df(
    events: list[dict[str, object]],
    listings_per_event: int = 5,
) -> pd.DataFrame:
    """Build a synthetic listings DataFrame from event specs.

    Each event spec: {"event_id": str, "event_datetime": str, "artist": str, "city": str}
    """
    rows = []
    for spec in events:
        for i in range(listings_per_event):
            rows.append(
                {
                    "event_id": spec["event_id"],
                    "event_datetime": spec["event_datetime"],
                    "artist_or_team": spec["artist"],
                    "city": spec["city"],
                    "listing_price": 100.0 + i,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core unseen-event fraction
# ---------------------------------------------------------------------------


class TestUnseen:
    def test_basic_unseen_fraction(self) -> None:
        """8 events in train period, 2 events only in test → unseen_pct = 0.5."""
        # 10 events; first 8 by date land in train (80% of listings at ratio=0.80)
        events = [
            {
                "event_id": f"e{i:02d}",
                "event_datetime": f"2024-0{i + 1}-01",
                "artist": "A",
                "city": "NYC",
            }
            for i in range(8)
        ] + [
            {"event_id": "e08", "event_datetime": "2024-09-15", "artist": "B", "city": "LA"},
            {"event_id": "e09", "event_datetime": "2024-10-01", "artist": "C", "city": "LA"},
        ]
        df = _make_df(events, listings_per_event=5)

        # With ratio=0.80, first 80% of listings (40/50) → train covers e00..e07
        metrics = compute_coverage(df, train_ratio=0.80)

        assert metrics["n_listings_total"] == 50
        assert metrics["n_listings_train"] == 40
        assert metrics["n_listings_test"] == 10
        # e08 and e09 are unseen
        assert metrics["n_events_unseen"] == 2
        assert metrics["n_events_test"] == 2
        assert metrics["unseen_event_pct_by_event"] == pytest.approx(1.0)

    def test_no_unseen_when_all_events_in_train(self) -> None:
        """If test events are a subset of train events, unseen_pct = 0."""
        # Single event spanning many listings: all end up in both train and test
        events = [{"event_id": "e1", "event_datetime": "2024-06-01", "artist": "A", "city": "NYC"}]
        df = _make_df(events, listings_per_event=20)
        metrics = compute_coverage(df, train_ratio=0.85)
        # e1 appears in both train and test slices → 0 unseen
        assert metrics["n_events_unseen"] == 0
        assert metrics["unseen_event_pct_by_event"] == pytest.approx(0.0)

    def test_exact_half_unseen(self) -> None:
        """3 early events in train, 1 late event only in test → unseen_pct = 1.0."""
        # 4 events × 5 listings = 20 total. ratio=0.80 → 16 train / 4 test.
        # Sorted by date: eA(5) + eB(5) + eC(5) + eD(5).
        # split_idx=16: train = eA(5)+eB(5)+eC(5)+1 from eD → eD in both splits,
        # so this would be 0. Instead use ratio=0.75 → 15 train / 5 test.
        # train = eA(5)+eB(5)+eC(5), test = eD(5) → eD unseen.
        events = [
            {"event_id": "eA", "event_datetime": "2024-01-01", "artist": "X", "city": "NYC"},
            {"event_id": "eB", "event_datetime": "2024-02-01", "artist": "X", "city": "NYC"},
            {"event_id": "eC", "event_datetime": "2024-03-01", "artist": "Y", "city": "LA"},
            {"event_id": "eD", "event_datetime": "2024-09-01", "artist": "Z", "city": "SF"},
        ]
        df = _make_df(events, listings_per_event=5)
        # 20 listings; ratio=0.75 → 15 train / 5 test
        # train = eA+eB+eC (all 15 listings) → test = eD only → eD fully unseen
        metrics = compute_coverage(df, train_ratio=0.75)

        assert metrics["n_events_unseen"] == 1
        assert metrics["n_events_test"] == 1
        assert metrics["unseen_event_pct_by_event"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------


class TestHistogram:
    def test_histogram_buckets(self) -> None:
        """Artists with 1, 2, 5, 12 events land in correct buckets."""
        events = (
            [{"event_id": "s1", "event_datetime": "2024-01-01", "artist": "Solo", "city": "NYC"}]
            + [
                {
                    "event_id": f"d{i}",
                    "event_datetime": f"2024-02-0{i + 1}",
                    "artist": "Duo",
                    "city": "NYC",
                }
                for i in range(2)
            ]
            + [
                {
                    "event_id": f"m{i}",
                    "event_datetime": f"2024-03-{10 + i:02d}",
                    "artist": "Med",
                    "city": "LA",
                }
                for i in range(5)
            ]
            + [
                {
                    "event_id": f"h{i}",
                    "event_datetime": f"2024-04-{10 + i:02d}",
                    "artist": "High",
                    "city": "LA",
                }
                for i in range(12)
            ]
        )
        df = _make_df(events, listings_per_event=3)
        metrics = compute_coverage(df, train_ratio=0.85)
        hist = metrics["events_per_artist_histogram"]

        assert hist["1"] >= 1
        assert hist["2-3"] >= 1
        assert hist["4-10"] >= 1
        assert hist["11+"] >= 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_dataframe(self) -> None:
        """Empty input returns zeroed metrics without errors."""
        metrics = compute_coverage(pd.DataFrame(), train_ratio=0.85)
        assert metrics["n_listings_total"] == 0
        assert metrics["unseen_event_pct_by_event"] == pytest.approx(0.0)
        assert metrics["underrepresented_regions"] == []
        assert metrics["events_per_artist_histogram"] == {}

    def test_underrepresented_regions_threshold(self) -> None:
        """Regions with < 5 events per city×month appear in underrepresented list."""
        # 3 events in Chicago, Jan 2024 → should be underrepresented
        events = [
            {
                "event_id": f"chi{i}",
                "event_datetime": "2024-01-15",
                "artist": f"Band{i}",
                "city": "Chicago",
            }
            for i in range(3)
        ] + [
            # 6 events in NYC, Jan 2024 → should NOT be underrepresented
            {
                "event_id": f"nyc{i}",
                "event_datetime": "2024-01-20",
                "artist": f"Act{i}",
                "city": "NYC",
            }
            for i in range(6)
        ]
        df = _make_df(events, listings_per_event=4)
        metrics = compute_coverage(df, train_ratio=0.85)

        region_months = [r["region_month"] for r in metrics["underrepresented_regions"]]
        assert any("Chicago" in rm for rm in region_months)
        assert not any("NYC" in rm for rm in region_months)

    def test_split_ratio_boundary(self) -> None:
        """split_ratio=1.0 puts everything in train → test is empty, pct=0."""
        events = [{"event_id": "e1", "event_datetime": "2024-01-01", "artist": "A", "city": "NYC"}]
        df = _make_df(events, listings_per_event=10)
        metrics = compute_coverage(df, train_ratio=1.0)
        assert metrics["n_listings_test"] == 0
        assert metrics["unseen_event_pct_by_event"] == pytest.approx(0.0)
