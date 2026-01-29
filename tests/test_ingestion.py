"""Tests for ingestion module."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from ticket_price_predictor.ingestion import SnapshotCollector
from ticket_price_predictor.schemas import SeatZone
from ticket_price_predictor.storage import EventRepository, SnapshotRepository


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory."""
    return tmp_path / "data"


@pytest.fixture
def event_repo(temp_data_dir: Path) -> EventRepository:
    """Create an event repository."""
    return EventRepository(temp_data_dir)


@pytest.fixture
def snapshot_repo(temp_data_dir: Path) -> SnapshotRepository:
    """Create a snapshot repository."""
    return SnapshotRepository(temp_data_dir)


@pytest.fixture
def sample_raw_event() -> dict[str, Any]:
    """Create a sample raw event from Ticketmaster API."""
    return {
        "id": "test123",
        "name": "Test Concert",
        "dates": {
            "start": {
                "dateTime": (datetime.now(UTC) + timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        },
        "priceRanges": [
            {
                "type": "standard",
                "currency": "USD",
                "min": 50.0,
                "max": 200.0,
            }
        ],
        "classifications": [
            {
                "segment": {"name": "Music"},
                "genre": {"name": "Rock"},
            }
        ],
        "_embedded": {
            "venues": [
                {
                    "id": "venue123",
                    "name": "Test Arena",
                    "city": {"name": "Los Angeles"},
                    "country": {"countryCode": "US"},
                }
            ],
            "attractions": [
                {"name": "Test Artist"},
            ],
        },
    }


class TestSnapshotCollector:
    """Tests for SnapshotCollector."""

    def test_create_snapshots_from_event(
        self,
        event_repo: EventRepository,
        snapshot_repo: SnapshotRepository,
        sample_raw_event: dict[str, Any],
    ):
        """Test creating snapshots from a raw event."""
        collector = SnapshotCollector(event_repo, snapshot_repo)

        event_datetime = datetime.now(UTC) + timedelta(days=30)
        timestamp = datetime.now(UTC)

        snapshots = collector.create_snapshots_from_event(
            sample_raw_event,
            event_datetime,
            timestamp,
        )

        # Should create one snapshot per zone
        assert len(snapshots) == 4

        # Check zone prices are derived correctly
        zone_prices = {s.seat_zone: s.price_min for s in snapshots}

        # Floor VIP should be at max (200)
        assert zone_prices[SeatZone.FLOOR_VIP] == 200.0

        # Lower tier should be 70% of range
        assert zone_prices[SeatZone.LOWER_TIER] == 155.0  # 50 + (200-50) * 0.70

        # All should have same event_id
        assert all(s.event_id == "test123" for s in snapshots)

        # All should have approximately 30 days to event
        assert all(s.days_to_event >= 29 for s in snapshots)

    def test_create_snapshots_no_price_range(
        self,
        event_repo: EventRepository,
        snapshot_repo: SnapshotRepository,
    ):
        """Test handling events without price ranges."""
        collector = SnapshotCollector(event_repo, snapshot_repo)

        raw_event = {
            "id": "test456",
            "priceRanges": [],  # No price ranges
        }

        snapshots = collector.create_snapshots_from_event(
            raw_event,
            datetime.now(UTC) + timedelta(days=30),
            datetime.now(UTC),
        )

        assert len(snapshots) == 0

    def test_create_snapshot_for_zone(
        self,
        event_repo: EventRepository,
        snapshot_repo: SnapshotRepository,
    ):
        """Test creating a single snapshot for a zone."""
        collector = SnapshotCollector(event_repo, snapshot_repo)

        event_datetime = datetime.now(UTC) + timedelta(days=15)
        snapshot = collector.create_snapshot_for_zone(
            event_id="test123",
            seat_zone=SeatZone.LOWER_TIER,
            price=150.0,
            event_datetime=event_datetime,
            inventory=100,
        )

        assert snapshot.event_id == "test123"
        assert snapshot.seat_zone == SeatZone.LOWER_TIER
        assert snapshot.price_min == 150.0
        assert snapshot.inventory_remaining == 100
        assert snapshot.days_to_event >= 14


class TestIngestionResult:
    """Tests for IngestionResult."""

    def test_success(self):
        """Test successful result."""
        from ticket_price_predictor.ingestion import IngestionResult

        result = IngestionResult(events_fetched=10, events_saved=10)
        assert result.success

    def test_failure_with_errors(self):
        """Test result with errors."""
        from ticket_price_predictor.ingestion import IngestionResult

        result = IngestionResult(events_fetched=10, events_saved=5, errors=["Error 1"])
        assert not result.success


class TestCollectionResult:
    """Tests for CollectionResult."""

    def test_success(self):
        """Test successful result."""
        from ticket_price_predictor.ingestion import CollectionResult

        result = CollectionResult(events_processed=5, snapshots_created=20)
        assert result.success

    def test_failure_with_errors(self):
        """Test result with errors."""
        from ticket_price_predictor.ingestion import CollectionResult

        result = CollectionResult(events_processed=5, snapshots_created=15, errors=["Error 1"])
        assert not result.success
