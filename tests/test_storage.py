"""Tests for storage module."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ticket_price_predictor.schemas import EventMetadata, EventType, PriceSnapshot, SeatZone
from ticket_price_predictor.storage import EventRepository, SnapshotRepository
from ticket_price_predictor.storage.parquet import read_parquet, write_parquet


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory."""
    return tmp_path / "data"


@pytest.fixture
def sample_event() -> EventMetadata:
    """Create a sample event."""
    return EventMetadata(
        event_id="test123",
        event_type=EventType.CONCERT,
        event_datetime=datetime.now(UTC) + timedelta(days=30),
        artist_or_team="Test Artist",
        venue_id="venue123",
        venue_name="Test Arena",
        city="Los Angeles",
        country="US",
        venue_capacity=20000,
    )


@pytest.fixture
def sample_snapshot() -> PriceSnapshot:
    """Create a sample snapshot."""
    return PriceSnapshot(
        event_id="test123",
        seat_zone=SeatZone.LOWER_TIER,
        timestamp=datetime.now(UTC),
        price_min=100.0,
        price_avg=150.0,
        price_max=200.0,
        inventory_remaining=50,
        days_to_event=30,
    )


class TestParquetIO:
    """Tests for Parquet read/write utilities."""

    def test_write_and_read_events(self, temp_data_dir: Path, sample_event: EventMetadata):
        """Test writing and reading events."""
        path = temp_data_dir / "events.parquet"

        # Write
        count = write_parquet([sample_event], path, EventMetadata.parquet_schema())
        assert count == 1

        # Read
        table = read_parquet(path)
        assert table is not None
        assert table.num_rows == 1

        records = table.to_pylist()
        assert records[0]["event_id"] == "test123"
        assert records[0]["event_type"] == "concert"

    def test_read_nonexistent_file(self, temp_data_dir: Path):
        """Test reading a file that doesn't exist."""
        table = read_parquet(temp_data_dir / "nonexistent.parquet")
        assert table is None

    def test_write_empty_list(self, temp_data_dir: Path):
        """Test writing an empty list."""
        path = temp_data_dir / "empty.parquet"
        count = write_parquet([], path, EventMetadata.parquet_schema())
        assert count == 0


class TestEventRepository:
    """Tests for EventRepository."""

    def test_save_and_get_event(self, temp_data_dir: Path, sample_event: EventMetadata):
        """Test saving and retrieving an event."""
        repo = EventRepository(temp_data_dir)

        # Save
        saved = repo.save_events([sample_event])
        assert saved == 1

        # Get
        loaded = repo.get_event("test123")
        assert loaded is not None
        assert loaded.event_id == sample_event.event_id
        assert loaded.artist_or_team == sample_event.artist_or_team

    def test_save_duplicates(self, temp_data_dir: Path, sample_event: EventMetadata):
        """Test that duplicates are not saved."""
        repo = EventRepository(temp_data_dir)

        # Save first time
        saved1 = repo.save_events([sample_event])
        assert saved1 == 1

        # Save again (duplicate)
        saved2 = repo.save_events([sample_event])
        assert saved2 == 0

    def test_list_event_ids(self, temp_data_dir: Path, sample_event: EventMetadata):
        """Test listing event IDs."""
        repo = EventRepository(temp_data_dir)

        # Empty initially
        assert repo.list_event_ids() == []

        # Save events
        repo.save_events([sample_event])
        ids = repo.list_event_ids()
        assert "test123" in ids

    def test_get_events_with_filter(self, temp_data_dir: Path):
        """Test getting events with filters."""
        repo = EventRepository(temp_data_dir)

        # Create events of different types
        concert = EventMetadata(
            event_id="concert1",
            event_type=EventType.CONCERT,
            event_datetime=datetime.now(UTC) + timedelta(days=30),
            artist_or_team="Concert Artist",
            venue_id="venue1",
            venue_name="Arena",
            city="LA",
        )
        sports = EventMetadata(
            event_id="sports1",
            event_type=EventType.SPORTS,
            event_datetime=datetime.now(UTC) + timedelta(days=30),
            artist_or_team="Sports Team",
            venue_id="venue2",
            venue_name="Stadium",
            city="NY",
        )

        repo.save_events([concert, sports])

        # Filter by type
        from ticket_price_predictor.storage.repository import EventFilters

        concerts = repo.get_events(EventFilters(event_type=EventType.CONCERT))
        assert len(concerts) == 1
        assert concerts[0].event_id == "concert1"


class TestSnapshotRepository:
    """Tests for SnapshotRepository."""

    def test_save_and_get_snapshots(self, temp_data_dir: Path, sample_snapshot: PriceSnapshot):
        """Test saving and retrieving snapshots."""
        repo = SnapshotRepository(temp_data_dir)

        # Save
        saved = repo.save_snapshots([sample_snapshot])
        assert saved == 1

        # Get
        loaded = repo.get_snapshots(event_id="test123")
        assert len(loaded) == 1
        assert loaded[0].event_id == sample_snapshot.event_id
        assert loaded[0].price_min == sample_snapshot.price_min

    def test_get_latest_snapshot(self, temp_data_dir: Path):
        """Test getting the latest snapshot."""
        repo = SnapshotRepository(temp_data_dir)

        now = datetime.now(UTC)

        # Create snapshots at different times
        old = PriceSnapshot(
            event_id="test123",
            seat_zone=SeatZone.LOWER_TIER,
            timestamp=now - timedelta(hours=1),
            price_min=100.0,
            days_to_event=30,
        )
        new = PriceSnapshot(
            event_id="test123",
            seat_zone=SeatZone.LOWER_TIER,
            timestamp=now,
            price_min=110.0,
            days_to_event=29,
        )

        repo.save_snapshots([old, new])

        latest = repo.get_latest_snapshot("test123", SeatZone.LOWER_TIER)
        assert latest is not None
        assert latest.price_min == 110.0

    def test_partition_by_month(self, temp_data_dir: Path):
        """Test that snapshots are partitioned by year/month."""
        repo = SnapshotRepository(temp_data_dir)

        snapshot = PriceSnapshot(
            event_id="test123",
            seat_zone=SeatZone.FLOOR_VIP,
            timestamp=datetime(2026, 3, 15, tzinfo=UTC),
            price_min=200.0,
            days_to_event=30,
        )

        repo.save_snapshots([snapshot])

        # Check partition path exists
        partition_path = temp_data_dir / "raw" / "snapshots" / "year=2026" / "month=03"
        assert partition_path.exists()
