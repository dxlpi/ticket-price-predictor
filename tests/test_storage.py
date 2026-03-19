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


class TestListingRepositoryDedupHash:
    """Tests for compute_listing_hash() dedup hash redesign."""

    def _make_listing(self, price: float):
        from ticket_price_predictor.schemas import TicketListing

        return TicketListing(
            listing_id="l1",
            event_id="e1",
            source="vividseats",
            timestamp=datetime.now(UTC),
            event_name="Test Show",
            artist_or_team="Artist",
            venue_name="Venue",
            city="LA",
            event_datetime=datetime.now(UTC) + timedelta(days=30),
            section="Floor A",
            row="1",
            seat_from="1",
            seat_to="2",
            quantity=2,
            listing_price=price,
            total_price=price * 2,
            days_to_event=30,
        )

    def test_same_price_same_hash(self):
        """Same seat at same price produces same hash."""
        from ticket_price_predictor.storage.repository import ListingRepository

        l1 = self._make_listing(100.0)
        l2 = self._make_listing(100.0)
        assert ListingRepository.compute_listing_hash(l1) == ListingRepository.compute_listing_hash(
            l2
        )

    def test_different_price_different_hash(self):
        """Same seat at different price produces different hash (preserves price change signal)."""
        from ticket_price_predictor.storage.repository import ListingRepository

        l1 = self._make_listing(100.0)
        l2 = self._make_listing(150.0)
        assert ListingRepository.compute_listing_hash(l1) != ListingRepository.compute_listing_hash(
            l2
        )


class TestAtomicParquetWrite:
    """Tests for crash-safe parquet writes."""

    def test_no_tmp_file_after_write_parquet(self, tmp_path: Path, sample_event: EventMetadata):
        """write_parquet leaves no .parquet.tmp after successful write."""
        path = tmp_path / "events.parquet"
        write_parquet([sample_event], path, EventMetadata.parquet_schema())
        assert not path.with_suffix(".parquet.tmp").exists()
        assert path.exists()

    def test_no_tmp_file_after_append_parquet(self, tmp_path: Path, sample_event: EventMetadata):
        """append_parquet leaves no .parquet.tmp after successful write."""
        from ticket_price_predictor.storage.parquet import append_parquet

        path = tmp_path / "events.parquet"
        append_parquet([sample_event], path, EventMetadata.parquet_schema())
        assert not path.with_suffix(".parquet.tmp").exists()
        assert path.exists()


class TestSnapshotProvenance:
    """Tests for PriceSnapshot source provenance field."""

    def test_snapshot_default_source(self):
        """PriceSnapshot defaults to vividseats source."""
        s = PriceSnapshot(
            event_id="e1",
            seat_zone=SeatZone.FLOOR_VIP,
            timestamp=datetime.now(UTC),
            price_min=100.0,
            days_to_event=30,
        )
        assert s.source == "vividseats"

    def test_snapshot_ticketmaster_source(self):
        """PriceSnapshot accepts ticketmaster source."""
        s = PriceSnapshot(
            event_id="e1",
            seat_zone=SeatZone.FLOOR_VIP,
            timestamp=datetime.now(UTC),
            price_min=100.0,
            days_to_event=30,
            source="ticketmaster",
        )
        assert s.source == "ticketmaster"

    def test_snapshot_roundtrip_source(self, temp_data_dir: Path):
        """PriceSnapshot source field survives save/load roundtrip."""
        repo = SnapshotRepository(temp_data_dir)
        s = PriceSnapshot(
            event_id="e1",
            seat_zone=SeatZone.LOWER_TIER,
            timestamp=datetime.now(UTC),
            price_min=100.0,
            days_to_event=30,
            source="ticketmaster",
        )
        repo.save_snapshots([s])
        loaded = repo.get_snapshots(event_id="e1")
        assert len(loaded) == 1
        assert loaded[0].source == "ticketmaster"

    def test_snapshot_backward_compat_missing_source(self):
        """PriceSnapshot deserialized from dict without source defaults to vividseats."""
        from ticket_price_predictor.storage.repository import SnapshotRepository

        data = {
            "event_id": "e1",
            "seat_zone": "lower_tier",
            "timestamp": datetime.now(UTC),
            "price_min": 100.0,
            "price_avg": None,
            "price_max": None,
            "inventory_remaining": None,
            "days_to_event": 30,
            # source intentionally missing
        }
        repo = SnapshotRepository(Path("/tmp"))
        snapshot = repo._dict_to_snapshot(data)
        assert snapshot.source == "vividseats"
