"""Tests for data schemas."""

from datetime import UTC, datetime

import pyarrow as pa
import pytest

from ticket_price_predictor.schemas import EventMetadata, EventType, PriceSnapshot, SeatZone


class TestEventMetadata:
    """Tests for EventMetadata schema."""

    def test_create_valid_metadata(self):
        """Test creating valid event metadata."""
        metadata = EventMetadata(
            event_id="test123",
            event_type=EventType.CONCERT,
            event_datetime=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
            artist_or_team="Test Artist",
            venue_id="venue123",
            venue_name="Test Arena",
            city="Los Angeles",
            country="US",
            venue_capacity=20000,
        )

        assert metadata.event_id == "test123"
        assert metadata.event_type == EventType.CONCERT
        assert metadata.city == "Los Angeles"
        assert metadata.venue_capacity == 20000

    def test_default_country(self):
        """Test default country is US."""
        metadata = EventMetadata(
            event_id="test123",
            event_type=EventType.SPORTS,
            event_datetime=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
            artist_or_team="Test Team",
            venue_id="venue123",
            venue_name="Test Stadium",
            city="Chicago",
        )

        assert metadata.country == "US"

    def test_parquet_schema(self):
        """Test Parquet schema generation."""
        schema = EventMetadata.parquet_schema()

        assert isinstance(schema, pa.Schema)
        assert "event_id" in schema.names
        assert "event_type" in schema.names
        assert "event_datetime" in schema.names
        assert len(schema) == 9


class TestPriceSnapshot:
    """Tests for PriceSnapshot schema."""

    def test_create_valid_snapshot(self):
        """Test creating valid price snapshot."""
        snapshot = PriceSnapshot(
            event_id="test123",
            seat_zone=SeatZone.LOWER_TIER,
            timestamp=datetime(2024, 10, 1, 12, 0, tzinfo=UTC),
            price_min=100.0,
            price_avg=150.0,
            price_max=200.0,
            inventory_remaining=50,
            days_to_event=14,
        )

        assert snapshot.event_id == "test123"
        assert snapshot.seat_zone == SeatZone.LOWER_TIER
        assert snapshot.price_min == 100.0
        assert snapshot.days_to_event == 14

    def test_minimal_snapshot(self):
        """Test creating snapshot with only required fields."""
        snapshot = PriceSnapshot(
            event_id="test123",
            seat_zone=SeatZone.UPPER_TIER,
            timestamp=datetime(2024, 10, 1, 12, 0, tzinfo=UTC),
            price_min=50.0,
            days_to_event=7,
        )

        assert snapshot.price_avg is None
        assert snapshot.price_max is None
        assert snapshot.inventory_remaining is None

    def test_invalid_price_avg_less_than_min(self):
        """Test validation fails when price_avg < price_min."""
        with pytest.raises(ValueError, match="price_avg must be >= price_min"):
            PriceSnapshot(
                event_id="test123",
                seat_zone=SeatZone.FLOOR_VIP,
                timestamp=datetime(2024, 10, 1, 12, 0, tzinfo=UTC),
                price_min=100.0,
                price_avg=50.0,  # Invalid: less than min
                days_to_event=7,
            )

    def test_invalid_price_max_less_than_min(self):
        """Test validation fails when price_max < price_min."""
        with pytest.raises(ValueError, match="price_max must be >= price_min"):
            PriceSnapshot(
                event_id="test123",
                seat_zone=SeatZone.FLOOR_VIP,
                timestamp=datetime(2024, 10, 1, 12, 0, tzinfo=UTC),
                price_min=100.0,
                price_max=50.0,  # Invalid: less than min
                days_to_event=7,
            )

    def test_invalid_price_avg_greater_than_max(self):
        """Test validation fails when price_avg > price_max."""
        with pytest.raises(ValueError, match="price_avg must be <= price_max"):
            PriceSnapshot(
                event_id="test123",
                seat_zone=SeatZone.FLOOR_VIP,
                timestamp=datetime(2024, 10, 1, 12, 0, tzinfo=UTC),
                price_min=100.0,
                price_avg=200.0,
                price_max=150.0,  # Invalid: less than avg
                days_to_event=7,
            )

    def test_negative_price_rejected(self):
        """Test negative prices are rejected."""
        with pytest.raises(ValueError):
            PriceSnapshot(
                event_id="test123",
                seat_zone=SeatZone.BALCONY,
                timestamp=datetime(2024, 10, 1, 12, 0, tzinfo=UTC),
                price_min=-10.0,  # Invalid: negative
                days_to_event=7,
            )

    def test_parquet_schema(self):
        """Test Parquet schema generation."""
        schema = PriceSnapshot.parquet_schema()

        assert isinstance(schema, pa.Schema)
        assert "event_id" in schema.names
        assert "seat_zone" in schema.names
        assert "price_min" in schema.names
        assert "source" in schema.names
        assert len(schema) == 9


class TestSeatZone:
    """Tests for SeatZone enum."""

    def test_all_zones_defined(self):
        """Test all expected seat zones are defined."""
        zones = list(SeatZone)
        assert len(zones) == 4
        assert SeatZone.FLOOR_VIP in zones
        assert SeatZone.LOWER_TIER in zones
        assert SeatZone.UPPER_TIER in zones
        assert SeatZone.BALCONY in zones

    def test_zone_values(self):
        """Test seat zone string values."""
        assert SeatZone.FLOOR_VIP.value == "floor_vip"
        assert SeatZone.LOWER_TIER.value == "lower_tier"


class TestEventTypeComedy:
    """Tests for EventType.COMEDY addition."""

    def test_comedy_exists(self):
        """EventType.COMEDY is defined with value 'comedy'."""
        assert EventType.COMEDY == "comedy"
        assert EventType.COMEDY.value == "comedy"

    def test_comedy_in_enum_members(self):
        """EventType.COMEDY appears in the full enum listing."""
        assert EventType.COMEDY in list(EventType)

    def test_all_four_event_types_defined(self):
        """All four event types (concert, sports, theater, comedy) are present."""
        values = {e.value for e in EventType}
        assert values == {"concert", "sports", "theater", "comedy"}


class TestTicketListingEventType:
    """Tests for TicketListing.event_type field."""

    def _base_listing(self, **overrides):
        from datetime import UTC, datetime, timedelta

        from ticket_price_predictor.schemas import TicketListing

        defaults = {
            "listing_id": "l1",
            "event_id": "e1",
            "source": "vividseats",
            "timestamp": datetime.now(UTC),
            "event_name": "Test Show",
            "artist_or_team": "Artist",
            "venue_name": "Venue",
            "city": "LA",
            "event_datetime": datetime.now(UTC) + timedelta(days=30),
            "section": "Floor A",
            "row": "1",
            "quantity": 2,
            "listing_price": 100.0,
            "total_price": 200.0,
            "days_to_event": 30,
        }
        defaults.update(overrides)
        return TicketListing(**defaults)

    def test_event_type_accepts_string(self):
        """TicketListing accepts event_type as a plain string."""
        listing = self._base_listing(event_type="comedy")
        assert listing.event_type == "comedy"

    def test_event_type_defaults_to_none(self):
        """TicketListing.event_type defaults to None when not provided."""
        listing = self._base_listing()
        assert listing.event_type is None

    def test_event_type_accepts_none_explicitly(self):
        """TicketListing.event_type can be set to None explicitly."""
        listing = self._base_listing(event_type=None)
        assert listing.event_type is None

    def test_parquet_schema_includes_event_type(self):
        """TicketListing.parquet_schema() includes a nullable string event_type field."""
        import pyarrow as pa

        from ticket_price_predictor.schemas import TicketListing

        schema = TicketListing.parquet_schema()
        assert "event_type" in schema.names
        idx = schema.get_field_index("event_type")
        field = schema.field(idx)
        assert pa.types.is_string(field.type)
        assert field.nullable


class TestScrapedEventEventType:
    """Tests for ScrapedEvent.event_type field."""

    def test_scraped_event_accepts_event_type(self):
        """ScrapedEvent accepts event_type string."""
        from datetime import UTC, datetime

        from ticket_price_predictor.schemas import ScrapedEvent

        ev = ScrapedEvent(
            stubhub_event_id="s1",
            event_name="Comedy Night",
            artist_or_team="Dave Chappelle",
            venue_name="The Venue",
            city="Chicago",
            event_datetime=datetime.now(UTC),
            event_url="https://example.com",
            event_type="comedy",
        )
        assert ev.event_type == "comedy"

    def test_scraped_event_event_type_defaults_none(self):
        """ScrapedEvent.event_type defaults to None."""
        from datetime import UTC, datetime

        from ticket_price_predictor.schemas import ScrapedEvent

        ev = ScrapedEvent(
            stubhub_event_id="s2",
            event_name="Show",
            artist_or_team="Artist",
            venue_name="Venue",
            city="NYC",
            event_datetime=datetime.now(UTC),
            event_url="https://example.com",
        )
        assert ev.event_type is None


class TestCreateListingFromScraped:
    """Tests for create_listing_from_scraped threading event_type."""

    def _make_scraped_listing(self):
        from ticket_price_predictor.schemas import ScrapedListing

        return ScrapedListing(
            listing_id="lst1",
            section="Floor",
            row="1",
            quantity=2,
            price_per_ticket=150.0,
            total_price=300.0,
        )

    def _make_scraped_event(self, event_type=None):
        from datetime import UTC, datetime, timedelta

        from ticket_price_predictor.schemas import ScrapedEvent

        return ScrapedEvent(
            stubhub_event_id="ev1",
            event_name="Show",
            artist_or_team="Artist",
            venue_name="Venue",
            city="NYC",
            event_datetime=datetime.now(UTC) + timedelta(days=10),
            event_url="https://example.com",
            event_type=event_type,
        )

    def test_threads_event_type_to_listing(self):
        """create_listing_from_scraped copies event_type from ScrapedEvent."""
        from ticket_price_predictor.schemas import create_listing_from_scraped

        event = self._make_scraped_event(event_type="comedy")
        listing = create_listing_from_scraped(self._make_scraped_listing(), event)
        assert listing.event_type == "comedy"

    def test_none_event_type_passes_through(self):
        """create_listing_from_scraped sets event_type=None when ScrapedEvent has none."""
        from ticket_price_predictor.schemas import create_listing_from_scraped

        event = self._make_scraped_event(event_type=None)
        listing = create_listing_from_scraped(self._make_scraped_listing(), event)
        assert listing.event_type is None
