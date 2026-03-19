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
