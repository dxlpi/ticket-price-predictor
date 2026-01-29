"""Tests for validation module."""

from datetime import UTC, datetime, timedelta

import pytest

from ticket_price_predictor.schemas import EventMetadata, EventType, PriceSnapshot, SeatZone
from ticket_price_predictor.validation import BatchValidationResult, DataValidator, ValidationResult


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_valid_result(self):
        """Test creating a valid result."""
        result = ValidationResult(is_valid=True)
        assert result.is_valid
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_invalid_result(self):
        """Test creating an invalid result."""
        result = ValidationResult(is_valid=False, errors=["Error 1", "Error 2"])
        assert not result.is_valid
        assert len(result.errors) == 2


class TestBatchValidationResult:
    """Tests for BatchValidationResult."""

    def test_all_valid(self):
        """Test when all records are valid."""
        result = BatchValidationResult(
            total_records=10,
            valid_records=10,
            invalid_records=0,
        )
        assert result.is_valid
        assert result.error_rate == 0.0

    def test_some_invalid(self):
        """Test when some records are invalid."""
        result = BatchValidationResult(
            total_records=10,
            valid_records=7,
            invalid_records=3,
        )
        assert not result.is_valid
        assert result.error_rate == 0.3

    def test_empty_batch(self):
        """Test error rate for empty batch."""
        result = BatchValidationResult(
            total_records=0,
            valid_records=0,
            invalid_records=0,
        )
        assert result.error_rate == 0.0


class TestDataValidatorEvents:
    """Tests for event validation."""

    @pytest.fixture
    def validator(self) -> DataValidator:
        """Create a validator instance."""
        return DataValidator()

    @pytest.fixture
    def valid_event(self) -> EventMetadata:
        """Create a valid event."""
        return EventMetadata(
            event_id="test123",
            event_type=EventType.CONCERT,
            event_datetime=datetime.now(UTC) + timedelta(days=30),
            artist_or_team="Test Artist",
            venue_id="venue123",
            venue_name="Test Arena",
            city="Los Angeles",
        )

    def test_valid_event(self, validator: DataValidator, valid_event: EventMetadata):
        """Test validating a valid event."""
        result = validator.validate_event(valid_event)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_empty_event_id(self, validator: DataValidator, valid_event: EventMetadata):
        """Test that empty event_id is an error."""
        event = EventMetadata(
            event_id="",
            event_type=valid_event.event_type,
            event_datetime=valid_event.event_datetime,
            artist_or_team=valid_event.artist_or_team,
            venue_id=valid_event.venue_id,
            venue_name=valid_event.venue_name,
            city=valid_event.city,
        )
        result = validator.validate_event(event)
        assert not result.is_valid
        assert any("event_id" in e for e in result.errors)

    def test_empty_venue_id(self, validator: DataValidator, valid_event: EventMetadata):
        """Test that empty venue_id is an error."""
        event = EventMetadata(
            event_id=valid_event.event_id,
            event_type=valid_event.event_type,
            event_datetime=valid_event.event_datetime,
            artist_or_team=valid_event.artist_or_team,
            venue_id="",
            venue_name=valid_event.venue_name,
            city=valid_event.city,
        )
        result = validator.validate_event(event)
        assert not result.is_valid
        assert any("venue_id" in e for e in result.errors)

    def test_past_event_warning(self, validator: DataValidator):
        """Test that past events generate a warning."""
        event = EventMetadata(
            event_id="test123",
            event_type=EventType.CONCERT,
            event_datetime=datetime.now(UTC) - timedelta(days=1),
            artist_or_team="Test Artist",
            venue_id="venue123",
            venue_name="Test Arena",
            city="Los Angeles",
        )
        result = validator.validate_event(event)
        assert result.is_valid  # Still valid, just warning
        assert any("past" in w for w in result.warnings)

    def test_allow_past_events(self):
        """Test allowing past events."""
        validator = DataValidator(allow_past_events=True)
        event = EventMetadata(
            event_id="test123",
            event_type=EventType.CONCERT,
            event_datetime=datetime.now(UTC) - timedelta(days=1),
            artist_or_team="Test Artist",
            venue_id="venue123",
            venue_name="Test Arena",
            city="Los Angeles",
        )
        result = validator.validate_event(event)
        assert result.is_valid
        assert len(result.warnings) == 0

    def test_negative_capacity(self, validator: DataValidator, valid_event: EventMetadata):
        """Test that negative capacity is an error."""
        event = EventMetadata(
            event_id=valid_event.event_id,
            event_type=valid_event.event_type,
            event_datetime=valid_event.event_datetime,
            artist_or_team=valid_event.artist_or_team,
            venue_id=valid_event.venue_id,
            venue_name=valid_event.venue_name,
            city=valid_event.city,
            venue_capacity=-100,
        )
        result = validator.validate_event(event)
        assert not result.is_valid
        assert any("capacity" in e for e in result.errors)


class TestDataValidatorSnapshots:
    """Tests for snapshot validation."""

    @pytest.fixture
    def validator(self) -> DataValidator:
        """Create a validator instance."""
        return DataValidator()

    @pytest.fixture
    def valid_snapshot(self) -> PriceSnapshot:
        """Create a valid snapshot."""
        return PriceSnapshot(
            event_id="test123",
            seat_zone=SeatZone.LOWER_TIER,
            timestamp=datetime.now(UTC),
            price_min=100.0,
            price_avg=150.0,
            price_max=200.0,
            days_to_event=30,
        )

    def test_valid_snapshot(self, validator: DataValidator, valid_snapshot: PriceSnapshot):
        """Test validating a valid snapshot."""
        result = validator.validate_snapshot(valid_snapshot)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_empty_event_id(self, validator: DataValidator):
        """Test that empty event_id is an error."""
        snapshot = PriceSnapshot(
            event_id="",
            seat_zone=SeatZone.LOWER_TIER,
            timestamp=datetime.now(UTC),
            price_min=100.0,
            days_to_event=30,
        )
        result = validator.validate_snapshot(snapshot)
        assert not result.is_valid
        assert any("event_id" in e for e in result.errors)

    def test_negative_days_to_event(self, validator: DataValidator):
        """Test that negative days_to_event is an error."""
        snapshot = PriceSnapshot(
            event_id="test123",
            seat_zone=SeatZone.LOWER_TIER,
            timestamp=datetime.now(UTC),
            price_min=100.0,
            days_to_event=-5,
        )
        result = validator.validate_snapshot(snapshot)
        assert not result.is_valid
        assert any("days_to_event" in e for e in result.errors)

    def test_future_timestamp_warning(self, validator: DataValidator):
        """Test that future timestamps generate a warning."""
        snapshot = PriceSnapshot(
            event_id="test123",
            seat_zone=SeatZone.LOWER_TIER,
            timestamp=datetime.now(UTC) + timedelta(hours=1),
            price_min=100.0,
            days_to_event=30,
        )
        result = validator.validate_snapshot(snapshot)
        assert result.is_valid  # Still valid, just warning
        assert any("future" in w for w in result.warnings)


class TestBatchValidation:
    """Tests for batch validation."""

    @pytest.fixture
    def validator(self) -> DataValidator:
        """Create a validator instance."""
        return DataValidator()

    def test_validate_events_batch(self, validator: DataValidator):
        """Test validating a batch of events."""
        events = [
            EventMetadata(
                event_id="valid1",
                event_type=EventType.CONCERT,
                event_datetime=datetime.now(UTC) + timedelta(days=30),
                artist_or_team="Artist 1",
                venue_id="venue1",
                venue_name="Arena 1",
                city="LA",
            ),
            EventMetadata(
                event_id="",  # Invalid
                event_type=EventType.SPORTS,
                event_datetime=datetime.now(UTC) + timedelta(days=30),
                artist_or_team="Team 1",
                venue_id="venue2",
                venue_name="Stadium 1",
                city="NY",
            ),
        ]

        result = validator.validate_events(events)
        assert result.total_records == 2
        assert result.valid_records == 1
        assert result.invalid_records == 1
        assert not result.is_valid

    def test_validate_snapshots_batch(self, validator: DataValidator):
        """Test validating a batch of snapshots."""
        snapshots = [
            PriceSnapshot(
                event_id="test1",
                seat_zone=SeatZone.FLOOR_VIP,
                timestamp=datetime.now(UTC),
                price_min=100.0,
                days_to_event=30,
            ),
            PriceSnapshot(
                event_id="test2",
                seat_zone=SeatZone.BALCONY,
                timestamp=datetime.now(UTC),
                price_min=50.0,
                days_to_event=30,
            ),
        ]

        result = validator.validate_snapshots(snapshots)
        assert result.total_records == 2
        assert result.valid_records == 2
        assert result.is_valid
