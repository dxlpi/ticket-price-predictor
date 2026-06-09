"""DTO validator tests for the serving layer (AC14).

Pure schema-level tests — no FastAPI, no `TestClient`. Covers `PredictRequest`
field/model validators (mutual presence rule, whitespace stripping, quantity
bounds, enum membership, default values) plus `EventSummary` /`EventDetail`
round-trips from `EventMetadata.model_dump()`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ticket_price_predictor.schemas import EventMetadata, EventType, SeatZone
from ticket_price_predictor.serving.schemas import (
    EventDetail,
    EventSummary,
    PredictRequest,
)


def _sample_event_metadata() -> EventMetadata:
    """Build a representative `EventMetadata` for round-trip tests."""
    return EventMetadata(
        event_id="evt_001",
        event_type=EventType.CONCERT,
        event_datetime=datetime(2026, 6, 15, 20, 0, tzinfo=UTC),
        artist_or_team="Bruno Mars",
        venue_id="venue_001",
        venue_name="MGM Grand Garden Arena",
        city="Las Vegas",
        country="US",
        venue_capacity=17000,
    )


# ---------------------------------------------------------------------------
# Mutual presence rule: seat_zone or section is required
# ---------------------------------------------------------------------------


def test_predict_request_requires_zone_or_section() -> None:
    with pytest.raises(ValidationError, match="either seat_zone or section is required"):
        PredictRequest(event_id="x")


def test_predict_request_accepts_seat_zone_only() -> None:
    req = PredictRequest(event_id="x", seat_zone=SeatZone.LOWER_TIER)
    assert req.seat_zone is SeatZone.LOWER_TIER
    assert req.section is None


def test_predict_request_accepts_section_only() -> None:
    req = PredictRequest(event_id="x", section="Lower Level 100")
    assert req.section == "Lower Level 100"
    assert req.seat_zone is None


# ---------------------------------------------------------------------------
# Whitespace handling on the `section` field
# ---------------------------------------------------------------------------


def test_predict_request_whitespace_only_section_is_treated_as_absent() -> None:
    # `_strip_section` normalises whitespace-only input to None, then the
    # model_validator catches the missing-both case.
    with pytest.raises(ValidationError, match="either seat_zone or section is required"):
        PredictRequest(event_id="x", section="   ")


def test_predict_request_strips_section_whitespace() -> None:
    req = PredictRequest(event_id="x", section="  Lower Level 100  ")
    assert req.section == "Lower Level 100"


# ---------------------------------------------------------------------------
# `quantity` bounds (1..8) + default value
# ---------------------------------------------------------------------------


def test_predict_request_quantity_default_is_two() -> None:
    req = PredictRequest(event_id="x", seat_zone=SeatZone.FLOOR_VIP)
    assert req.quantity == 2


def test_predict_request_quantity_zero_rejected() -> None:
    with pytest.raises(ValidationError, match="quantity"):
        PredictRequest(event_id="x", seat_zone=SeatZone.FLOOR_VIP, quantity=0)


def test_predict_request_quantity_nine_rejected() -> None:
    with pytest.raises(ValidationError, match="quantity"):
        PredictRequest(event_id="x", seat_zone=SeatZone.FLOOR_VIP, quantity=9)


# ---------------------------------------------------------------------------
# Enum membership for `seat_zone`
# ---------------------------------------------------------------------------


def test_predict_request_invalid_seat_zone_rejected() -> None:
    with pytest.raises(ValidationError, match="seat_zone"):
        PredictRequest(event_id="x", seat_zone="invalid_zone")


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


def test_predict_request_defaults() -> None:
    req = PredictRequest(event_id="x", seat_zone=SeatZone.FLOOR_VIP)
    assert req.row == "10"
    assert req.quantity == 2


# ---------------------------------------------------------------------------
# EventSummary / EventDetail round-trips from EventMetadata.model_dump()
# ---------------------------------------------------------------------------


def test_event_summary_round_trips_from_event_metadata() -> None:
    event = _sample_event_metadata()
    summary = EventSummary.model_validate(event.model_dump())

    assert summary.event_id == "evt_001"
    assert summary.artist_or_team == "Bruno Mars"
    assert summary.venue_name == "MGM Grand Garden Arena"
    assert summary.city == "Las Vegas"
    assert summary.event_datetime == datetime(2026, 6, 15, 20, 0, tzinfo=UTC)
    assert summary.event_type is EventType.CONCERT


def test_event_detail_populates_country_and_capacity() -> None:
    event = _sample_event_metadata()
    detail = EventDetail.model_validate(event.model_dump())

    assert detail.country == "US"
    assert detail.venue_capacity == 17000
    # Inherited summary fields still populate.
    assert detail.event_id == "evt_001"
    assert detail.artist_or_team == "Bruno Mars"
