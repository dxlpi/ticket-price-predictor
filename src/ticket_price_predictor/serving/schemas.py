"""Pydantic DTOs for the serving layer.

These are independent of the domain models in `schemas/` and `ml/schemas`; route
handlers convert between them via `model_validate(domain.model_dump())`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ticket_price_predictor.schemas import EventType, SeatZone


class EventSummary(BaseModel):
    """Compact event view returned by the search endpoint."""

    event_id: str
    artist_or_team: str
    venue_name: str
    city: str
    event_datetime: datetime
    event_type: EventType


class EventDetail(EventSummary):
    """Full event view returned by the per-event endpoint."""

    country: str
    venue_capacity: int | None = None


class PredictRequest(BaseModel):
    """Request body for `POST /api/predict`.

    Either `seat_zone` or `section` is required; when both are set, `section`
    wins (see route docstring for the AC5 precedence rule).
    """

    event_id: str
    seat_zone: SeatZone | None = None
    section: str | None = None
    row: str = "10"  # matches PricePredictor.predict default
    quantity: int = Field(default=2, ge=1, le=8)
    as_of_date: datetime | None = None  # reference date for days_to_event; None → now()

    @field_validator("section", mode="before")
    @classmethod
    def _strip_section(cls, v: str | None) -> str | None:
        # Treat whitespace-only as absent to avoid silently dispatching the predictor
        # with a meaningless section that falls back to UPPER_TIER.
        if isinstance(v, str):
            v = v.strip()
            return v if v else None
        return v

    @model_validator(mode="after")
    def _zone_or_section(self) -> PredictRequest:
        if self.seat_zone is None and not self.section:
            raise ValueError("either seat_zone or section is required")
        return self


class PredictResponse(BaseModel):
    """Response for `POST /api/predict` — every field of `PricePrediction`."""

    event_id: str
    seat_zone: str
    target_days_to_event: int
    predicted_price: float
    price_lower_bound: float
    price_upper_bound: float
    confidence_score: float
    predicted_direction: Literal["UP", "DOWN", "STABLE"]
    direction_probability: float
    model_version: str
    prediction_timestamp: datetime
