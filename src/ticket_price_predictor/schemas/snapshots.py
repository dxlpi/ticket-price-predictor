"""Data schemas for event metadata and price snapshots."""

from datetime import datetime
from enum import Enum
from typing import Self

import pyarrow as pa
from pydantic import BaseModel, Field, model_validator


class EventType(str, Enum):
    """Type of event."""

    CONCERT = "concert"
    SPORTS = "sports"
    THEATER = "theater"


class SeatZone(str, Enum):
    """Standardized seat zone categories for cross-venue learning."""

    FLOOR_VIP = "floor_vip"
    LOWER_TIER = "lower_tier"
    UPPER_TIER = "upper_tier"
    BALCONY = "balcony"


class EventMetadata(BaseModel):
    """Metadata for an event from Ticketmaster."""

    event_id: str = Field(..., description="Unique event identifier")
    event_type: EventType = Field(..., description="Type of event")
    event_datetime: datetime = Field(..., description="Event date and time")
    artist_or_team: str = Field(..., description="Artist, team, or show name")
    venue_id: str = Field(..., description="Venue identifier")
    venue_name: str = Field(..., description="Venue name")
    city: str = Field(..., description="City where venue is located")
    country: str = Field(default="US", description="Country code")
    venue_capacity: int | None = Field(default=None, description="Venue seating capacity")

    @classmethod
    def parquet_schema(cls) -> pa.Schema:
        """Return PyArrow schema for Parquet storage."""
        return pa.schema(
            [
                pa.field("event_id", pa.string(), nullable=False),
                pa.field("event_type", pa.string(), nullable=False),
                pa.field("event_datetime", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("artist_or_team", pa.string(), nullable=False),
                pa.field("venue_id", pa.string(), nullable=False),
                pa.field("venue_name", pa.string(), nullable=False),
                pa.field("city", pa.string(), nullable=False),
                pa.field("country", pa.string(), nullable=False),
                pa.field("venue_capacity", pa.int32(), nullable=True),
            ]
        )


class PriceSnapshot(BaseModel):
    """A point-in-time snapshot of ticket prices for a seat zone."""

    event_id: str = Field(..., description="Event identifier")
    seat_zone: SeatZone = Field(..., description="Normalized seat zone")
    timestamp: datetime = Field(..., description="Snapshot capture time")
    price_min: float = Field(..., ge=0, description="Minimum observed price in zone")
    price_avg: float | None = Field(default=None, ge=0, description="Average price in zone")
    price_max: float | None = Field(default=None, ge=0, description="Maximum price in zone")
    inventory_remaining: int | None = Field(
        default=None, ge=0, description="Tickets remaining in zone"
    )
    days_to_event: int = Field(..., description="Days until event date")

    @model_validator(mode="after")
    def validate_price_ordering(self) -> Self:
        """Ensure price_min <= price_avg <= price_max when all are present."""
        if self.price_avg is not None and self.price_avg < self.price_min:
            raise ValueError("price_avg must be >= price_min")
        if self.price_max is not None and self.price_max < self.price_min:
            raise ValueError("price_max must be >= price_min")
        if (
            self.price_avg is not None
            and self.price_max is not None
            and self.price_avg > self.price_max
        ):
            raise ValueError("price_avg must be <= price_max")
        return self

    @classmethod
    def parquet_schema(cls) -> pa.Schema:
        """Return PyArrow schema for Parquet storage."""
        return pa.schema(
            [
                pa.field("event_id", pa.string(), nullable=False),
                pa.field("seat_zone", pa.string(), nullable=False),
                pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("price_min", pa.float64(), nullable=False),
                pa.field("price_avg", pa.float64(), nullable=True),
                pa.field("price_max", pa.float64(), nullable=True),
                pa.field("inventory_remaining", pa.int32(), nullable=True),
                pa.field("days_to_event", pa.int32(), nullable=False),
            ]
        )
