"""Schema for individual ticket listings with seat-level detail."""

from datetime import UTC, datetime

import pyarrow as pa
from pydantic import BaseModel, computed_field


class TicketListing(BaseModel):
    """Individual ticket listing from a resale marketplace.

    Contains seat-level detail including section, row, and seat numbers,
    along with both face value (original price) and current listing price.
    """

    # Identifiers
    listing_id: str
    event_id: str
    source: str = "stubhub"  # Data source (stubhub, seatgeek, etc.)
    timestamp: datetime

    # Event info (denormalized for convenience)
    event_name: str
    artist_or_team: str
    venue_name: str
    city: str
    event_datetime: datetime

    # Seating
    section: str
    row: str
    seat_from: str | None = None
    seat_to: str | None = None
    quantity: int

    # Pricing
    face_value: float | None = None  # Original ticket price
    listing_price: float  # Current asking price per ticket
    total_price: float  # Total including fees
    currency: str = "USD"

    # Time-based
    days_to_event: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def markup_ratio(self) -> float | None:
        """Calculate markup ratio (listing_price / face_value)."""
        if self.face_value and self.face_value > 0:
            return round(self.listing_price / self.face_value, 2)
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def seat_description(self) -> str:
        """Human-readable seat description."""
        if self.seat_from and self.seat_to:
            if self.seat_from == self.seat_to:
                return f"{self.section}, Row {self.row}, Seat {self.seat_from}"
            return f"{self.section}, Row {self.row}, Seats {self.seat_from}-{self.seat_to}"
        return f"{self.section}, Row {self.row}"

    @classmethod
    def parquet_schema(cls) -> pa.Schema:
        """Return PyArrow schema for Parquet storage."""
        return pa.schema(
            [
                ("listing_id", pa.string()),
                ("event_id", pa.string()),
                ("source", pa.string()),
                ("timestamp", pa.timestamp("us", tz="UTC")),
                ("event_name", pa.string()),
                ("artist_or_team", pa.string()),
                ("venue_name", pa.string()),
                ("city", pa.string()),
                ("event_datetime", pa.timestamp("us", tz="UTC")),
                ("section", pa.string()),
                ("row", pa.string()),
                ("seat_from", pa.string()),
                ("seat_to", pa.string()),
                ("quantity", pa.int32()),
                ("face_value", pa.float64()),
                ("listing_price", pa.float64()),
                ("total_price", pa.float64()),
                ("currency", pa.string()),
                ("days_to_event", pa.int32()),
                ("markup_ratio", pa.float64()),
                ("seat_description", pa.string()),
            ]
        )


class ScrapedEvent(BaseModel):
    """Event found from scraping search results."""

    stubhub_event_id: str
    event_name: str
    artist_or_team: str
    venue_name: str
    city: str
    event_datetime: datetime
    event_url: str
    min_price: float | None = None
    ticket_count: int | None = None


class ScrapedListing(BaseModel):
    """Raw listing data extracted from scraping."""

    listing_id: str
    section: str
    row: str
    seat_from: str | None = None
    seat_to: str | None = None
    quantity: int
    price_per_ticket: float
    total_price: float
    face_value: float | None = None


def create_listing_from_scraped(
    scraped: ScrapedListing,
    event: ScrapedEvent,
    timestamp: datetime | None = None,
    source: str = "vividseats",
) -> TicketListing:
    """Create a TicketListing from scraped data.

    Args:
        scraped: Raw scraped listing data
        event: Event information
        timestamp: Collection timestamp (default: now)
        source: Data source identifier (default: vividseats)

    Returns:
        TicketListing with all fields populated
    """
    if timestamp is None:
        timestamp = datetime.now(UTC)

    # Ensure event_datetime is timezone-aware
    event_dt = event.event_datetime
    if event_dt.tzinfo is None:
        event_dt = event_dt.replace(tzinfo=UTC)

    days_to_event = max(0, (event_dt - timestamp).days)

    return TicketListing(
        listing_id=scraped.listing_id,
        event_id=event.stubhub_event_id,
        source=source,
        timestamp=timestamp,
        event_name=event.event_name,
        artist_or_team=event.artist_or_team,
        venue_name=event.venue_name,
        city=event.city,
        event_datetime=event_dt,
        section=scraped.section,
        row=scraped.row,
        seat_from=scraped.seat_from,
        seat_to=scraped.seat_to,
        quantity=scraped.quantity,
        face_value=scraped.face_value,
        listing_price=scraped.price_per_ticket,
        total_price=scraped.total_price,
        days_to_event=days_to_event,
    )
