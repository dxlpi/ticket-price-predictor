"""Price snapshot collection from Ticketmaster API."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ticket_price_predictor.api import TicketmasterClient
from ticket_price_predictor.config import Settings, get_settings
from ticket_price_predictor.normalization import SeatZoneMapper
from ticket_price_predictor.schemas import PriceSnapshot, SeatZone
from ticket_price_predictor.storage import EventRepository, SnapshotRepository
from ticket_price_predictor.validation import DataValidator


@dataclass
class CollectionResult:
    """Result of a snapshot collection operation."""

    events_processed: int = 0
    snapshots_created: int = 0
    snapshots_saved: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Return True if collection completed without errors."""
        return len(self.errors) == 0


class SnapshotCollector:
    """Collects price snapshots for tracked events."""

    def __init__(
        self,
        event_repository: EventRepository,
        snapshot_repository: SnapshotRepository,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the collector.

        Args:
            event_repository: Repository for reading events
            snapshot_repository: Repository for storing snapshots
            settings: Application settings
        """
        self._event_repo = event_repository
        self._snapshot_repo = snapshot_repository
        self._settings = settings or get_settings()
        self._zone_mapper = SeatZoneMapper()
        self._validator = DataValidator()

    async def collect_snapshots(
        self,
        event_ids: list[str] | None = None,
    ) -> CollectionResult:
        """Collect price snapshots for events.

        Args:
            event_ids: Specific event IDs to collect (None = all tracked events)

        Returns:
            CollectionResult with statistics
        """
        result = CollectionResult()

        # Get event IDs to process
        if event_ids is None:
            event_ids = self._event_repo.list_event_ids()

        if not event_ids:
            return result

        timestamp = datetime.now(UTC)

        async with TicketmasterClient(self._settings) as client:
            for event_id in event_ids:
                try:
                    # Fetch current event data
                    raw_event = await client.get_event(event_id)

                    # Get event metadata for days_to_event calculation
                    event = self._event_repo.get_event(event_id)
                    if event is None:
                        # Event not in our repo, parse from API response
                        event = client.parse_event_metadata(raw_event)

                    # Create snapshots for all zones
                    snapshots = self.create_snapshots_from_event(
                        raw_event,
                        event.event_datetime,
                        timestamp,
                    )

                    result.events_processed += 1
                    result.snapshots_created += len(snapshots)

                    # Validate and save
                    valid_snapshots = []
                    for snapshot in snapshots:
                        validation = self._validator.validate_snapshot(snapshot)
                        if validation.is_valid:
                            valid_snapshots.append(snapshot)

                    saved = self._snapshot_repo.save_snapshots(valid_snapshots)
                    result.snapshots_saved += saved

                except Exception as e:
                    result.errors.append(f"Failed to collect snapshot for {event_id}: {e}")

        return result

    def create_snapshots_from_event(
        self,
        raw_event: dict[str, Any],
        event_datetime: datetime,
        timestamp: datetime,
    ) -> list[PriceSnapshot]:
        """Create price snapshots from a raw event response.

        Since the Discovery API only provides priceRanges (min/max),
        we derive synthetic zone prices using the SeatZoneMapper.

        Args:
            raw_event: Raw event dictionary from API
            event_datetime: Event date/time for days_to_event calculation
            timestamp: Snapshot timestamp

        Returns:
            List of PriceSnapshot objects (one per zone)
        """
        event_id = raw_event.get("id", "")

        # Extract price range
        price_ranges = raw_event.get("priceRanges", [])
        if not price_ranges:
            # No pricing available
            return []

        # Use the first price range (typically "standard")
        price_range = price_ranges[0]
        price_min = float(price_range.get("min", 0))
        price_max = float(price_range.get("max", price_min))

        # Ensure event_datetime is timezone-aware
        if event_datetime.tzinfo is None:
            event_datetime = event_datetime.replace(tzinfo=UTC)

        # Calculate days to event
        days_to_event = (event_datetime - timestamp).days
        if days_to_event < 0:
            days_to_event = 0

        # Derive zone prices
        zone_prices = self._zone_mapper.map_price_range_to_zones(price_min, price_max)

        # Create a snapshot for each zone
        snapshots: list[PriceSnapshot] = []

        for zone, zone_price in zone_prices.items():
            snapshot = PriceSnapshot(
                event_id=event_id,
                seat_zone=zone,
                timestamp=timestamp,
                price_min=zone_price,
                price_avg=zone_price,  # Same as min for derived prices
                price_max=zone_price,
                inventory_remaining=None,  # Not available from Discovery API
                days_to_event=days_to_event,
                source="ticketmaster",
            )
            snapshots.append(snapshot)

        return snapshots

    def create_snapshot_for_zone(
        self,
        event_id: str,
        seat_zone: SeatZone,
        price: float,
        event_datetime: datetime,
        timestamp: datetime | None = None,
        inventory: int | None = None,
    ) -> PriceSnapshot:
        """Create a single snapshot for a specific zone.

        Args:
            event_id: Event ID
            seat_zone: Seat zone
            price: Zone price
            event_datetime: Event date/time
            timestamp: Snapshot timestamp (default: now)
            inventory: Optional inventory count

        Returns:
            PriceSnapshot object
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        if event_datetime.tzinfo is None:
            event_datetime = event_datetime.replace(tzinfo=UTC)

        days_to_event = max(0, (event_datetime - timestamp).days)

        return PriceSnapshot(
            event_id=event_id,
            seat_zone=seat_zone,
            timestamp=timestamp,
            price_min=price,
            price_avg=price,
            price_max=price,
            inventory_remaining=inventory,
            days_to_event=days_to_event,
            source="ticketmaster",
        )
