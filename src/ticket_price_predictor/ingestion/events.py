"""Event metadata ingestion from Ticketmaster API."""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from ticket_price_predictor.api import TicketmasterClient
from ticket_price_predictor.config import Settings, get_settings
from ticket_price_predictor.schemas import EventMetadata, EventType
from ticket_price_predictor.storage import EventRepository
from ticket_price_predictor.validation import DataValidator

# Map classification names to event types
CLASSIFICATION_MAP: dict[str, EventType] = {
    "Music": EventType.CONCERT,
    "Sports": EventType.SPORTS,
    "Arts & Theatre": EventType.THEATER,
}


@dataclass
class IngestionResult:
    """Result of an ingestion operation."""

    events_fetched: int = 0
    events_saved: int = 0
    events_skipped: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Return True if ingestion completed without errors."""
        return len(self.errors) == 0


class EventIngestionService:
    """Service for ingesting event metadata from Ticketmaster."""

    def __init__(
        self,
        repository: EventRepository,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the service.

        Args:
            repository: Repository for storing events
            settings: Application settings
        """
        self._repository = repository
        self._settings = settings or get_settings()
        self._validator = DataValidator(allow_past_events=False)

    async def ingest_upcoming_events(
        self,
        days_ahead: int = 90,
        event_types: list[EventType] | None = None,
        cities: list[str] | None = None,
        max_events: int | None = None,
    ) -> IngestionResult:
        """Ingest upcoming events from Ticketmaster.

        Args:
            days_ahead: Number of days into the future to search
            event_types: Event types to include (None = all types)
            cities: Cities to filter by (None = all cities)
            max_events: Maximum number of events to fetch per category

        Returns:
            IngestionResult with statistics
        """
        result = IngestionResult()

        # Default to all event types
        if event_types is None:
            event_types = list(EventType)

        start_date = datetime.now(UTC)
        end_date = start_date + timedelta(days=days_ahead)

        async with TicketmasterClient(self._settings) as client:
            for event_type in event_types:
                classification = self._event_type_to_classification(event_type)

                try:
                    events = await self._fetch_events(
                        client,
                        classification_name=classification,
                        start_date=start_date,
                        end_date=end_date,
                        cities=cities,
                        max_events=max_events,
                    )

                    # Parse and validate events
                    parsed_events: list[EventMetadata] = []
                    for raw_event in events:
                        try:
                            metadata = client.parse_event_metadata(raw_event)
                            validation = self._validator.validate_event(metadata)

                            if validation.is_valid:
                                parsed_events.append(metadata)
                            else:
                                result.events_skipped += 1
                        except Exception as e:
                            result.errors.append(f"Failed to parse event: {e}")

                    result.events_fetched += len(events)

                    # Save to repository
                    saved = self._repository.save_events(parsed_events)
                    result.events_saved += saved
                    result.events_skipped += len(parsed_events) - saved

                except Exception as e:
                    result.errors.append(f"Failed to fetch {event_type.value} events: {e}")

        return result

    async def ingest_event(self, event_id: str) -> EventMetadata | None:
        """Ingest a single event by ID.

        Args:
            event_id: Ticketmaster event ID

        Returns:
            EventMetadata if successful, None otherwise
        """
        async with TicketmasterClient(self._settings) as client:
            try:
                raw_event = await client.get_event(event_id)
                metadata = client.parse_event_metadata(raw_event)

                validation = self._validator.validate_event(metadata)
                if not validation.is_valid:
                    return None

                self._repository.save_events([metadata])
                return metadata

            except Exception:
                return None

    async def _fetch_events(
        self,
        client: TicketmasterClient,
        classification_name: str,
        start_date: datetime,
        end_date: datetime,
        cities: list[str] | None = None,
        max_events: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch events with pagination.

        Args:
            client: Ticketmaster client
            classification_name: Event classification
            start_date: Search start date
            end_date: Search end date
            cities: Optional city filter
            max_events: Maximum events to fetch

        Returns:
            List of raw event dictionaries
        """
        all_events: list[dict[str, Any]] = []
        page = 0
        page_size = 200  # Max allowed by API

        while True:
            if cities:
                # Search each city separately
                for city in cities:
                    events = await client.search_events(
                        city=city,
                        classification_name=classification_name,
                        start_date=start_date,
                        end_date=end_date,
                        size=page_size,
                        page=page,
                    )
                    all_events.extend(events)

                    if max_events and len(all_events) >= max_events:
                        return all_events[:max_events]
            else:
                events = await client.search_events(
                    classification_name=classification_name,
                    start_date=start_date,
                    end_date=end_date,
                    size=page_size,
                    page=page,
                )
                all_events.extend(events)

            # Check if we should continue pagination
            if len(events) < page_size:
                break

            page += 1

            # Safety limit to prevent infinite loops
            if page >= 5:
                break

            if max_events and len(all_events) >= max_events:
                break

        return all_events[:max_events] if max_events else all_events

    def _event_type_to_classification(self, event_type: EventType) -> str:
        """Convert EventType to Ticketmaster classification name."""
        reverse_map = {v: k for k, v in CLASSIFICATION_MAP.items()}
        return reverse_map.get(event_type, "Music")
