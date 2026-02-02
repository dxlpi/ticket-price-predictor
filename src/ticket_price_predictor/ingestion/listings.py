"""Listing collection from ticket marketplaces."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ticket_price_predictor.schemas import (
    ScrapedEvent,
    TicketListing,
    create_listing_from_scraped,
)
from ticket_price_predictor.scrapers import (
    PlaywrightStubHubScraper,
    StubHubScraper,
    VividSeatsScraper,
)
from ticket_price_predictor.storage import ListingRepository


class DataSource(str, Enum):
    """Supported ticket marketplace data sources."""

    STUBHUB = "stubhub"
    VIVIDSEATS = "vividseats"


@dataclass
class ListingCollectionResult:
    """Result of a listing collection operation."""

    events_found: int = 0
    events_processed: int = 0
    listings_collected: int = 0
    listings_saved: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Return True if collection completed without errors."""
        return len(self.errors) == 0


class ListingCollector:
    """Collects real seat-level listings from ticket marketplaces."""

    def __init__(
        self,
        data_dir: Path,
        delay_seconds: float = 2.0,
        use_browser: bool = False,
        source: DataSource = DataSource.VIVIDSEATS,
    ) -> None:
        """Initialize the collector.

        Args:
            data_dir: Base directory for data storage
            delay_seconds: Delay between scraper requests
            use_browser: Use Playwright browser automation (slower but bypasses anti-bot)
            source: Data source to use (vividseats recommended, stubhub blocked)
        """
        self._data_dir = data_dir
        self._delay = delay_seconds
        self._use_browser = use_browser
        self._source = source
        self._repository = ListingRepository(data_dir)

    def _get_scraper_class(self) -> type[Any]:
        """Get the appropriate scraper class based on configuration."""
        if self._source == DataSource.VIVIDSEATS:
            return VividSeatsScraper
        elif self._use_browser:
            return PlaywrightStubHubScraper
        else:
            return StubHubScraper

    async def collect_for_artist(
        self,
        artist_name: str,
        max_events: int = 10,
        max_listings_per_event: int = 500,
    ) -> ListingCollectionResult:
        """Collect listings for all upcoming events by an artist.

        Args:
            artist_name: Artist name to search for
            max_events: Maximum number of events to process
            max_listings_per_event: Maximum listings per event

        Returns:
            ListingCollectionResult with statistics
        """
        result = ListingCollectionResult()
        timestamp = datetime.now(UTC)

        scraper_class = self._get_scraper_class()
        async with scraper_class(delay_seconds=self._delay) as scraper:
            # Search for events
            try:
                events = await scraper.search_events(artist_name, max_results=max_events)
                result.events_found = len(events)
            except Exception as e:
                result.errors.append(f"Failed to search for '{artist_name}': {e}")
                return result

            if not events:
                return result

            # Process each event
            for event in events:
                try:
                    listings = await self._collect_event_listings(
                        scraper,
                        event,
                        max_listings_per_event,
                        timestamp,
                    )

                    result.events_processed += 1
                    result.listings_collected += len(listings)

                    # Save listings
                    if listings:
                        saved = self._repository.save_listings(listings)
                        result.listings_saved += saved

                except Exception as e:
                    result.errors.append(f"Failed to collect listings for {event.event_name}: {e}")

        return result

    async def collect_for_event_url(
        self,
        event_url: str,
        event_name: str = "Unknown Event",
        artist_name: str = "Unknown Artist",
        venue_name: str = "Unknown Venue",
        city: str = "Unknown",
        event_datetime: datetime | None = None,
        max_listings: int = 500,
    ) -> ListingCollectionResult:
        """Collect listings for a specific event URL.

        Args:
            event_url: StubHub event page URL
            event_name: Event name (for metadata)
            artist_name: Artist/team name
            venue_name: Venue name
            city: City name
            event_datetime: Event date/time
            max_listings: Maximum listings to collect

        Returns:
            ListingCollectionResult with statistics
        """
        result = ListingCollectionResult()
        timestamp = datetime.now(UTC)

        # Create event object from provided info
        event = ScrapedEvent(
            stubhub_event_id=event_url.split("/")[-1] if "/" in event_url else event_url,
            event_name=event_name,
            artist_or_team=artist_name,
            venue_name=venue_name,
            city=city,
            event_datetime=event_datetime or datetime.now(UTC),
            event_url=event_url,
        )

        scraper_class = self._get_scraper_class()
        async with scraper_class(delay_seconds=self._delay) as scraper:
            try:
                listings = await self._collect_event_listings(
                    scraper,
                    event,
                    max_listings,
                    timestamp,
                )

                result.events_found = 1
                result.events_processed = 1
                result.listings_collected = len(listings)

                if listings:
                    saved = self._repository.save_listings(listings)
                    result.listings_saved = saved

            except Exception as e:
                result.errors.append(f"Failed to collect listings: {e}")

        return result

    async def _collect_event_listings(
        self,
        scraper: StubHubScraper | PlaywrightStubHubScraper | VividSeatsScraper,
        event: ScrapedEvent,
        max_listings: int,
        timestamp: datetime,
    ) -> list[TicketListing]:
        """Collect and convert listings for a single event."""
        # Fetch raw listings
        scraped_listings = await scraper.get_event_listings(
            event.event_url,
            max_listings=max_listings,
        )

        # Convert to TicketListing objects
        listings: list[TicketListing] = []
        for scraped in scraped_listings:
            listing = create_listing_from_scraped(scraped, event, timestamp)
            listings.append(listing)

        return listings
