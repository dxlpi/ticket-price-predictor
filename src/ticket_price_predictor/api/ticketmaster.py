"""Ticketmaster Discovery API client for event metadata."""

import asyncio
from datetime import datetime
from typing import Any

import httpx

from ticket_price_predictor.config import Settings, get_settings
from ticket_price_predictor.schemas import EventMetadata, EventType


class TicketmasterError(Exception):
    """Base exception for Ticketmaster API errors."""


class RateLimitError(TicketmasterError):
    """Raised when rate limit is exceeded."""


class TicketmasterClient:
    """Async client for Ticketmaster Discovery API.

    Handles event search and metadata retrieval with rate limiting awareness.
    Free tier allows 5 requests/second.
    """

    # Rate limit: 5 requests per second for free tier
    RATE_LIMIT_REQUESTS = 5
    RATE_LIMIT_WINDOW = 1.0  # seconds

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the client.

        Args:
            settings: Application settings. Uses default settings if not provided.
        """
        self._settings = settings or get_settings()
        self._client: httpx.AsyncClient | None = None
        self._request_times: list[float] = []

    @property
    def api_key(self) -> str:
        """Get the API key from settings."""
        return self._settings.ticketmaster_api_key

    @property
    def base_url(self) -> str:
        """Get the base URL from settings."""
        return self._settings.ticketmaster_base_url

    async def __aenter__(self) -> "TicketmasterClient":
        """Enter async context and create HTTP client."""
        self._client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit async context and close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _rate_limit(self) -> None:
        """Enforce rate limiting by waiting if necessary."""
        now = asyncio.get_event_loop().time()

        # Remove old request times outside the window
        self._request_times = [t for t in self._request_times if now - t < self.RATE_LIMIT_WINDOW]

        # If at limit, wait until oldest request exits the window
        if len(self._request_times) >= self.RATE_LIMIT_REQUESTS:
            sleep_time = self.RATE_LIMIT_WINDOW - (now - self._request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self._request_times.append(asyncio.get_event_loop().time())

    async def _request(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make a rate-limited request to the API.

        Args:
            endpoint: API endpoint path (e.g., "/events")
            params: Query parameters

        Returns:
            JSON response as dictionary

        Raises:
            TicketmasterError: If API request fails
            RateLimitError: If rate limit is exceeded
        """
        if not self._client:
            raise TicketmasterError("Client not initialized. Use async context manager.")

        if not self.api_key:
            raise TicketmasterError("TICKETMASTER_API_KEY not configured")

        await self._rate_limit()

        url = f"{self.base_url}{endpoint}"
        request_params = {"apikey": self.api_key, **(params or {})}

        response = await self._client.get(url, params=request_params)

        if response.status_code == 429:
            raise RateLimitError("Rate limit exceeded. Please wait before retrying.")

        if response.status_code != 200:
            raise TicketmasterError(f"API request failed: {response.status_code} - {response.text}")

        return response.json()  # type: ignore[no-any-return]

    async def search_events(
        self,
        keyword: str | None = None,
        city: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        classification_name: str | None = None,
        size: int = 20,
        page: int = 0,
    ) -> list[dict[str, Any]]:
        """Search for events matching criteria.

        Args:
            keyword: Search keyword (artist, team, event name)
            city: City name to filter by
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            classification_name: Event classification (Music, Sports, Arts & Theatre)
            size: Number of results per page (max 200)
            page: Page number (0-indexed)

        Returns:
            List of raw event dictionaries from API
        """
        params: dict[str, Any] = {
            "countryCode": "US",
            "size": min(size, 200),
            "page": page,
        }

        if keyword:
            params["keyword"] = keyword
        if city:
            params["city"] = city
        if start_date:
            params["startDateTime"] = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        if end_date:
            params["endDateTime"] = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        if classification_name:
            params["classificationName"] = classification_name

        data = await self._request("/events.json", params)

        # Handle empty results
        if "_embedded" not in data:
            return []

        return data["_embedded"].get("events", [])  # type: ignore[no-any-return]

    async def get_event(self, event_id: str) -> dict[str, Any]:
        """Get detailed information for a single event.

        Args:
            event_id: Ticketmaster event ID

        Returns:
            Raw event dictionary from API
        """
        return await self._request(f"/events/{event_id}.json")

    def parse_event_metadata(self, raw_event: dict[str, Any]) -> EventMetadata:
        """Parse raw API response into EventMetadata schema.

        Args:
            raw_event: Raw event dictionary from API

        Returns:
            Validated EventMetadata object
        """
        # Determine event type from classifications
        event_type = EventType.CONCERT  # default
        classifications = raw_event.get("classifications", [])
        if classifications:
            segment = classifications[0].get("segment", {}).get("name", "").lower()
            if "sport" in segment:
                event_type = EventType.SPORTS
            elif "art" in segment or "theatre" in segment or "theater" in segment:
                event_type = EventType.THEATER

        # Extract venue info
        venues = raw_event.get("_embedded", {}).get("venues", [])
        venue = venues[0] if venues else {}
        venue_id = venue.get("id", "unknown")
        venue_name = venue.get("name", "Unknown Venue")
        city = venue.get("city", {}).get("name", "Unknown")
        country = venue.get("country", {}).get("countryCode", "US")

        # Parse datetime
        dates = raw_event.get("dates", {})
        start = dates.get("start", {})
        date_str = start.get("dateTime") or start.get("localDate", "")
        if "T" in date_str:
            event_datetime = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        else:
            event_datetime = datetime.fromisoformat(f"{date_str}T00:00:00+00:00")

        # Extract artist/team name
        attractions = raw_event.get("_embedded", {}).get("attractions", [])
        if attractions:
            artist_or_team = attractions[0].get("name", raw_event.get("name", "Unknown"))
        else:
            artist_or_team = raw_event.get("name", "Unknown")

        return EventMetadata(
            event_id=raw_event["id"],
            event_type=event_type,
            event_datetime=event_datetime,
            artist_or_team=artist_or_team,
            venue_id=venue_id,
            venue_name=venue_name,
            city=city,
            country=country,
            venue_capacity=None,  # Not provided by Discovery API
        )
