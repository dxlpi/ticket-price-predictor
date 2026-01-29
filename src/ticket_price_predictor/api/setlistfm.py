"""Setlist.fm API client for historical concert data."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import httpx

from ticket_price_predictor.config import Settings


@dataclass
class HistoricalConcert:
    """A historical concert from setlist.fm."""

    event_id: str
    artist_name: str
    artist_mbid: str  # MusicBrainz ID
    event_date: datetime
    venue_name: str
    city: str
    country: str
    tour_name: str | None = None
    setlist_url: str | None = None


class SetlistFMClient:
    """Client for the setlist.fm API.

    API docs: https://api.setlist.fm/docs/1.0/index.html

    Note: Free for non-commercial use only.
    """

    BASE_URL = "https://api.setlist.fm/rest/1.0"

    def __init__(self, api_key: str | None = None, settings: Settings | None = None) -> None:
        """Initialize the client.

        Args:
            api_key: Setlist.fm API key (get one at https://www.setlist.fm/settings/api)
            settings: Application settings (will look for SETLISTFM_API_KEY)
        """
        self._api_key: str | None
        if api_key:
            self._api_key = api_key
        elif settings and hasattr(settings, "setlistfm_api_key"):
            self._api_key = settings.setlistfm_api_key
        else:
            self._api_key = None

        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "SetlistFMClient":
        """Enter async context."""
        if not self._api_key:
            raise ValueError(
                "Setlist.fm API key required. Get one at https://www.setlist.fm/settings/api"
            )

        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "x-api-key": self._api_key,
                "Accept": "application/json",
            },
            timeout=30.0,
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def search_artists(self, artist_name: str) -> list[dict[str, Any]]:
        """Search for artists by name.

        Args:
            artist_name: Artist name to search for

        Returns:
            List of matching artists with mbid, name, etc.
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use async with.")

        response = await self._client.get(
            "/search/artists",
            params={"artistName": artist_name, "sort": "relevance"},
        )
        response.raise_for_status()

        data = response.json()
        return cast(list[dict[str, Any]], data.get("artist", []))

    async def get_artist_setlists(
        self,
        mbid: str,
        page: int = 1,
    ) -> tuple[list[dict[str, Any]], int]:
        """Get all setlists (concerts) for an artist.

        Args:
            mbid: MusicBrainz ID of the artist
            page: Page number (20 results per page)

        Returns:
            Tuple of (list of setlists, total number of setlists)
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use async with.")

        response = await self._client.get(
            f"/artist/{mbid}/setlists",
            params={"p": page},
        )
        response.raise_for_status()

        data = response.json()
        return data.get("setlist", []), data.get("total", 0)

    async def get_artist_concerts(
        self,
        artist_name: str,
        max_concerts: int | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> list[HistoricalConcert]:
        """Get historical concerts for an artist.

        Args:
            artist_name: Artist name to search for
            max_concerts: Maximum number of concerts to return
            year_from: Only include concerts from this year onwards
            year_to: Only include concerts up to this year

        Returns:
            List of HistoricalConcert objects
        """
        # Find the artist
        artists = await self.search_artists(artist_name)
        if not artists:
            return []

        # Use the first (most relevant) match
        artist = artists[0]
        mbid = artist.get("mbid", "")
        resolved_name = artist.get("name", artist_name)

        concerts: list[HistoricalConcert] = []
        page = 1

        while True:
            setlists, total = await self.get_artist_setlists(mbid, page)

            if not setlists:
                break

            for setlist in setlists:
                concert = self._parse_setlist(setlist, resolved_name, mbid)
                if concert is None:
                    continue

                # Apply year filters
                if year_from and concert.event_date.year < year_from:
                    continue
                if year_to and concert.event_date.year > year_to:
                    continue

                concerts.append(concert)

                if max_concerts and len(concerts) >= max_concerts:
                    return concerts

            # Check if we need more pages
            if len(setlists) < 20:  # Less than full page means no more data
                break

            page += 1

            # Safety limit
            if page > 50:
                break

        return concerts

    def _parse_setlist(
        self,
        setlist: dict[str, Any],
        artist_name: str,
        mbid: str,
    ) -> HistoricalConcert | None:
        """Parse a setlist response into a HistoricalConcert."""
        try:
            # Parse date (format: DD-MM-YYYY)
            date_str = setlist.get("eventDate", "")
            if not date_str:
                return None

            day, month, year = date_str.split("-")
            event_date = datetime(int(year), int(month), int(day))

            # Get venue info
            venue = setlist.get("venue", {})
            venue_name = venue.get("name", "Unknown Venue")

            city_info = venue.get("city", {})
            city = city_info.get("name", "Unknown City")

            country_info = city_info.get("country", {})
            country = country_info.get("code", "US")

            # Get tour name if available
            tour = setlist.get("tour", {})
            tour_name = tour.get("name") if tour else None

            return HistoricalConcert(
                event_id=setlist.get("id", ""),
                artist_name=artist_name,
                artist_mbid=mbid,
                event_date=event_date,
                venue_name=venue_name,
                city=city,
                country=country,
                tour_name=tour_name,
                setlist_url=setlist.get("url"),
            )
        except (ValueError, KeyError):
            return None
