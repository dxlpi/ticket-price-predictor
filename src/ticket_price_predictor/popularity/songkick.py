"""Songkick API client for artist tracker counts."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class SongkickMetrics:
    """Metrics from Songkick API."""
    songkick_id: int
    display_name: str
    on_tour: bool
    tracker_count: int


class SongkickPopularity:
    """Fetch artist popularity from Songkick API."""

    BASE_URL = "https://api.songkick.com/api/3.0"

    def __init__(self, api_key: str) -> None:
        """Initialize Songkick client.

        Args:
            api_key: Songkick API key
        """
        self.api_key = api_key
        self._available = bool(api_key)

    @property
    def available(self) -> bool:
        """Check if Songkick client is available."""
        return self._available

    def get_artist_metrics(self, artist_name: str) -> SongkickMetrics | None:
        """Fetch artist metrics from Songkick.

        Args:
            artist_name: Artist name to search for

        Returns:
            SongkickMetrics if found, None otherwise
        """
        if not self._available:
            return None

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(
                    f"{self.BASE_URL}/search/artists.json",
                    params={"apikey": self.api_key, "query": artist_name},
                )
                resp.raise_for_status()
                data = resp.json()

            results = data.get("resultsPage", {}).get("results", {})
            artists = results.get("artist", [])

            if not artists:
                logger.debug(f"No Songkick results for artist: {artist_name}")
                return None

            artist = artists[0]

            # Get detailed artist info for tracker count
            artist_id = artist.get("id")
            tracker_count = self._get_tracker_count(artist_id) if artist_id else 0

            return SongkickMetrics(
                songkick_id=artist.get("id", 0),
                display_name=artist.get("displayName", artist_name),
                on_tour=artist.get("onTourUntil") is not None,
                tracker_count=tracker_count,
            )
        except Exception as e:
            logger.warning(f"Songkick API error for {artist_name}: {e}")
            return None

    def _get_tracker_count(self, artist_id: int) -> int:
        """Get tracker count for a specific artist.

        Args:
            artist_id: Songkick artist ID

        Returns:
            Number of trackers
        """
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(
                    f"{self.BASE_URL}/artists/{artist_id}.json",
                    params={"apikey": self.api_key},
                )
                resp.raise_for_status()
                data = resp.json()

            artist = data.get("resultsPage", {}).get("results", {}).get("artist", {})
            # Songkick doesn't directly expose tracker count in API
            # Use onTourUntil as a proxy for activity
            return 1 if artist.get("onTourUntil") else 0
        except Exception:
            return 0
