"""Bandsintown API client for artist tracker counts."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class BandsintownMetrics:
    """Metrics from Bandsintown API."""

    bandsintown_id: str
    tracker_count: int
    upcoming_event_count: int


class BandsintownPopularity:
    """Fetch artist popularity from Bandsintown API."""

    BASE_URL = "https://rest.bandsintown.com"

    def __init__(self, app_id: str) -> None:
        """Initialize Bandsintown client.

        Args:
            app_id: Bandsintown app ID
        """
        self.app_id = app_id
        self._available = bool(app_id)

    @property
    def available(self) -> bool:
        """Check if Bandsintown client is available."""
        return self._available

    def get_artist_metrics(self, artist_name: str) -> BandsintownMetrics | None:
        """Fetch artist metrics from Bandsintown.

        Args:
            artist_name: Artist name to search for

        Returns:
            BandsintownMetrics if found, None otherwise
        """
        if not self._available:
            return None

        try:
            # URL encode the artist name
            encoded_name = artist_name.replace(" ", "%20")

            with httpx.Client(timeout=10.0) as client:
                resp = client.get(
                    f"{self.BASE_URL}/artists/{encoded_name}",
                    params={"app_id": self.app_id},
                )

                if resp.status_code == 404:
                    logger.debug(f"No Bandsintown results for artist: {artist_name}")
                    return None

                resp.raise_for_status()
                data = resp.json()

            # Handle error responses
            if isinstance(data, dict) and data.get("error"):
                logger.debug(f"Bandsintown error for {artist_name}: {data.get('error')}")
                return None

            return BandsintownMetrics(
                bandsintown_id=str(data.get("id", "")),
                tracker_count=data.get("tracker_count", 0),
                upcoming_event_count=data.get("upcoming_event_count", 0),
            )
        except Exception as e:
            logger.warning(f"Bandsintown API error for {artist_name}: {e}")
            return None
