"""Last.fm API client for artist popularity metrics."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class LastfmMetrics:
    """Metrics from Last.fm API."""

    name: str
    listener_count: int
    play_count: int
    tags: list[str]


class LastfmPopularity:
    """Fetch artist popularity from Last.fm API."""

    BASE_URL = "https://ws.audioscrobbler.com/2.0/"

    def __init__(self, api_key: str) -> None:
        """Initialize Last.fm client.

        Args:
            api_key: Last.fm API key
        """
        self.api_key = api_key
        self._available = bool(api_key)

    @property
    def available(self) -> bool:
        """Check if Last.fm client is available."""
        return self._available

    def get_artist_metrics(self, artist_name: str) -> LastfmMetrics | None:
        """Fetch artist metrics from Last.fm.

        Args:
            artist_name: Artist name to search for

        Returns:
            LastfmMetrics if found, None otherwise
        """
        if not self._available:
            return None

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(
                    self.BASE_URL,
                    params={
                        "method": "artist.getinfo",
                        "artist": artist_name,
                        "api_key": self.api_key,
                        "format": "json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            artist = data.get("artist")
            if not artist:
                logger.debug(f"No Last.fm results for artist: {artist_name}")
                return None

            # Check for error response
            if "error" in data:
                logger.debug(f"Last.fm error for {artist_name}: {data.get('message', '')}")
                return None

            # Extract tags
            tags_data = artist.get("tags", {}).get("tag", [])
            tags = [t.get("name", "") for t in tags_data if isinstance(t, dict)]

            stats = artist.get("stats", {})

            return LastfmMetrics(
                name=artist.get("name", artist_name),
                listener_count=int(stats.get("listeners", 0)),
                play_count=int(stats.get("playcount", 0)),
                tags=tags,
            )
        except Exception as e:
            logger.warning(f"Last.fm API error for {artist_name}: {e}")
            return None

    def get_top_artists(self, limit: int = 50) -> list[str]:
        """Fetch top global artists from Last.fm charts.

        Returns list of artist names, empty list on failure.
        """
        if not self._available:
            return []

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(
                    self.BASE_URL,
                    params={
                        "method": "chart.getTopArtists",
                        "limit": limit,
                        "api_key": self.api_key,
                        "format": "json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            return [artist["name"] for artist in data["artists"]["artist"]]
        except Exception as e:
            logger.warning(f"Last.fm chart.getTopArtists error: {e}")
            return []

    def get_top_artists_by_tag(self, tag: str, limit: int = 50) -> list[str]:
        """Fetch top artists for a genre tag from Last.fm.

        Returns list of artist names, empty list on failure.
        """
        if not self._available:
            return []

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(
                    self.BASE_URL,
                    params={
                        "method": "tag.getTopArtists",
                        "tag": tag,
                        "limit": limit,
                        "api_key": self.api_key,
                        "format": "json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            return [artist["name"] for artist in data["topartists"]["artist"]]
        except Exception as e:
            logger.warning(f"Last.fm tag.getTopArtists error for tag '{tag}': {e}")
            return []
