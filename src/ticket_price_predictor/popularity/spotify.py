"""Spotify API client for artist popularity metrics."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SpotifyMetrics:
    """Metrics from Spotify API."""
    spotify_id: str
    popularity: int  # 0-100
    followers: int
    genres: list[str]


class SpotifyPopularity:
    """Fetch artist popularity from Spotify API."""

    def __init__(self, client_id: str, client_secret: str) -> None:
        """Initialize Spotify client.

        Args:
            client_id: Spotify API client ID
            client_secret: Spotify API client secret
        """
        try:
            import spotipy
            from spotipy.oauth2 import SpotifyClientCredentials

            self.sp = spotipy.Spotify(
                auth_manager=SpotifyClientCredentials(
                    client_id=client_id,
                    client_secret=client_secret,
                )
            )
            self._available = True
        except Exception as e:
            logger.warning(f"Spotify client initialization failed: {e}")
            self.sp = None
            self._available = False

    @property
    def available(self) -> bool:
        """Check if Spotify client is available."""
        return self._available

    def get_artist_metrics(self, artist_name: str) -> SpotifyMetrics | None:
        """Fetch artist metrics from Spotify.

        Args:
            artist_name: Artist name to search for

        Returns:
            SpotifyMetrics if found, None otherwise
        """
        if not self._available or self.sp is None:
            return None

        try:
            results = self.sp.search(q=f"artist:{artist_name}", type="artist", limit=1)
            items = results.get("artists", {}).get("items", [])

            if not items:
                logger.debug(f"No Spotify results for artist: {artist_name}")
                return None

            artist = items[0]
            return SpotifyMetrics(
                spotify_id=artist["id"],
                popularity=artist.get("popularity", 0),
                followers=artist.get("followers", {}).get("total", 0),
                genres=artist.get("genres", []),
            )
        except Exception as e:
            logger.warning(f"Spotify API error for {artist_name}: {e}")
            return None
