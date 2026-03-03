"""YouTube Music client for artist popularity metrics."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class YouTubeMetrics:
    """Metrics from YouTube Music."""

    channel_id: str
    name: str
    subscriber_count: int
    view_count: int


class YouTubePopularity:
    """Fetch artist popularity from YouTube Music via ytmusicapi."""

    def __init__(self) -> None:
        """Initialize YouTube Music client. No API key needed."""
        try:
            from ytmusicapi import YTMusic

            self._ytmusic = YTMusic()
            self._available = True
        except Exception as e:
            logger.warning(f"YouTube Music client initialization failed: {e}")
            self._ytmusic = None
            self._available = False

    @property
    def available(self) -> bool:
        """Check if YouTube Music client is available."""
        return self._available

    def get_artist_metrics(self, artist_name: str) -> YouTubeMetrics | None:
        """Fetch artist metrics from YouTube Music.

        Args:
            artist_name: Artist name to search for

        Returns:
            YouTubeMetrics if found, None otherwise
        """
        if not self._available or self._ytmusic is None:
            return None

        try:
            results = self._ytmusic.search(artist_name, filter="artists", limit=1)

            if not results:
                logger.debug(f"No YouTube Music results for artist: {artist_name}")
                return None

            artist = results[0]
            browse_id = artist.get("browseId", "")

            if not browse_id:
                logger.debug(f"No browse ID for artist: {artist_name}")
                return None

            # Get detailed artist info
            artist_info = self._ytmusic.get_artist(browse_id)

            # Extract subscriber count from channel info
            subscriber_count = 0
            sub_text = artist_info.get("subscribers")
            if sub_text:
                subscriber_count = self._parse_subscriber_count(sub_text)

            # Extract view count from channel info
            view_count = 0
            views_text = artist_info.get("views")
            if views_text:
                view_count = self._parse_view_count(views_text)

            return YouTubeMetrics(
                channel_id=browse_id,
                name=artist_info.get("name", artist_name),
                subscriber_count=subscriber_count,
                view_count=view_count,
            )
        except Exception as e:
            logger.warning(f"YouTube Music API error for {artist_name}: {e}")
            return None

    @staticmethod
    def _parse_subscriber_count(text: str) -> int:
        """Parse subscriber count from text like '1.5M subscribers'.

        Args:
            text: Subscriber count text

        Returns:
            Parsed count as integer
        """
        try:
            # Remove 'subscribers' and whitespace
            cleaned = text.lower().replace("subscribers", "").replace("subscriber", "").strip()
            multiplier = 1
            if cleaned.endswith("k"):
                multiplier = 1_000
                cleaned = cleaned[:-1]
            elif cleaned.endswith("m"):
                multiplier = 1_000_000
                cleaned = cleaned[:-1]
            elif cleaned.endswith("b"):
                multiplier = 1_000_000_000
                cleaned = cleaned[:-1]
            return int(float(cleaned) * multiplier)
        except (ValueError, IndexError):
            return 0

    @staticmethod
    def _parse_view_count(text: str) -> int:
        """Parse view count from text like '1,234,567 views'.

        Args:
            text: View count text

        Returns:
            Parsed count as integer
        """
        try:
            # Remove 'views' and non-numeric characters except digits
            cleaned = text.lower().replace("views", "").replace("view", "").replace(",", "").strip()
            # Handle K/M/B suffixes
            multiplier = 1
            if cleaned.endswith("k"):
                multiplier = 1_000
                cleaned = cleaned[:-1]
            elif cleaned.endswith("m"):
                multiplier = 1_000_000
                cleaned = cleaned[:-1]
            elif cleaned.endswith("b"):
                multiplier = 1_000_000_000
                cleaned = cleaned[:-1]
            return int(float(cleaned) * multiplier)
        except (ValueError, IndexError):
            return 0
