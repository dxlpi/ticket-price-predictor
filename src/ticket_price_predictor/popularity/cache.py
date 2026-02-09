"""Cache for popularity scores to avoid repeated API calls."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ticket_price_predictor.popularity.aggregator import ArtistPopularity

logger = logging.getLogger(__name__)


class PopularityCache:
    """File-based cache for artist popularity scores."""

    DEFAULT_TTL_HOURS = 24  # Cache for 24 hours

    def __init__(
        self,
        cache_dir: Path | str,
        ttl_hours: int = DEFAULT_TTL_HOURS,
    ) -> None:
        """Initialize cache.

        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time-to-live in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        self._cache_file = self.cache_dir / "popularity_cache.json"
        self._cache: dict[str, dict[str, Any]] = self._load_cache()

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        """Load cache from disk."""
        if self._cache_file.exists():
            try:
                with open(self._cache_file) as f:
                    data: dict[str, dict[str, Any]] = json.load(f)
                    return data
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            with open(self._cache_file, "w") as f:
                json.dump(self._cache, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _normalize_key(self, artist_name: str) -> str:
        """Normalize artist name for cache key."""
        return artist_name.lower().strip()

    def get(self, artist_name: str) -> dict[str, Any] | None:
        """Get cached popularity data for an artist.

        Args:
            artist_name: Artist name

        Returns:
            Cached data dict or None if not found/expired
        """
        key = self._normalize_key(artist_name)

        if key not in self._cache:
            return None

        entry = self._cache[key]

        # Check TTL
        cached_at = datetime.fromisoformat(entry.get("cached_at", "2000-01-01"))
        if datetime.now() - cached_at > self.ttl:
            logger.debug(f"Cache expired for {artist_name}")
            del self._cache[key]
            self._save_cache()
            return None

        return entry.get("data")

    def set(self, artist_name: str, popularity: ArtistPopularity) -> None:
        """Cache popularity data for an artist.

        Args:
            artist_name: Artist name
            popularity: ArtistPopularity data to cache
        """
        key = self._normalize_key(artist_name)

        # Convert dataclass to dict for JSON serialization
        data = asdict(popularity)
        data["tier"] = popularity.tier.value  # Convert enum to string
        data["last_updated"] = popularity.last_updated.isoformat()

        self._cache[key] = {
            "data": data,
            "cached_at": datetime.now().isoformat(),
        }

        self._save_cache()

    def clear(self) -> None:
        """Clear all cached data."""
        self._cache = {}
        if self._cache_file.exists():
            self._cache_file.unlink()

    def clear_expired(self) -> int:
        """Remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        now = datetime.now()
        expired_keys = []

        for key, entry in self._cache.items():
            cached_at = datetime.fromisoformat(entry.get("cached_at", "2000-01-01"))
            if now - cached_at > self.ttl:
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            self._save_cache()

        return len(expired_keys)
