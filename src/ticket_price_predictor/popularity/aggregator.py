"""Popularity score aggregation from multiple sources."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from ticket_price_predictor.config import get_ml_config

_config = get_ml_config()


class PopularityTier(StrEnum):
    """Artist popularity tiers."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ArtistPopularity:
    """Aggregated artist popularity data."""

    name: str
    popularity_score: float  # 0-100 combined score
    tier: PopularityTier

    # Raw metrics from each source
    spotify_popularity: int | None = None
    spotify_followers: int | None = None
    songkick_trackers: int | None = None
    bandsintown_trackers: int | None = None

    # Metadata
    sources_available: list[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

    # Tier allocation for events
    @property
    def tier_allocation(self) -> int:
        """Number of events to collect based on tier."""
        if self.tier == PopularityTier.HIGH:
            return 5
        elif self.tier == PopularityTier.MEDIUM:
            return 3
        return 1


class PopularityAggregator:
    """Combine popularity metrics from multiple sources into a unified score."""

    # Default weights for each source (from centralized config)
    WEIGHTS = {
        "spotify_popularity": _config.weight_spotify_popularity,
        "spotify_followers": _config.weight_spotify_followers,
        "songkick_trackers": _config.weight_songkick_trackers,
        "bandsintown_trackers": _config.weight_bandsintown_trackers,
    }

    # Tier thresholds (from config)
    HIGH_THRESHOLD = _config.popularity_high_threshold
    MEDIUM_THRESHOLD = _config.popularity_medium_threshold

    # Normalization constants (from config)
    MAX_FOLLOWERS = _config.max_spotify_followers
    MAX_TRACKERS = _config.max_tracker_count

    def calculate_score(
        self,
        artist_name: str,
        spotify_popularity: int | None = None,
        spotify_followers: int | None = None,
        songkick_trackers: int | None = None,
        bandsintown_trackers: int | None = None,
    ) -> ArtistPopularity:
        """Calculate combined popularity score with fallback weights.

        Args:
            artist_name: Artist name
            spotify_popularity: Spotify popularity (0-100)
            spotify_followers: Spotify follower count
            songkick_trackers: Songkick tracker count
            bandsintown_trackers: Bandsintown tracker count

        Returns:
            ArtistPopularity with calculated score and tier
        """
        # Track which sources are available
        sources: list[str] = []
        metrics: dict[str, float] = {}

        if spotify_popularity is not None:
            sources.append("spotify_popularity")
            metrics["spotify_popularity"] = float(spotify_popularity)  # Already 0-100

        if spotify_followers is not None:
            sources.append("spotify_followers")
            metrics["spotify_followers"] = self._normalize_log(
                spotify_followers, self.MAX_FOLLOWERS
            )

        if songkick_trackers is not None:
            sources.append("songkick_trackers")
            metrics["songkick_trackers"] = self._normalize_log(songkick_trackers, self.MAX_TRACKERS)

        if bandsintown_trackers is not None:
            sources.append("bandsintown_trackers")
            metrics["bandsintown_trackers"] = self._normalize_log(
                bandsintown_trackers, self.MAX_TRACKERS
            )

        # Calculate weighted score with redistributed weights
        score = self._weighted_score(metrics, sources)

        # Determine tier
        if score >= self.HIGH_THRESHOLD:
            tier = PopularityTier.HIGH
        elif score >= self.MEDIUM_THRESHOLD:
            tier = PopularityTier.MEDIUM
        else:
            tier = PopularityTier.LOW

        return ArtistPopularity(
            name=artist_name,
            popularity_score=round(score, 2),
            tier=tier,
            spotify_popularity=spotify_popularity,
            spotify_followers=spotify_followers,
            songkick_trackers=songkick_trackers,
            bandsintown_trackers=bandsintown_trackers,
            sources_available=sources,
            last_updated=datetime.now(),
        )

    def _normalize_log(self, value: int, max_value: int) -> float:
        """Normalize a value using log scale to 0-100.

        Args:
            value: Raw value
            max_value: Maximum expected value

        Returns:
            Normalized value 0-100
        """
        if value <= 0:
            return 0.0
        if value >= max_value:
            return 100.0

        # Log scale normalization
        log_value = math.log10(value + 1)
        log_max = math.log10(max_value + 1)
        return (log_value / log_max) * 100

    def _weighted_score(self, metrics: dict[str, float], sources: list[str]) -> float:
        """Calculate weighted score, redistributing weights for missing sources.

        Args:
            metrics: Dictionary of normalized metric values
            sources: List of available source names

        Returns:
            Weighted score 0-100
        """
        if not sources:
            return 0.0

        # Get weights for available sources
        available_weights = {s: self.WEIGHTS[s] for s in sources}

        # Redistribute weights to sum to 1.0
        total_weight = sum(available_weights.values())
        if total_weight == 0:
            return 0.0

        normalized_weights = {s: w / total_weight for s, w in available_weights.items()}

        # Calculate weighted sum
        score = sum(metrics[source] * normalized_weights[source] for source in sources)

        return min(100.0, max(0.0, score))
