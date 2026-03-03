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
    youtube_subscribers: int | None = None
    youtube_views: int | None = None
    lastfm_listeners: int | None = None
    lastfm_play_count: int | None = None

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
        "youtube_subscribers": _config.weight_youtube_subscribers,
        "youtube_views": _config.weight_youtube_views,
        "lastfm_listeners": _config.weight_lastfm_listeners,
        "lastfm_play_count": _config.weight_lastfm_play_count,
    }

    # Tier thresholds (from config)
    HIGH_THRESHOLD = _config.popularity_high_threshold
    MEDIUM_THRESHOLD = _config.popularity_medium_threshold

    # Normalization constants (from config)
    MAX_YOUTUBE_SUBSCRIBERS = _config.max_youtube_subscribers
    MAX_YOUTUBE_VIEWS = _config.max_youtube_views
    MAX_LASTFM_LISTENERS = _config.max_lastfm_listeners
    MAX_LASTFM_PLAY_COUNT = _config.max_lastfm_play_count

    def calculate_score(
        self,
        artist_name: str,
        youtube_subscribers: int | None = None,
        youtube_views: int | None = None,
        lastfm_listeners: int | None = None,
        lastfm_play_count: int | None = None,
    ) -> ArtistPopularity:
        """Calculate combined popularity score with fallback weights.

        Args:
            artist_name: Artist name
            youtube_subscribers: YouTube Music subscriber count
            youtube_views: YouTube Music total view count
            lastfm_listeners: Last.fm unique listener count
            lastfm_play_count: Last.fm total play count

        Returns:
            ArtistPopularity with calculated score and tier
        """
        # Track which sources are available
        sources: list[str] = []
        metrics: dict[str, float] = {}

        if youtube_subscribers is not None:
            sources.append("youtube_subscribers")
            metrics["youtube_subscribers"] = self._normalize_log(
                youtube_subscribers, self.MAX_YOUTUBE_SUBSCRIBERS
            )

        if youtube_views is not None:
            sources.append("youtube_views")
            metrics["youtube_views"] = self._normalize_log(youtube_views, self.MAX_YOUTUBE_VIEWS)

        if lastfm_listeners is not None:
            sources.append("lastfm_listeners")
            metrics["lastfm_listeners"] = self._normalize_log(
                lastfm_listeners, self.MAX_LASTFM_LISTENERS
            )

        if lastfm_play_count is not None:
            sources.append("lastfm_play_count")
            metrics["lastfm_play_count"] = self._normalize_log(
                lastfm_play_count, self.MAX_LASTFM_PLAY_COUNT
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
            youtube_subscribers=youtube_subscribers,
            youtube_views=youtube_views,
            lastfm_listeners=lastfm_listeners,
            lastfm_play_count=lastfm_play_count,
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
