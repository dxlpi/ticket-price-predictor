"""Popularity feature extraction from external APIs."""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

from ticket_price_predictor.ml.features.base import FeatureExtractor
from ticket_price_predictor.popularity.aggregator import ArtistPopularity, PopularityTier

logger = logging.getLogger(__name__)

# Tier encoding map
_TIER_ENCODING = {
    PopularityTier.LOW: 0,
    PopularityTier.MEDIUM: 1,
    PopularityTier.HIGH: 2,
}


def _safe_log10(value: int | float | None) -> float:
    """Compute log10 safely, returning 0.0 for None/zero/negative values."""
    if value is None or value <= 0:
        return 0.0
    return math.log10(value)


def _encode_tier(tier: PopularityTier) -> int:
    """Encode popularity tier as integer."""
    return _TIER_ENCODING.get(tier, 0)


class PopularityFeatureExtractor(FeatureExtractor):
    """Extract popularity features from external API data.

    Features computed:
    - popularity_score: 0-100 weighted aggregate from all sources
    - popularity_tier_encoded: 0=LOW, 1=MEDIUM, 2=HIGH
    - youtube_subscribers_log: log10-scaled subscriber count
    - youtube_views_log: log10-scaled view count
    - lastfm_listeners_log: log10-scaled listener count
    - lastfm_play_count_log: log10-scaled play count

    Graceful degradation: when popularity_service is None (no API keys),
    all 6 features default to 0.0 with a warning logged once during fit().
    """

    def __init__(self, popularity_service: Any | None = None) -> None:
        """Initialize extractor.

        Args:
            popularity_service: PopularityService instance, or None for graceful degradation
        """
        self._service = popularity_service
        self._artist_cache: dict[str, ArtistPopularity] = {}
        self._warned = False

    @property
    def feature_names(self) -> list[str]:
        """Return list of feature names."""
        return [
            "popularity_score",
            "popularity_tier_encoded",
            "youtube_subscribers_log",
            "youtube_views_log",
            "lastfm_listeners_log",
            "lastfm_play_count_log",
        ]

    def fit(self, df: pd.DataFrame) -> PopularityFeatureExtractor:
        """Pre-fetch popularity data for all unique artists in training data.

        Args:
            df: Training DataFrame with artist_or_team column

        Returns:
            self
        """
        if self._service is None:
            if not self._warned:
                logger.warning(
                    "PopularityFeatureExtractor: no PopularityService provided. "
                    "All popularity features will default to 0.0."
                )
                self._warned = True
            return self

        if "artist_or_team" not in df.columns:
            logger.warning("No 'artist_or_team' column found. Skipping popularity pre-fetch.")
            return self

        unique_artists = df["artist_or_team"].unique()
        logger.info(f"Pre-fetching popularity for {len(unique_artists)} artists...")

        for artist in unique_artists:
            artist_str = str(artist)
            try:
                popularity = self._service.get_artist_popularity(artist_str)
                self._artist_cache[artist_str.lower().strip()] = popularity
            except Exception as e:
                logger.warning(f"Failed to fetch popularity for {artist_str}: {e}")

        logger.info(f"Cached popularity for {len(self._artist_cache)} artists.")
        return self

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract popularity features for each row.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with 6 popularity features
        """
        result = pd.DataFrame(index=df.index)

        scores = []
        tiers = []
        yt_subscribers = []
        yt_views = []
        lfm_listeners = []
        lfm_play_counts = []

        for _, row in df.iterrows():
            artist = str(row.get("artist_or_team", "Unknown")).lower().strip()
            pop = self._artist_cache.get(artist)

            if pop is not None:
                scores.append(pop.popularity_score)
                tiers.append(_encode_tier(pop.tier))
                yt_subscribers.append(_safe_log10(pop.youtube_subscribers))
                yt_views.append(_safe_log10(pop.youtube_views))
                lfm_listeners.append(_safe_log10(pop.lastfm_listeners))
                lfm_play_counts.append(_safe_log10(pop.lastfm_play_count))
            else:
                scores.append(0.0)
                tiers.append(0)
                yt_subscribers.append(0.0)
                yt_views.append(0.0)
                lfm_listeners.append(0.0)
                lfm_play_counts.append(0.0)

        result["popularity_score"] = scores
        result["popularity_tier_encoded"] = tiers
        result["youtube_subscribers_log"] = yt_subscribers
        result["youtube_views_log"] = yt_views
        result["lastfm_listeners_log"] = lfm_listeners
        result["lastfm_play_count_log"] = lfm_play_counts

        return result

    def get_params(self) -> dict[str, Any]:
        """Return extractor parameters."""
        return {
            "has_service": self._service is not None,
            "cached_artists": len(self._artist_cache),
        }
