"""Performer/artist feature extraction using data-driven statistics."""

import pandas as pd

from ticket_price_predictor.ml.features.artist_stats import ArtistStatsCache
from ticket_price_predictor.ml.features.base import FeatureExtractor


class PerformerFeatureExtractor(FeatureExtractor):
    """Extract features related to performers/artists.

    Uses data-driven statistics computed from historical listing data
    instead of hardcoded artist lists.
    """

    def __init__(self, stats_cache: ArtistStatsCache | None = None) -> None:
        """Initialize extractor.

        Args:
            stats_cache: Pre-computed artist statistics cache.
                        If None, will be computed during fit().
        """
        self._stats_cache = stats_cache or ArtistStatsCache()

    @property
    def feature_names(self) -> list[str]:
        """Return list of feature names."""
        return [
            "artist_avg_price",
            "artist_median_price",
            "artist_price_std",
            "artist_event_count",
            "artist_listing_count",
            "artist_premium_ratio",
            "is_known_artist",
        ]

    def fit(self, df: pd.DataFrame) -> "PerformerFeatureExtractor":
        """Fit extractor by computing artist statistics.

        Args:
            df: Training DataFrame with artist_or_team and listing_price columns

        Returns:
            self
        """
        if not self._stats_cache.is_fitted:
            self._stats_cache.fit(df)
        return self

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract performer features.

        Expects 'artist_or_team' column in input DataFrame.
        """
        result = pd.DataFrame(index=df.index)

        # Get stats for each artist
        stats_list = df["artist_or_team"].apply(self._stats_cache.get_stats)

        # Extract features from stats
        result["artist_avg_price"] = stats_list.apply(lambda s: s.avg_price)
        result["artist_median_price"] = stats_list.apply(lambda s: s.median_price)
        result["artist_price_std"] = stats_list.apply(lambda s: s.price_std)
        result["artist_event_count"] = stats_list.apply(lambda s: float(s.event_count))
        result["artist_listing_count"] = stats_list.apply(lambda s: float(s.listing_count))
        result["artist_premium_ratio"] = stats_list.apply(lambda s: s.premium_ratio)

        # Flag for known vs unknown artists
        result["is_known_artist"] = df["artist_or_team"].apply(
            lambda a: 1 if self._stats_cache.is_known_artist(a) else 0
        )

        return result

    @property
    def stats_cache(self) -> ArtistStatsCache:
        """Return the artist stats cache."""
        return self._stats_cache

    def get_params(self) -> dict:
        """Return extractor parameters."""
        return {"stats_cache_fitted": self._stats_cache.is_fitted}
