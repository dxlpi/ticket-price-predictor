"""Performer/artist feature extraction using data-driven statistics."""

from typing import Any

import pandas as pd

from ticket_price_predictor.ml.features.artist_stats import ArtistStatsCache
from ticket_price_predictor.ml.features.base import FeatureExtractor
from ticket_price_predictor.normalization.seat_zones import SeatZoneMapper


class PerformerFeatureExtractor(FeatureExtractor):
    """Extract features related to performers/artists.

    Uses data-driven statistics computed from historical listing data
    instead of hardcoded artist lists.
    """

    _ZONE_SMOOTHING = 20  # Bayesian smoothing factor for artist×zone medians

    def __init__(self, stats_cache: ArtistStatsCache | None = None) -> None:
        """Initialize extractor.

        Args:
            stats_cache: Pre-computed artist statistics cache.
                        If None, will be computed during fit().
        """
        self._stats_cache = stats_cache or ArtistStatsCache()
        self._zone_mapper = SeatZoneMapper()
        # dict[(artist_key, zone_value)] -> smoothed median price
        self._artist_zone_medians: dict[tuple[str, str], float] = {}
        self._global_median: float = 150.0

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
            "artist_zone_median_price",
        ]

    def fit(self, df: pd.DataFrame) -> "PerformerFeatureExtractor":
        """Fit extractor by computing artist statistics.

        Args:
            df: Training DataFrame with artist_or_team, listing_price, and
                optionally section columns.

        Returns:
            self
        """
        if not self._stats_cache.is_fitted:
            self._stats_cache.fit(df)

        # Compute global median for ultimate fallback
        if "listing_price" in df.columns:
            self._global_median = float(df["listing_price"].median())

        # Build artist×zone median price lookup
        self._artist_zone_medians = {}
        if (
            "section" in df.columns
            and "listing_price" in df.columns
            and "artist_or_team" in df.columns
        ):
            working = df[["artist_or_team", "section", "listing_price"]].copy()
            working["_zone"] = working["section"].apply(
                lambda s: self._zone_mapper.normalize_zone_name(str(s)).value
            )
            working["_artist_key"] = working["artist_or_team"].str.lower().str.strip()

            m = self._ZONE_SMOOTHING
            for (artist_key, zone_val), group in working.groupby(["_artist_key", "_zone"]):
                n = len(group)
                group_median = float(group["listing_price"].median())
                artist_stats = self._stats_cache.get_stats(str(artist_key))
                artist_median = artist_stats.median_price
                smoothed = (n * group_median + m * artist_median) / (n + m)
                self._artist_zone_medians[(str(artist_key), str(zone_val))] = smoothed

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

        # Artist × zone median price (Bayesian-smoothed)
        def _lookup_zone_median(row: pd.Series) -> float:
            artist_key = str(row["artist_or_team"]).lower().strip()
            if "section" in row.index:
                zone_val = self._zone_mapper.normalize_zone_name(str(row["section"])).value
                key = (artist_key, zone_val)
                if key in self._artist_zone_medians:
                    return self._artist_zone_medians[key]
            # Fallback to artist median
            artist_stats = self._stats_cache.get_stats(artist_key)
            if artist_stats.median_price > 0:
                return artist_stats.median_price
            return self._global_median

        result["artist_zone_median_price"] = df.apply(_lookup_zone_median, axis=1)

        return result

    @property
    def stats_cache(self) -> ArtistStatsCache:
        """Return the artist stats cache."""
        return self._stats_cache

    @property
    def artist_stats_smoothing(self) -> int:
        """Bayesian smoothing factor forwarded to the nested ArtistStatsCache."""
        return self._stats_cache.SMOOTHING_FACTOR

    @artist_stats_smoothing.setter
    def artist_stats_smoothing(self, value: int) -> None:
        self._stats_cache.SMOOTHING_FACTOR = value

    def get_params(self) -> dict[str, Any]:
        """Return extractor parameters."""
        return {"stats_cache_fitted": self._stats_cache.is_fitted}
