"""Regional popularity feature extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ticket_price_predictor.config import get_ml_config
from ticket_price_predictor.ml.features.base import FeatureExtractor
from ticket_price_predictor.ml.features.geo_mapping import (
    _normalize_city,
    get_country,
    get_region_key,
)

logger = logging.getLogger(__name__)
_config = get_ml_config()


@dataclass
class RegionalArtistStats:
    """Per-artist per-region statistics."""

    artist: str
    region: str
    avg_price: float
    median_price: float
    listing_count: int


@dataclass
class RegionalMarketStats:
    """Per-city market statistics."""

    city: str
    distinct_artists: int
    total_listings: int
    avg_price: float


class RegionalStatsCache:
    """Cache of regional statistics computed from listing data.

    Follows the same fit/save/load pattern as ArtistStatsCache.
    """

    def __init__(self) -> None:
        """Initialize empty cache."""
        self._artist_city_stats: dict[str, RegionalArtistStats] = {}
        self._artist_country_stats: dict[str, RegionalArtistStats] = {}
        self._artist_global_stats: dict[str, RegionalArtistStats] = {}
        self._market_stats: dict[str, RegionalMarketStats] = {}
        self._max_distinct_artists: int = 1
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """Return True if cache has been computed."""
        return self._fitted

    # Bayesian smoothing factor: minimum samples before trusting group averages.
    # Small groups are pulled toward the global mean to prevent memorization.
    SMOOTHING_FACTOR = 75

    def fit(self, df: pd.DataFrame) -> RegionalStatsCache:
        """Compute regional statistics from listing data with Bayesian smoothing.

        Uses smoothed averages to prevent the model from memorizing small-sample
        group means: smoothed = (n * group_mean + m * global_mean) / (n + m)

        Args:
            df: DataFrame with columns: artist_or_team, listing_price, city

        Returns:
            self
        """
        required = {"artist_or_team", "listing_price", "city"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        df = df.copy()
        df["_city_lower"] = df["city"].apply(_normalize_city)
        df["_country"] = df["_city_lower"].map(get_country)
        df["_region_key"] = df["_city_lower"].map(get_region_key)

        global_mean = df["listing_price"].mean()
        global_median = df["listing_price"].median()
        m = self.SMOOTHING_FACTOR

        # Per-artist per-city stats (smoothed toward artist global)
        self._artist_city_stats = {}
        # Pre-compute artist globals for smoothing target
        artist_globals: dict[str, tuple[float, float]] = {}
        for artist, group in df.groupby("artist_or_team"):
            artist_globals[str(artist)] = (
                group["listing_price"].mean(),
                group["listing_price"].median(),
            )

        for (artist, region_key), group in df.groupby(["artist_or_team", "_region_key"]):
            key = f"{str(artist).lower().strip()}|{region_key}"
            n = len(group)
            group_mean = group["listing_price"].mean()
            group_median = group["listing_price"].median()
            # Smooth toward the artist's global average
            artist_mean, artist_median = artist_globals.get(
                str(artist), (global_mean, global_median)
            )
            smoothed_avg = (n * group_mean + m * artist_mean) / (n + m)
            smoothed_median = (n * group_median + m * artist_median) / (n + m)
            self._artist_city_stats[key] = RegionalArtistStats(
                artist=str(artist),
                region=str(region_key),
                avg_price=smoothed_avg,
                median_price=smoothed_median,
                listing_count=n,
            )

        # Per-artist per-country stats (smoothed toward global)
        self._artist_country_stats = {}
        for (artist, country), group in df.groupby(["artist_or_team", "_country"]):
            key = f"{str(artist).lower().strip()}|{country}"
            n = len(group)
            group_mean = group["listing_price"].mean()
            group_median = group["listing_price"].median()
            smoothed_avg = (n * group_mean + m * global_mean) / (n + m)
            smoothed_median = (n * group_median + m * global_median) / (n + m)
            self._artist_country_stats[key] = RegionalArtistStats(
                artist=str(artist),
                region=str(country),
                avg_price=smoothed_avg,
                median_price=smoothed_median,
                listing_count=n,
            )

        # Per-artist global stats (no smoothing needed — these are the smoothing targets)
        self._artist_global_stats = {}
        for artist, group in df.groupby("artist_or_team"):
            key = str(artist).lower().strip()
            self._artist_global_stats[key] = RegionalArtistStats(
                artist=str(artist),
                region="GLOBAL",
                avg_price=group["listing_price"].mean(),
                median_price=group["listing_price"].median(),
                listing_count=len(group),
            )

        # Per-city market stats
        self._market_stats = {}
        for city, group in df.groupby("_city_lower"):
            self._market_stats[str(city)] = RegionalMarketStats(
                city=str(city),
                distinct_artists=group["artist_or_team"].nunique(),
                total_listings=len(group),
                avg_price=group["listing_price"].mean(),
            )

        self._max_distinct_artists = max(
            (m.distinct_artists for m in self._market_stats.values()), default=1
        )

        self._fitted = True
        return self

    def get_artist_city_stats(self, artist: str, city: str) -> RegionalArtistStats | None:
        """Get stats for artist in specific city.

        Fallback chain: city -> country -> global -> None
        """
        artist_key = artist.lower().strip()
        city_lower = _normalize_city(city)
        region_key = get_region_key(city_lower)

        # Try city-level
        key = f"{artist_key}|{region_key}"
        if key in self._artist_city_stats:
            return self._artist_city_stats[key]

        # Try country-level
        country = get_country(city_lower)
        country_key = f"{artist_key}|{country}"
        if country_key in self._artist_country_stats:
            return self._artist_country_stats[country_key]

        # Try global
        if artist_key in self._artist_global_stats:
            return self._artist_global_stats[artist_key]

        return None

    def get_artist_country_stats(self, artist: str, city: str) -> RegionalArtistStats | None:
        """Get stats for artist at country level."""
        artist_key = artist.lower().strip()
        country = get_country(city)
        key = f"{artist_key}|{country}"
        return self._artist_country_stats.get(key)

    def get_artist_global_stats(self, artist: str) -> RegionalArtistStats | None:
        """Get global stats for artist."""
        return self._artist_global_stats.get(artist.lower().strip())

    def get_market_strength(self, city: str) -> float:
        """Get normalized market strength for a city (0-1).

        Based on distinct artist count relative to the maximum.
        """
        city_lower = _normalize_city(city)
        stats = self._market_stats.get(city_lower)
        if stats is None:
            return _config.regional_default_market_strength
        return stats.distinct_artists / max(self._max_distinct_artists, 1)

    def save(self, path: Path) -> None:
        """Save cache to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "artist_city_stats": self._artist_city_stats,
                "artist_country_stats": self._artist_country_stats,
                "artist_global_stats": self._artist_global_stats,
                "market_stats": self._market_stats,
                "max_distinct_artists": self._max_distinct_artists,
                "fitted": self._fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> RegionalStatsCache:
        """Load cache from disk."""
        data = joblib.load(path)
        cache = cls()
        cache._artist_city_stats = data["artist_city_stats"]
        cache._artist_country_stats = data["artist_country_stats"]
        cache._artist_global_stats = data["artist_global_stats"]
        cache._market_stats = data["market_stats"]
        cache._max_distinct_artists = data["max_distinct_artists"]
        cache._fitted = data["fitted"]
        return cache


class RegionalPopularityFeatureExtractor(FeatureExtractor):
    """Extract regional popularity features from listing data.

    Features computed:
    - artist_regional_avg_price: Average price for this artist in this city
    - artist_regional_median_price: Median price for this artist in this city
    - artist_regional_listing_count: Number of listings for this artist in this city
    - regional_price_ratio: regional_avg / global_avg (how much more/less expensive here)
    - artist_country_avg_price: Average price for this artist in this country
    - country_popularity_ratio: country_avg / global_avg for this artist
    - regional_market_strength: Normalized count of distinct artists in this city (0-1)
    """

    def __init__(self) -> None:
        """Initialize extractor."""
        self._cache = RegionalStatsCache()

    @property
    def feature_names(self) -> list[str]:
        """Return list of feature names."""
        return [
            "artist_regional_avg_price",
            "artist_regional_median_price",
            "artist_regional_listing_count",
            "regional_price_ratio",
            "artist_country_avg_price",
            "country_popularity_ratio",
            "regional_market_strength",
        ]

    @property
    def regional_smoothing(self) -> int:
        """Bayesian smoothing factor forwarded to the nested RegionalStatsCache."""
        return self._cache.SMOOTHING_FACTOR

    @regional_smoothing.setter
    def regional_smoothing(self, value: int) -> None:
        self._cache.SMOOTHING_FACTOR = value

    def fit(self, df: pd.DataFrame) -> RegionalPopularityFeatureExtractor:
        """Fit on training data to build regional stats cache.

        Args:
            df: Training DataFrame with artist_or_team, listing_price, city columns

        Returns:
            self
        """
        self._cache.fit(df)
        return self

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract regional features for each row.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with 7 regional features
        """
        result = pd.DataFrame(index=df.index)

        # Defaults from config
        default_avg = _config.regional_default_avg_price
        default_count = _config.regional_default_listing_count
        default_ratio = _config.regional_default_price_ratio

        regional_avg = []
        regional_median = []
        regional_count = []
        price_ratio = []
        country_avg = []
        country_ratio = []
        market_strength = []

        for _, row in df.iterrows():
            artist = str(row.get("artist_or_team", "Unknown"))
            city = str(row.get("city", "Unknown"))

            # Get regional stats with fallback
            city_stats = self._cache.get_artist_city_stats(artist, city)
            global_stats = self._cache.get_artist_global_stats(artist)
            country_stats = self._cache.get_artist_country_stats(artist, city)

            if city_stats is not None:
                regional_avg.append(city_stats.avg_price)
                regional_median.append(city_stats.median_price)
                regional_count.append(float(city_stats.listing_count))
            else:
                regional_avg.append(default_avg)
                regional_median.append(default_avg)
                regional_count.append(default_count)

            # Price ratio: regional / global
            if city_stats is not None and global_stats is not None and global_stats.avg_price > 0:
                price_ratio.append(city_stats.avg_price / global_stats.avg_price)
            else:
                price_ratio.append(default_ratio)

            # Country-level stats
            if country_stats is not None:
                country_avg.append(country_stats.avg_price)
                if global_stats is not None and global_stats.avg_price > 0:
                    country_ratio.append(country_stats.avg_price / global_stats.avg_price)
                else:
                    country_ratio.append(default_ratio)
            else:
                country_avg.append(default_avg)
                country_ratio.append(default_ratio)

            # Market strength
            market_strength.append(self._cache.get_market_strength(city))

        result["artist_regional_avg_price"] = regional_avg
        result["artist_regional_median_price"] = regional_median
        result["artist_regional_listing_count"] = regional_count
        result["regional_price_ratio"] = price_ratio
        result["artist_country_avg_price"] = country_avg
        result["country_popularity_ratio"] = country_ratio
        result["regional_market_strength"] = market_strength

        return result

    def get_params(self) -> dict[str, Any]:
        """Return extractor parameters."""
        return {"fitted": self._cache.is_fitted}
