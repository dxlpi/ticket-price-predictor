"""Artist statistics cache for data-driven popularity features."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from ticket_price_predictor.config import get_ml_config

_config = get_ml_config()


@dataclass
class ArtistStats:
    """Statistics for a single artist."""

    artist_name: str
    avg_price: float
    median_price: float
    price_std: float
    event_count: int
    listing_count: int
    premium_ratio: float  # % of listings > $200

    def to_dict(self) -> dict[str, float]:
        """Convert to feature dictionary."""
        return {
            "artist_avg_price": self.avg_price,
            "artist_median_price": self.median_price,
            "artist_price_std": self.price_std,
            "artist_event_count": float(self.event_count),
            "artist_listing_count": float(self.listing_count),
            "artist_premium_ratio": self.premium_ratio,
        }


class ArtistStatsCache:
    """Cache of artist statistics computed from listing data."""

    PREMIUM_THRESHOLD = _config.premium_price_threshold  # From centralized config

    def __init__(self) -> None:
        """Initialize empty cache."""
        self._stats: dict[str, ArtistStats] = {}
        self._global_stats: ArtistStats | None = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """Return True if cache has been computed."""
        return self._fitted

    @property
    def artist_count(self) -> int:
        """Return number of artists in cache."""
        return len(self._stats)

    def fit(self, df: pd.DataFrame) -> "ArtistStatsCache":
        """Compute artist statistics from listing data.

        Args:
            df: DataFrame with columns: artist_or_team, listing_price, event_id

        Returns:
            self
        """
        if "artist_or_team" not in df.columns or "listing_price" not in df.columns:
            raise ValueError("DataFrame must have 'artist_or_team' and 'listing_price' columns")

        # Compute global stats (for unknown artists)
        self._global_stats = ArtistStats(
            artist_name="__GLOBAL__",
            avg_price=df["listing_price"].mean(),
            median_price=df["listing_price"].median(),
            price_std=df["listing_price"].std(),
            event_count=df["event_id"].nunique() if "event_id" in df.columns else 0,
            listing_count=len(df),
            premium_ratio=(df["listing_price"] > self.PREMIUM_THRESHOLD).mean(),
        )

        # Compute per-artist stats
        self._stats = {}
        grouped = df.groupby("artist_or_team")

        for artist, group in grouped:
            artist_key = self._normalize_artist(str(artist))
            self._stats[artist_key] = ArtistStats(
                artist_name=str(artist),
                avg_price=group["listing_price"].mean(),
                median_price=group["listing_price"].median(),
                price_std=group["listing_price"].std() if len(group) > 1 else 0.0,
                event_count=group["event_id"].nunique() if "event_id" in group.columns else 1,
                listing_count=len(group),
                premium_ratio=(group["listing_price"] > self.PREMIUM_THRESHOLD).mean(),
            )

        self._fitted = True
        return self

    def get_stats(self, artist: str) -> ArtistStats:
        """Get statistics for an artist.

        Args:
            artist: Artist name

        Returns:
            ArtistStats (or global defaults if artist not found)
        """
        if not self._fitted or self._global_stats is None:
            # Return sensible defaults if not fitted (from config)
            return ArtistStats(
                artist_name=artist,
                avg_price=_config.default_avg_price,
                median_price=_config.default_avg_price,
                price_std=_config.default_price_std,
                event_count=0,
                listing_count=0,
                premium_ratio=_config.default_premium_ratio,
            )

        artist_key = self._normalize_artist(artist)

        if artist_key in self._stats:
            return self._stats[artist_key]

        # Return global stats for unknown artists
        return ArtistStats(
            artist_name=artist,
            avg_price=self._global_stats.avg_price,
            median_price=self._global_stats.median_price,
            price_std=self._global_stats.price_std,
            event_count=0,  # Unknown artist has no events in our data
            listing_count=0,
            premium_ratio=self._global_stats.premium_ratio,
        )

    def is_known_artist(self, artist: str) -> bool:
        """Check if artist is in the cache.

        Args:
            artist: Artist name

        Returns:
            True if artist has statistics in cache
        """
        return self._normalize_artist(artist) in self._stats

    def _normalize_artist(self, artist: str) -> str:
        """Normalize artist name for lookup."""
        return artist.lower().strip()

    def save(self, path: Path) -> None:
        """Save cache to disk.

        Args:
            path: Path to save cache
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "stats": self._stats,
                "global_stats": self._global_stats,
                "fitted": self._fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "ArtistStatsCache":
        """Load cache from disk.

        Args:
            path: Path to load cache from

        Returns:
            Loaded cache
        """
        data = joblib.load(path)

        cache = cls()
        cache._stats = data["stats"]
        cache._global_stats = data["global_stats"]
        cache._fitted = data["fitted"]

        return cache

    def summary(self) -> str:
        """Return summary of cache contents."""
        if not self._fitted:
            return "ArtistStatsCache: not fitted"

        lines = [
            f"ArtistStatsCache: {len(self._stats)} artists",
            f"Global avg price: ${self._global_stats.avg_price:.2f}" if self._global_stats else "",
        ]

        # Top 5 artists by avg price
        if self._stats:
            sorted_artists = sorted(
                self._stats.values(), key=lambda x: x.avg_price, reverse=True
            )[:5]
            lines.append("Top artists by avg price:")
            for stats in sorted_artists:
                lines.append(f"  {stats.artist_name}: ${stats.avg_price:.2f}")

        return "\n".join(lines)
