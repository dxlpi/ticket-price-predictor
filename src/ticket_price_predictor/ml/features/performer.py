"""Performer/artist feature extraction."""

import pandas as pd

from ticket_price_predictor.ml.features.base import FeatureExtractor


class PerformerFeatureExtractor(FeatureExtractor):
    """Extract features related to performers/artists."""

    # K-pop artists with high demand
    KPOP_KEYWORDS = frozenset([
        "blackpink", "bts", "twice", "stray kids", "aespa", "newjeans",
        "seventeen", "nct", "exo", "red velvet", "itzy", "ive", "le sserafim",
        "txt", "enhypen", "ateez", "skz", "got7", "monsta x", "mamamoo",
    ])

    # Major artists with premium pricing
    MAJOR_ARTISTS = frozenset([
        "taylor swift", "beyonce", "coldplay", "ed sheeran", "eagles",
        "the weeknd", "drake", "bad bunny", "harry styles", "adele",
        "bruno mars", "lady gaga", "rihanna", "justin bieber", "billie eilish",
        "post malone", "dua lipa", "ariana grande", "kanye west", "travis scott",
    ])

    # Country artists (different pricing patterns)
    COUNTRY_ARTISTS = frozenset([
        "morgan wallen", "zach bryan", "luke combs", "chris stapleton",
        "kenny chesney", "jason aldean", "thomas rhett", "carrie underwood",
        "luke bryan", "eric church", "lainey wilson", "cody johnson",
    ])

    @property
    def feature_names(self) -> list[str]:
        """Return list of feature names."""
        return [
            "is_kpop",
            "is_major_artist",
            "is_country",
            "popularity_tier",
            "artist_name_length",
        ]

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract performer features.

        Expects 'artist_or_team' column in input DataFrame.
        """
        result = pd.DataFrame(index=df.index)

        # Normalize artist names for matching
        artist_lower = df["artist_or_team"].str.lower().str.strip()

        # Boolean features
        result["is_kpop"] = artist_lower.apply(self._is_kpop).astype(int)
        result["is_major_artist"] = artist_lower.apply(self._is_major_artist).astype(int)
        result["is_country"] = artist_lower.apply(self._is_country).astype(int)

        # Popularity tier (1-5)
        result["popularity_tier"] = result.apply(
            lambda row: self._compute_popularity_tier(
                row["is_kpop"], row["is_major_artist"], row["is_country"]
            ),
            axis=1,
        )

        # Artist name length (proxy for complexity/uniqueness)
        result["artist_name_length"] = df["artist_or_team"].str.len()

        return result

    def _is_kpop(self, artist: str) -> bool:
        """Check if artist is K-pop."""
        return any(kw in artist for kw in self.KPOP_KEYWORDS)

    def _is_major_artist(self, artist: str) -> bool:
        """Check if artist is a major headliner."""
        return any(major in artist for major in self.MAJOR_ARTISTS)

    def _is_country(self, artist: str) -> bool:
        """Check if artist is country."""
        return any(c in artist for c in self.COUNTRY_ARTISTS)

    def _compute_popularity_tier(
        self, is_kpop: int, is_major: int, is_country: int
    ) -> int:
        """Compute popularity tier 1-5."""
        if is_kpop:
            return 5  # Highest tier - K-pop has extreme demand
        elif is_major:
            return 4  # Major artists
        elif is_country:
            return 3  # Country (strong regional demand)
        else:
            return 2  # Default mid-tier
