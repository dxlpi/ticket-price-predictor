"""Tests for popularity and regional feature extractors."""

import math

import pandas as pd
import pytest

from ticket_price_predictor.ml.features.popularity import (
    PopularityFeatureExtractor,
    _encode_tier,
    _safe_log10,
)
from ticket_price_predictor.ml.features.regional import (
    RegionalPopularityFeatureExtractor,
    RegionalStatsCache,
)
from ticket_price_predictor.popularity.aggregator import ArtistPopularity, PopularityTier

# ============================================================================
# Helper Functions Tests
# ============================================================================


class TestSafeLog10:
    """Tests for _safe_log10 helper."""

    def test_positive_value(self):
        assert _safe_log10(1000) == pytest.approx(3.0)
        assert _safe_log10(100) == pytest.approx(2.0)

    def test_zero(self):
        assert _safe_log10(0) == 0.0

    def test_negative(self):
        assert _safe_log10(-5) == 0.0

    def test_none(self):
        assert _safe_log10(None) == 0.0

    def test_one(self):
        assert _safe_log10(1) == 0.0


class TestEncodeTier:
    """Tests for _encode_tier helper."""

    def test_low(self):
        assert _encode_tier(PopularityTier.LOW) == 0

    def test_medium(self):
        assert _encode_tier(PopularityTier.MEDIUM) == 1

    def test_high(self):
        assert _encode_tier(PopularityTier.HIGH) == 2


# ============================================================================
# PopularityFeatureExtractor Tests
# ============================================================================


class TestPopularityFeatureExtractor:
    """Tests for PopularityFeatureExtractor."""

    def test_feature_names(self):
        """Test feature names are correct."""
        extractor = PopularityFeatureExtractor()
        assert len(extractor.feature_names) == 6
        assert "popularity_score" in extractor.feature_names
        assert "popularity_tier_encoded" in extractor.feature_names
        assert "youtube_subscribers_log" in extractor.feature_names
        assert "youtube_views_log" in extractor.feature_names
        assert "lastfm_listeners_log" in extractor.feature_names
        assert "lastfm_play_count_log" in extractor.feature_names

    def test_graceful_degradation_no_service(self):
        """Test that extractor works without a popularity service."""
        extractor = PopularityFeatureExtractor(popularity_service=None)

        df = pd.DataFrame(
            {
                "artist_or_team": ["Artist A", "Artist B"],
                "listing_price": [100.0, 200.0],
            }
        )

        extractor.fit(df)
        result = extractor.extract(df)

        assert len(result) == 2
        assert len(result.columns) == 6
        # All features should be 0 when no service
        assert result["popularity_score"].sum() == 0.0
        assert result["popularity_tier_encoded"].sum() == 0
        assert result["youtube_subscribers_log"].sum() == 0.0

    def test_extract_with_cached_data(self):
        """Test extraction when cache has data."""
        extractor = PopularityFeatureExtractor(popularity_service=None)

        # Manually populate cache
        extractor._artist_cache["artist a"] = ArtistPopularity(
            name="Artist A",
            popularity_score=75.0,
            tier=PopularityTier.HIGH,
            youtube_subscribers=10_000_000,
            youtube_views=5_000_000_000,
            lastfm_listeners=5_000_000,
            lastfm_play_count=100_000_000,
            sources_available=["youtube_subscribers"],
        )

        df = pd.DataFrame(
            {
                "artist_or_team": ["Artist A", "Unknown Artist"],
            }
        )

        result = extractor.extract(df)

        # Artist A should have values
        assert result["popularity_score"].iloc[0] == 75.0
        assert result["popularity_tier_encoded"].iloc[0] == 2  # HIGH
        assert result["youtube_subscribers_log"].iloc[0] == pytest.approx(math.log10(10_000_000))
        assert result["youtube_views_log"].iloc[0] == pytest.approx(math.log10(5_000_000_000))
        assert result["lastfm_listeners_log"].iloc[0] == pytest.approx(math.log10(5_000_000))
        assert result["lastfm_play_count_log"].iloc[0] == pytest.approx(math.log10(100_000_000))

        # Unknown artist should be 0
        assert result["popularity_score"].iloc[1] == 0.0
        assert result["popularity_tier_encoded"].iloc[1] == 0

    def test_get_params(self):
        """Test get_params."""
        extractor = PopularityFeatureExtractor(popularity_service=None)
        params = extractor.get_params()
        assert params["has_service"] is False
        assert params["cached_artists"] == 0


# ============================================================================
# RegionalStatsCache Tests
# ============================================================================


class TestRegionalStatsCache:
    """Tests for RegionalStatsCache."""

    @pytest.fixture
    def sample_data(self):
        """Create sample listing data with multiple cities."""
        return pd.DataFrame(
            {
                "artist_or_team": [
                    "Taylor Swift",
                    "Taylor Swift",
                    "Taylor Swift",
                    "BTS",
                    "BTS",
                    "BTS",
                    "Taylor Swift",
                    "BTS",
                ],
                "listing_price": [
                    200.0,
                    250.0,
                    300.0,
                    350.0,
                    400.0,
                    450.0,
                    180.0,
                    500.0,
                ],
                "city": [
                    "New York",
                    "New York",
                    "Los Angeles",
                    "New York",
                    "Seoul",
                    "Seoul",
                    "Los Angeles",
                    "Seoul",
                ],
                "event_id": ["e1", "e1", "e2", "e3", "e4", "e4", "e2", "e4"],
            }
        )

    def test_fit(self, sample_data):
        """Test fitting cache."""
        cache = RegionalStatsCache()
        cache.fit(sample_data)
        assert cache.is_fitted

    def test_missing_columns_raises(self):
        """Test that missing required columns raises ValueError."""
        cache = RegionalStatsCache()
        df = pd.DataFrame({"other": [1, 2, 3]})
        with pytest.raises(ValueError, match="missing required columns"):
            cache.fit(df)

    def test_get_artist_city_stats(self, sample_data):
        """Test getting artist stats for a specific city."""
        cache = RegionalStatsCache()
        cache.fit(sample_data)

        stats = cache.get_artist_city_stats("Taylor Swift", "New York")
        assert stats is not None
        assert stats.avg_price == pytest.approx(232.3, abs=1.0)  # Bayesian smoothed
        assert stats.listing_count == 2

    def test_get_artist_city_stats_fallback_to_country(self, sample_data):
        """Test fallback to country level when city not found."""
        cache = RegionalStatsCache()
        cache.fit(sample_data)

        # Tampa has no data, but US country-level exists for Taylor Swift
        stats = cache.get_artist_city_stats("Taylor Swift", "Tampa")
        assert stats is not None
        # Should fall back to US-level stats for Taylor Swift
        # (New York + Los Angeles data, all US)

    def test_get_artist_city_stats_fallback_to_global(self, sample_data):
        """Test fallback to global when country not found."""
        cache = RegionalStatsCache()
        cache.fit(sample_data)

        # Unknown city in unknown country falls back to global
        stats = cache.get_artist_city_stats("Taylor Swift", "Unknown City")
        assert stats is not None

    def test_get_artist_city_stats_unknown_artist(self, sample_data):
        """Test None returned for unknown artist."""
        cache = RegionalStatsCache()
        cache.fit(sample_data)

        stats = cache.get_artist_city_stats("Unknown Artist", "New York")
        assert stats is None

    def test_get_market_strength(self, sample_data):
        """Test market strength calculation."""
        cache = RegionalStatsCache()
        cache.fit(sample_data)

        # New York has both Taylor Swift and BTS
        ny_strength = cache.get_market_strength("New York")
        assert ny_strength > 0

        # Seoul has only BTS
        seoul_strength = cache.get_market_strength("Seoul")
        assert seoul_strength > 0

        # NY should have higher or equal market strength (more artists)
        assert ny_strength >= seoul_strength

    def test_get_market_strength_unknown_city(self, sample_data):
        """Test default market strength for unknown city."""
        cache = RegionalStatsCache()
        cache.fit(sample_data)

        strength = cache.get_market_strength("Unknown City")
        assert strength == 0.5  # default

    def test_save_and_load(self, sample_data, tmp_path):
        """Test saving and loading cache."""
        cache = RegionalStatsCache()
        cache.fit(sample_data)

        save_path = tmp_path / "regional_cache.joblib"
        cache.save(save_path)

        loaded = RegionalStatsCache.load(save_path)
        assert loaded.is_fitted

        # Verify data survived round-trip
        stats = loaded.get_artist_city_stats("Taylor Swift", "New York")
        assert stats is not None
        assert stats.avg_price == pytest.approx(232.3, abs=1.0)  # Bayesian smoothed

    def test_get_artist_country_stats(self, sample_data):
        """Test getting artist stats at country level."""
        cache = RegionalStatsCache()
        cache.fit(sample_data)

        # BTS in Seoul -> KR country
        stats = cache.get_artist_country_stats("BTS", "Seoul")
        assert stats is not None
        assert stats.listing_count == 3  # 3 Seoul listings


# ============================================================================
# RegionalPopularityFeatureExtractor Tests
# ============================================================================


class TestRegionalPopularityFeatureExtractor:
    """Tests for RegionalPopularityFeatureExtractor."""

    @pytest.fixture
    def sample_data(self):
        """Create sample listing data."""
        return pd.DataFrame(
            {
                "artist_or_team": [
                    "Taylor Swift",
                    "Taylor Swift",
                    "BTS",
                    "BTS",
                    "Taylor Swift",
                    "BTS",
                ],
                "listing_price": [200.0, 250.0, 350.0, 400.0, 180.0, 500.0],
                "city": [
                    "New York",
                    "New York",
                    "Seoul",
                    "Seoul",
                    "Los Angeles",
                    "New York",
                ],
                "event_id": ["e1", "e1", "e2", "e2", "e3", "e4"],
            }
        )

    def test_feature_names(self):
        """Test feature names."""
        extractor = RegionalPopularityFeatureExtractor()
        assert len(extractor.feature_names) == 7
        assert "artist_regional_avg_price" in extractor.feature_names
        assert "regional_price_ratio" in extractor.feature_names
        assert "regional_market_strength" in extractor.feature_names

    def test_fit_and_extract(self, sample_data):
        """Test fit and extract produces correct shape."""
        extractor = RegionalPopularityFeatureExtractor()
        extractor.fit(sample_data)
        result = extractor.extract(sample_data)

        assert len(result) == len(sample_data)
        assert len(result.columns) == 7

    def test_extract_values(self, sample_data):
        """Test extracted values are reasonable."""
        extractor = RegionalPopularityFeatureExtractor()
        extractor.fit(sample_data)
        result = extractor.extract(sample_data)

        # All prices should be positive
        assert (result["artist_regional_avg_price"] > 0).all()

        # Price ratios should be positive
        assert (result["regional_price_ratio"] > 0).all()

        # Market strength should be between 0 and 1
        assert (result["regional_market_strength"] >= 0).all()
        assert (result["regional_market_strength"] <= 1).all()

    def test_extract_without_fit_uses_defaults(self):
        """Test extraction without fitting uses defaults."""
        extractor = RegionalPopularityFeatureExtractor()

        df = pd.DataFrame(
            {
                "artist_or_team": ["Unknown Artist"],
                "listing_price": [100.0],
                "city": ["Unknown City"],
            }
        )

        result = extractor.extract(df)
        assert len(result) == 1
        assert result["artist_regional_avg_price"].iloc[0] == 150.0  # default
        assert result["regional_price_ratio"].iloc[0] == 1.0  # default

    def test_get_params(self, sample_data):
        """Test get_params shows fitted state."""
        extractor = RegionalPopularityFeatureExtractor()
        assert extractor.get_params()["fitted"] is False

        extractor.fit(sample_data)
        assert extractor.get_params()["fitted"] is True
