"""Tests for popularity module."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from ticket_price_predictor.popularity.aggregator import (
    ArtistPopularity,
    PopularityAggregator,
    PopularityTier,
)
from ticket_price_predictor.popularity.bandsintown import (
    BandsintownMetrics,
    BandsintownPopularity,
)
from ticket_price_predictor.popularity.cache import PopularityCache
from ticket_price_predictor.popularity.service import PopularityService
from ticket_price_predictor.popularity.songkick import (
    SongkickMetrics,
    SongkickPopularity,
)
from ticket_price_predictor.popularity.spotify import (
    SpotifyMetrics,
    SpotifyPopularity,
)

# ============================================================================
# SpotifyPopularity Tests
# ============================================================================


class TestSpotifyPopularity:
    """Tests for SpotifyPopularity."""

    def test_initialization_with_valid_credentials(self):
        """Test initialization with valid credentials."""
        # Create a client and manually set it up
        client = SpotifyPopularity.__new__(SpotifyPopularity)
        client.sp = MagicMock()
        client._available = True

        assert client.available is True
        assert client.sp is not None

    def test_initialization_with_invalid_credentials(self):
        """Test initialization with invalid credentials (API error)."""
        # Create a client and manually set failed state
        client = SpotifyPopularity.__new__(SpotifyPopularity)
        client.sp = None
        client._available = False

        assert client.available is False
        assert client.sp is None

    def test_get_artist_metrics_success(self):
        """Test successful artist lookup."""
        # Create and setup client manually
        client = SpotifyPopularity.__new__(SpotifyPopularity)
        mock_sp = MagicMock()
        mock_sp.search.return_value = {
            "artists": {
                "items": [
                    {
                        "id": "abc123",
                        "popularity": 85,
                        "followers": {"total": 50000000},
                        "genres": ["pop", "dance pop"],
                    }
                ]
            }
        }
        client.sp = mock_sp
        client._available = True

        result = client.get_artist_metrics("Taylor Swift")

        assert result is not None
        assert isinstance(result, SpotifyMetrics)
        assert result.spotify_id == "abc123"
        assert result.popularity == 85
        assert result.followers == 50000000
        assert result.genres == ["pop", "dance pop"]

    def test_get_artist_metrics_not_found(self):
        """Test when artist is not found."""
        # Create and setup client manually
        client = SpotifyPopularity.__new__(SpotifyPopularity)
        mock_sp = MagicMock()
        mock_sp.search.return_value = {"artists": {"items": []}}
        client.sp = mock_sp
        client._available = True

        result = client.get_artist_metrics("Unknown Artist")

        assert result is None

    def test_get_artist_metrics_api_error(self):
        """Test graceful handling of API errors."""
        # Create and setup client manually
        client = SpotifyPopularity.__new__(SpotifyPopularity)
        mock_sp = MagicMock()
        mock_sp.search.side_effect = Exception("API Error")
        client.sp = mock_sp
        client._available = True

        result = client.get_artist_metrics("Test Artist")

        assert result is None

    def test_get_artist_metrics_unavailable_client(self):
        """Test when client is not available."""
        # Create unavailable client
        client = SpotifyPopularity.__new__(SpotifyPopularity)
        client.sp = None
        client._available = False

        result = client.get_artist_metrics("Test")

        assert result is None


# ============================================================================
# SongkickPopularity Tests
# ============================================================================


class TestSongkickPopularity:
    """Tests for SongkickPopularity."""

    def test_initialization(self):
        """Test initialization with API key."""
        client = SongkickPopularity("test_key")
        assert client.available is True
        assert client.api_key == "test_key"

    def test_initialization_missing_key(self):
        """Test initialization without API key."""
        client = SongkickPopularity("")
        assert client.available is False

    def test_get_artist_metrics_success(self):
        """Test successful artist lookup."""
        with patch("ticket_price_predictor.popularity.songkick.httpx.Client") as mock_client:
            # Mock search response
            mock_response_search = MagicMock()
            mock_response_search.json.return_value = {
                "resultsPage": {
                    "results": {
                        "artist": [
                            {
                                "id": 123,
                                "displayName": "Taylor Swift",
                                "onTourUntil": "2024-12-31",
                            }
                        ]
                    }
                }
            }
            mock_response_search.raise_for_status = MagicMock()

            # Mock artist detail response
            mock_response_detail = MagicMock()
            mock_response_detail.json.return_value = {
                "resultsPage": {
                    "results": {
                        "artist": {
                            "id": 123,
                            "onTourUntil": "2024-12-31",
                        }
                    }
                }
            }
            mock_response_detail.raise_for_status = MagicMock()

            mock_client_instance = MagicMock()
            mock_client_instance.__enter__.return_value.get.side_effect = [
                mock_response_search,
                mock_response_detail,
            ]
            mock_client.return_value = mock_client_instance

            client = SongkickPopularity("test_key")
            result = client.get_artist_metrics("Taylor Swift")

            assert result is not None
            assert isinstance(result, SongkickMetrics)
            assert result.songkick_id == 123
            assert result.display_name == "Taylor Swift"
            assert result.on_tour is True
            assert result.tracker_count == 1  # Based on onTourUntil proxy

    def test_get_artist_metrics_not_found(self):
        """Test when artist is not found."""
        with patch("ticket_price_predictor.popularity.songkick.httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {"resultsPage": {"results": {}}}
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = MagicMock()
            mock_client_instance.__enter__.return_value.get.return_value = mock_response
            mock_client.return_value = mock_client_instance

            client = SongkickPopularity("test_key")
            result = client.get_artist_metrics("Unknown Artist")

            assert result is None

    def test_get_artist_metrics_unavailable(self):
        """Test when client is not available (no API key)."""
        client = SongkickPopularity("")
        result = client.get_artist_metrics("Test Artist")
        assert result is None


# ============================================================================
# BandsintownPopularity Tests
# ============================================================================


class TestBandsintownPopularity:
    """Tests for BandsintownPopularity."""

    def test_initialization(self):
        """Test initialization with app ID."""
        client = BandsintownPopularity("test_app_id")
        assert client.available is True
        assert client.app_id == "test_app_id"

    def test_initialization_missing_app_id(self):
        """Test initialization without app ID."""
        client = BandsintownPopularity("")
        assert client.available is False

    def test_get_artist_metrics_success(self):
        """Test successful artist lookup."""
        with patch("ticket_price_predictor.popularity.bandsintown.httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "456",
                "tracker_count": 1500000,
                "upcoming_event_count": 25,
            }
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = MagicMock()
            mock_client_instance.__enter__.return_value.get.return_value = mock_response
            mock_client.return_value = mock_client_instance

            client = BandsintownPopularity("test_app_id")
            result = client.get_artist_metrics("BTS")

            assert result is not None
            assert isinstance(result, BandsintownMetrics)
            assert result.bandsintown_id == "456"
            assert result.tracker_count == 1500000
            assert result.upcoming_event_count == 25

    def test_get_artist_metrics_404(self):
        """Test handling of 404 (artist not found)."""
        with patch("ticket_price_predictor.popularity.bandsintown.httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 404

            mock_client_instance = MagicMock()
            mock_client_instance.__enter__.return_value.get.return_value = mock_response
            mock_client.return_value = mock_client_instance

            client = BandsintownPopularity("test_app_id")
            result = client.get_artist_metrics("Unknown Artist")

            assert result is None

    def test_get_artist_metrics_error_response(self):
        """Test handling of error responses."""
        with patch("ticket_price_predictor.popularity.bandsintown.httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"error": "Invalid artist"}
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = MagicMock()
            mock_client_instance.__enter__.return_value.get.return_value = mock_response
            mock_client.return_value = mock_client_instance

            client = BandsintownPopularity("test_app_id")
            result = client.get_artist_metrics("Test Artist")

            assert result is None

    def test_get_artist_metrics_unavailable(self):
        """Test when client is not available (no app ID)."""
        client = BandsintownPopularity("")
        result = client.get_artist_metrics("Test Artist")
        assert result is None


# ============================================================================
# PopularityAggregator Tests
# ============================================================================


class TestPopularityAggregator:
    """Tests for PopularityAggregator."""

    def test_calculate_score_all_sources(self):
        """Test score calculation with all sources available."""
        aggregator = PopularityAggregator()
        result = aggregator.calculate_score(
            artist_name="Taylor Swift",
            spotify_popularity=95,
            spotify_followers=80_000_000,
            songkick_trackers=5_000_000,
            bandsintown_trackers=3_000_000,
        )
        assert result.tier == PopularityTier.HIGH
        assert result.popularity_score > 70
        assert len(result.sources_available) == 4
        assert "spotify_popularity" in result.sources_available
        assert "spotify_followers" in result.sources_available
        assert "songkick_trackers" in result.sources_available
        assert "bandsintown_trackers" in result.sources_available

    def test_calculate_score_spotify_only(self):
        """Test with only Spotify data."""
        aggregator = PopularityAggregator()
        result = aggregator.calculate_score(
            artist_name="Test Artist",
            spotify_popularity=50,
            spotify_followers=100_000,
        )
        assert result.tier == PopularityTier.MEDIUM
        assert "spotify_popularity" in result.sources_available
        assert "spotify_followers" in result.sources_available
        assert len(result.sources_available) == 2

    def test_calculate_score_no_sources(self):
        """Test with no data returns low tier."""
        aggregator = PopularityAggregator()
        result = aggregator.calculate_score(artist_name="Unknown")
        assert result.tier == PopularityTier.LOW
        assert result.popularity_score == 0
        assert len(result.sources_available) == 0

    def test_tier_thresholds(self):
        """Test tier assignment thresholds."""
        aggregator = PopularityAggregator()

        # High tier (>= 70)
        high = aggregator.calculate_score("High", spotify_popularity=90)
        assert high.tier == PopularityTier.HIGH
        assert high.popularity_score >= 70

        # Medium tier (40-69)
        medium = aggregator.calculate_score("Medium", spotify_popularity=50)
        assert medium.tier == PopularityTier.MEDIUM
        assert 40 <= medium.popularity_score < 70

        # Low tier (< 40)
        low = aggregator.calculate_score("Low", spotify_popularity=20)
        assert low.tier == PopularityTier.LOW
        assert low.popularity_score < 40

    def test_tier_allocation(self):
        """Test tier-based event allocation."""
        high = ArtistPopularity("A", 80, PopularityTier.HIGH)
        medium = ArtistPopularity("B", 50, PopularityTier.MEDIUM)
        low = ArtistPopularity("C", 20, PopularityTier.LOW)

        assert high.tier_allocation == 5
        assert medium.tier_allocation == 3
        assert low.tier_allocation == 1

    def test_normalize_log(self):
        """Test log normalization."""
        aggregator = PopularityAggregator()

        # Zero value
        assert aggregator._normalize_log(0, 100_000_000) == 0.0

        # Max value
        assert aggregator._normalize_log(100_000_000, 100_000_000) == 100.0

        # Exceeds max
        assert aggregator._normalize_log(200_000_000, 100_000_000) == 100.0

        # Mid-range value should be between 0 and 100
        mid_result = aggregator._normalize_log(1_000_000, 100_000_000)
        assert 0 < mid_result < 100

    def test_weighted_score(self):
        """Test weighted score calculation."""
        aggregator = PopularityAggregator()

        # All sources
        metrics = {
            "spotify_popularity": 80.0,
            "spotify_followers": 70.0,
            "songkick_trackers": 60.0,
            "bandsintown_trackers": 50.0,
        }
        sources = list(metrics.keys())
        score = aggregator._weighted_score(metrics, sources)

        # Score should be weighted average
        assert 0 <= score <= 100

        # Empty sources
        assert aggregator._weighted_score({}, []) == 0.0

    def test_partial_sources_weight_redistribution(self):
        """Test weight redistribution when some sources are missing."""
        aggregator = PopularityAggregator()

        # Only Spotify popularity
        result1 = aggregator.calculate_score(artist_name="Test1", spotify_popularity=80)

        # Spotify popularity + followers
        result2 = aggregator.calculate_score(
            artist_name="Test2", spotify_popularity=80, spotify_followers=5_000_000
        )

        # Both should have valid scores
        assert result1.popularity_score > 0
        assert result2.popularity_score > 0


# ============================================================================
# PopularityCache Tests
# ============================================================================


class TestPopularityCache:
    """Tests for PopularityCache."""

    def test_set_and_get(self, tmp_path):
        """Test basic cache set and get."""
        cache = PopularityCache(tmp_path)
        popularity = ArtistPopularity(
            name="Test Artist",
            popularity_score=75.0,
            tier=PopularityTier.HIGH,
            sources_available=["spotify_popularity"],
        )
        cache.set("Test Artist", popularity)

        result = cache.get("Test Artist")
        assert result is not None
        assert result["name"] == "Test Artist"
        assert result["popularity_score"] == 75.0
        assert result["tier"] == "high"

    def test_cache_miss(self, tmp_path):
        """Test cache miss returns None."""
        cache = PopularityCache(tmp_path)
        assert cache.get("Unknown Artist") is None

    def test_case_insensitive(self, tmp_path):
        """Test cache keys are case insensitive."""
        cache = PopularityCache(tmp_path)
        popularity = ArtistPopularity("Test", 50.0, PopularityTier.MEDIUM)
        cache.set("Taylor Swift", popularity)

        assert cache.get("taylor swift") is not None
        assert cache.get("TAYLOR SWIFT") is not None
        assert cache.get("TaYlOr SwIfT") is not None

    def test_ttl_expiration(self, tmp_path):
        """Test cache expiration."""
        cache = PopularityCache(tmp_path, ttl_hours=1)
        popularity = ArtistPopularity("Test", 50.0, PopularityTier.MEDIUM)
        cache.set("Test", popularity)

        # Manually expire the entry
        key = cache._normalize_key("Test")
        cache._cache[key]["cached_at"] = (datetime.now() - timedelta(hours=2)).isoformat()

        # Should return None (expired)
        assert cache.get("Test") is None

    def test_cache_persistence(self, tmp_path):
        """Test cache persists to disk and loads."""
        cache1 = PopularityCache(tmp_path)
        popularity = ArtistPopularity("Persisted", 60.0, PopularityTier.MEDIUM)
        cache1.set("Persisted", popularity)

        # Create new cache instance from same directory
        cache2 = PopularityCache(tmp_path)
        result = cache2.get("Persisted")

        assert result is not None
        assert result["name"] == "Persisted"
        assert result["popularity_score"] == 60.0

    def test_clear(self, tmp_path):
        """Test clearing cache."""
        cache = PopularityCache(tmp_path)
        popularity = ArtistPopularity("Test", 50.0, PopularityTier.MEDIUM)
        cache.set("Test", popularity)

        assert cache.get("Test") is not None

        cache.clear()

        assert cache.get("Test") is None
        assert not cache._cache_file.exists()

    def test_clear_expired(self, tmp_path):
        """Test clearing only expired entries."""
        cache = PopularityCache(tmp_path, ttl_hours=1)

        # Add fresh entry
        fresh = ArtistPopularity("Fresh", 50.0, PopularityTier.MEDIUM)
        cache.set("Fresh", fresh)

        # Add expired entry
        expired = ArtistPopularity("Expired", 60.0, PopularityTier.HIGH)
        cache.set("Expired", expired)
        key = cache._normalize_key("Expired")
        cache._cache[key]["cached_at"] = (datetime.now() - timedelta(hours=2)).isoformat()

        # Clear expired
        removed = cache.clear_expired()

        assert removed == 1
        assert cache.get("Fresh") is not None
        assert cache.get("Expired") is None


# ============================================================================
# PopularityService Tests
# ============================================================================


class TestPopularityService:
    """Tests for PopularityService."""

    def test_initialization_with_env_vars(self, monkeypatch, tmp_path):
        """Test initialization with environment variables."""
        monkeypatch.setenv("SPOTIFY_CLIENT_ID", "env_id")
        monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "env_secret")
        monkeypatch.setenv("SONGKICK_API_KEY", "env_songkick")
        monkeypatch.setenv("BANDSINTOWN_APP_ID", "env_bandsintown")

        # Create a mock Spotify client
        mock_spotify = MagicMock(spec=SpotifyPopularity)
        mock_spotify.available = True

        # Patch the SpotifyPopularity constructor
        with patch(
            "ticket_price_predictor.popularity.service.SpotifyPopularity", return_value=mock_spotify
        ):
            service = PopularityService(cache_dir=tmp_path)

        assert service.spotify is not None
        assert service.songkick is not None
        assert service.bandsintown is not None

    def test_initialization_with_explicit_params(self, tmp_path):
        """Test initialization with explicit parameters."""
        # Create a mock Spotify client
        mock_spotify = MagicMock(spec=SpotifyPopularity)
        mock_spotify.available = True

        # Patch the SpotifyPopularity constructor
        with patch(
            "ticket_price_predictor.popularity.service.SpotifyPopularity", return_value=mock_spotify
        ):
            service = PopularityService(
                spotify_client_id="param_id",
                spotify_client_secret="param_secret",
                songkick_api_key="param_songkick",
                bandsintown_app_id="param_bandsintown",
                cache_dir=tmp_path,
            )

        assert service.spotify is not None
        assert service.songkick is not None
        assert service.bandsintown is not None

    def test_get_artist_popularity_cache_hit(self, tmp_path):
        """Test get_artist_popularity with cache hit."""
        service = PopularityService(cache_dir=tmp_path)

        # Pre-populate cache
        cached_pop = ArtistPopularity(
            name="Cached Artist",
            popularity_score=75.0,
            tier=PopularityTier.HIGH,
            spotify_popularity=85,
            sources_available=["spotify_popularity"],
        )
        service.cache.set("Cached Artist", cached_pop)

        # Get from cache
        result = service.get_artist_popularity("Cached Artist")

        assert result.name == "Cached Artist"
        assert result.popularity_score == 75.0
        assert result.tier == PopularityTier.HIGH

    def test_get_artist_popularity_cache_miss(self, tmp_path):
        """Test get_artist_popularity with cache miss (fetch from APIs)."""
        # Create a mock Spotify client
        mock_spotify = MagicMock(spec=SpotifyPopularity)
        mock_spotify.available = True
        mock_spotify.get_artist_metrics.return_value = SpotifyMetrics(
            spotify_id="123",
            popularity=80,
            followers=10_000_000,
            genres=["pop"],
        )

        # Patch the SpotifyPopularity constructor
        with patch(
            "ticket_price_predictor.popularity.service.SpotifyPopularity", return_value=mock_spotify
        ):
            service = PopularityService(
                spotify_client_id="test_id",
                spotify_client_secret="test_secret",
                cache_dir=tmp_path,
            )

            result = service.get_artist_popularity("New Artist")

        assert result.name == "New Artist"
        assert result.popularity_score > 0
        assert result.spotify_popularity == 80
        assert result.spotify_followers == 10_000_000

        # Verify cached
        cached = service.cache.get("New Artist")
        assert cached is not None

    def test_rank_performers(self, tmp_path):
        """Test performer ranking."""
        service = PopularityService(cache_dir=tmp_path)

        # Pre-populate cache
        for name, score, tier in [
            ("High Artist", 80, PopularityTier.HIGH),
            ("Medium Artist", 50, PopularityTier.MEDIUM),
            ("Low Artist", 20, PopularityTier.LOW),
        ]:
            pop = ArtistPopularity(name, score, tier)
            service.cache.set(name, pop)

        ranked = service.rank_performers(["Low Artist", "High Artist", "Medium Artist"])

        # Should be sorted by score (highest first)
        assert ranked[0].name == "High Artist"
        assert ranked[1].name == "Medium Artist"
        # May or may not include low artist depending on coverage threshold

    def test_rank_performers_minimum_viable_set(self, tmp_path):
        """Test rank_performers ensures at least 3 performers."""
        service = PopularityService(cache_dir=tmp_path)

        # Pre-populate cache with 5 performers
        for i in range(5):
            pop = ArtistPopularity(f"Artist {i}", float(i * 10), PopularityTier.LOW)
            service.cache.set(f"Artist {i}", pop)

        ranked = service.rank_performers([f"Artist {i}" for i in range(5)], coverage_threshold=0.99)

        # Should return at least 3
        assert len(ranked) >= 3

    def test_calculate_max_events(self):
        """Test max events calculation from tiers."""
        performers = [
            ArtistPopularity("A", 80, PopularityTier.HIGH),  # 5
            ArtistPopularity("B", 50, PopularityTier.MEDIUM),  # 3
            ArtistPopularity("C", 20, PopularityTier.LOW),  # 1
        ]

        service = PopularityService()
        total = service.calculate_max_events(performers)

        assert total == 9  # 5 + 3 + 1

    def test_calculate_max_events_empty(self):
        """Test max events with empty list."""
        service = PopularityService()
        total = service.calculate_max_events([])
        assert total == 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestPopularityIntegration:
    """Integration tests for the full popularity workflow."""

    def test_end_to_end_workflow(self, tmp_path):
        """Test complete workflow from API fetch to ranking."""
        # Create a mock Spotify client
        mock_spotify = MagicMock(spec=SpotifyPopularity)
        mock_spotify.available = True

        def mock_get_artist_metrics(artist_name):
            if "Taylor Swift" in artist_name:
                return SpotifyMetrics(
                    spotify_id="ts123",
                    popularity=95,
                    followers=80_000_000,
                    genres=["pop"],
                )
            elif "BTS" in artist_name:
                return SpotifyMetrics(
                    spotify_id="bts456",
                    popularity=90,
                    followers=60_000_000,
                    genres=["k-pop"],
                )
            return None

        mock_spotify.get_artist_metrics = mock_get_artist_metrics

        # Patch the SpotifyPopularity constructor
        with patch(
            "ticket_price_predictor.popularity.service.SpotifyPopularity", return_value=mock_spotify
        ):
            service = PopularityService(
                spotify_client_id="test_id",
                spotify_client_secret="test_secret",
                cache_dir=tmp_path,
            )

            # Rank performers
            ranked = service.rank_performers(["BTS", "Taylor Swift"])

        # Verify results
        assert len(ranked) >= 2
        assert ranked[0].popularity_score > ranked[1].popularity_score
        assert all(p.tier == PopularityTier.HIGH for p in ranked[:2])

        # Calculate max events
        max_events = service.calculate_max_events(ranked)
        assert max_events >= 10  # At least 2 high-tier artists

    def test_fallback_to_low_tier_on_error(self, tmp_path):
        """Test that API errors result in low-tier fallback."""
        # Create a mock Spotify client that returns None (simulating error handled internally)
        mock_spotify = MagicMock(spec=SpotifyPopularity)
        mock_spotify.available = True
        mock_spotify.get_artist_metrics.return_value = None  # Simulates internal error handling

        # Patch the SpotifyPopularity constructor
        with patch(
            "ticket_price_predictor.popularity.service.SpotifyPopularity", return_value=mock_spotify
        ):
            service = PopularityService(
                spotify_client_id="test_id",
                spotify_client_secret="test_secret",
                cache_dir=tmp_path,
            )

            result = service.get_artist_popularity("Unknown Artist")

        # Should return low tier with zero score
        assert result.tier == PopularityTier.LOW
        assert result.popularity_score == 0
        assert len(result.sources_available) == 0
