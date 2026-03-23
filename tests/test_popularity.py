"""Tests for popularity module."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from ticket_price_predictor.popularity.aggregator import (
    ArtistPopularity,
    PopularityAggregator,
    PopularityTier,
)
from ticket_price_predictor.popularity.cache import PopularityCache
from ticket_price_predictor.popularity.lastfm import (
    LastfmMetrics,
    LastfmPopularity,
)
from ticket_price_predictor.popularity.service import PopularityService
from ticket_price_predictor.popularity.youtube import (
    YouTubeMetrics,
    YouTubePopularity,
)

# ============================================================================
# YouTubePopularity Tests
# ============================================================================


class TestYouTubePopularity:
    """Tests for YouTubePopularity."""

    def test_initialization_available(self):
        """Test initialization when ytmusicapi is available."""
        client = YouTubePopularity.__new__(YouTubePopularity)
        client._ytmusic = MagicMock()
        client._available = True

        assert client.available is True

    def test_initialization_unavailable(self):
        """Test initialization when ytmusicapi is not available."""
        client = YouTubePopularity.__new__(YouTubePopularity)
        client._ytmusic = None
        client._available = False

        assert client.available is False

    def test_get_artist_metrics_success(self):
        """Test successful artist lookup."""
        client = YouTubePopularity.__new__(YouTubePopularity)
        mock_ytmusic = MagicMock()

        # Mock search results
        mock_ytmusic.search.return_value = [{"browseId": "UC_abc123", "artist": "Taylor Swift"}]

        # Mock artist info
        mock_ytmusic.get_artist.return_value = {
            "name": "Taylor Swift",
            "subscribers": "120M subscribers",
            "views": "45,000,000,000 views",
        }

        client._ytmusic = mock_ytmusic
        client._available = True

        result = client.get_artist_metrics("Taylor Swift")

        assert result is not None
        assert isinstance(result, YouTubeMetrics)
        assert result.channel_id == "UC_abc123"
        assert result.name == "Taylor Swift"
        assert result.subscriber_count == 120_000_000
        assert result.view_count == 45_000_000_000

    def test_get_artist_metrics_not_found(self):
        """Test when artist is not found."""
        client = YouTubePopularity.__new__(YouTubePopularity)
        mock_ytmusic = MagicMock()
        mock_ytmusic.search.return_value = []

        client._ytmusic = mock_ytmusic
        client._available = True

        result = client.get_artist_metrics("Unknown Artist")
        assert result is None

    def test_get_artist_metrics_api_error(self):
        """Test graceful handling of API errors."""
        client = YouTubePopularity.__new__(YouTubePopularity)
        mock_ytmusic = MagicMock()
        mock_ytmusic.search.side_effect = Exception("API Error")

        client._ytmusic = mock_ytmusic
        client._available = True

        result = client.get_artist_metrics("Test Artist")
        assert result is None

    def test_get_artist_metrics_unavailable_client(self):
        """Test when client is not available."""
        client = YouTubePopularity.__new__(YouTubePopularity)
        client._ytmusic = None
        client._available = False

        result = client.get_artist_metrics("Test")
        assert result is None

    def test_parse_subscriber_count(self):
        """Test subscriber count parsing."""
        assert YouTubePopularity._parse_subscriber_count("120M subscribers") == 120_000_000
        assert YouTubePopularity._parse_subscriber_count("1.5M subscribers") == 1_500_000
        assert YouTubePopularity._parse_subscriber_count("500K subscribers") == 500_000
        assert YouTubePopularity._parse_subscriber_count("1.2B subscribers") == 1_200_000_000
        assert YouTubePopularity._parse_subscriber_count("invalid") == 0

    def test_parse_view_count(self):
        """Test view count parsing."""
        assert YouTubePopularity._parse_view_count("45,000,000,000 views") == 45_000_000_000
        assert YouTubePopularity._parse_view_count("1.5M views") == 1_500_000
        assert YouTubePopularity._parse_view_count("500K views") == 500_000
        assert YouTubePopularity._parse_view_count("invalid") == 0


# ============================================================================
# LastfmPopularity Tests
# ============================================================================


class TestLastfmPopularity:
    """Tests for LastfmPopularity."""

    def test_initialization(self):
        """Test initialization with API key."""
        client = LastfmPopularity("test_key")
        assert client.available is True
        assert client.api_key == "test_key"

    def test_initialization_missing_key(self):
        """Test initialization without API key."""
        client = LastfmPopularity("")
        assert client.available is False

    def test_get_artist_metrics_success(self):
        """Test successful artist lookup."""
        with patch("ticket_price_predictor.popularity.lastfm.httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "artist": {
                    "name": "Taylor Swift",
                    "stats": {
                        "listeners": "25000000",
                        "playcount": "500000000",
                    },
                    "tags": {
                        "tag": [
                            {"name": "pop"},
                            {"name": "country"},
                        ]
                    },
                }
            }
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = MagicMock()
            mock_client_instance.__enter__.return_value.get.return_value = mock_response
            mock_client.return_value = mock_client_instance

            client = LastfmPopularity("test_key")
            result = client.get_artist_metrics("Taylor Swift")

            assert result is not None
            assert isinstance(result, LastfmMetrics)
            assert result.name == "Taylor Swift"
            assert result.listener_count == 25_000_000
            assert result.play_count == 500_000_000
            assert result.tags == ["pop", "country"]

    def test_get_artist_metrics_not_found(self):
        """Test when artist is not found."""
        with patch("ticket_price_predictor.popularity.lastfm.httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {"artist": None}
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = MagicMock()
            mock_client_instance.__enter__.return_value.get.return_value = mock_response
            mock_client.return_value = mock_client_instance

            client = LastfmPopularity("test_key")
            result = client.get_artist_metrics("Unknown Artist")

            assert result is None

    def test_get_artist_metrics_error_response(self):
        """Test handling of error responses."""
        with patch("ticket_price_predictor.popularity.lastfm.httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "error": 6,
                "message": "Artist not found",
            }
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = MagicMock()
            mock_client_instance.__enter__.return_value.get.return_value = mock_response
            mock_client.return_value = mock_client_instance

            client = LastfmPopularity("test_key")
            result = client.get_artist_metrics("Test Artist")

            # No "artist" key in response -> None
            assert result is None

    def test_get_artist_metrics_unavailable(self):
        """Test when client is not available (no API key)."""
        client = LastfmPopularity("")
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
            youtube_subscribers=80_000_000,
            youtube_views=45_000_000_000,
            lastfm_listeners=25_000_000,
            lastfm_play_count=500_000_000,
        )
        assert result.tier == PopularityTier.HIGH
        assert result.popularity_score > 70
        assert len(result.sources_available) == 4
        assert "youtube_subscribers" in result.sources_available
        assert "youtube_views" in result.sources_available
        assert "lastfm_listeners" in result.sources_available
        assert "lastfm_play_count" in result.sources_available

    def test_calculate_score_youtube_only(self):
        """Test with only YouTube data."""
        aggregator = PopularityAggregator()
        result = aggregator.calculate_score(
            artist_name="Test Artist",
            youtube_subscribers=1_000_000,
            youtube_views=500_000_000,
        )
        assert result.popularity_score > 0
        assert "youtube_subscribers" in result.sources_available
        assert "youtube_views" in result.sources_available
        assert len(result.sources_available) == 2

    def test_calculate_score_lastfm_only(self):
        """Test with only Last.fm data."""
        aggregator = PopularityAggregator()
        result = aggregator.calculate_score(
            artist_name="Test Artist",
            lastfm_listeners=5_000_000,
            lastfm_play_count=100_000_000,
        )
        assert result.popularity_score > 0
        assert "lastfm_listeners" in result.sources_available
        assert "lastfm_play_count" in result.sources_available
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
        high = aggregator.calculate_score(
            "High",
            youtube_subscribers=80_000_000,
            youtube_views=40_000_000_000,
            lastfm_listeners=25_000_000,
            lastfm_play_count=400_000_000,
        )
        assert high.tier == PopularityTier.HIGH
        assert high.popularity_score >= 70

        # Low tier (< 40)
        low = aggregator.calculate_score(
            "Low",
            youtube_subscribers=1_000,
            youtube_views=50_000,
            lastfm_listeners=500,
            lastfm_play_count=2_000,
        )
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
            "youtube_subscribers": 80.0,
            "youtube_views": 70.0,
            "lastfm_listeners": 60.0,
            "lastfm_play_count": 50.0,
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

        # Only YouTube subscribers
        result1 = aggregator.calculate_score(
            artist_name="Test1",
            youtube_subscribers=1_000_000,
        )

        # YouTube subscribers + Last.fm listeners
        result2 = aggregator.calculate_score(
            artist_name="Test2",
            youtube_subscribers=1_000_000,
            lastfm_listeners=5_000_000,
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
            sources_available=["youtube_subscribers"],
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
        monkeypatch.setenv("LASTFM_API_KEY", "env_lastfm")

        # Mock YouTube client
        mock_youtube = MagicMock(spec=YouTubePopularity)
        mock_youtube.available = True

        with patch(
            "ticket_price_predictor.popularity.service.YouTubePopularity",
            return_value=mock_youtube,
        ):
            service = PopularityService(cache_dir=tmp_path)

        assert service.youtube is not None
        assert service.lastfm is not None

    def test_initialization_with_explicit_params(self, tmp_path):
        """Test initialization with explicit parameters."""
        mock_youtube = MagicMock(spec=YouTubePopularity)
        mock_youtube.available = True

        with patch(
            "ticket_price_predictor.popularity.service.YouTubePopularity",
            return_value=mock_youtube,
        ):
            service = PopularityService(
                lastfm_api_key="param_lastfm",
                cache_dir=tmp_path,
            )

        assert service.youtube is not None
        assert service.lastfm is not None

    def test_get_artist_popularity_cache_hit(self, tmp_path):
        """Test get_artist_popularity with cache hit."""
        service = PopularityService(cache_dir=tmp_path)

        # Pre-populate cache
        cached_pop = ArtistPopularity(
            name="Cached Artist",
            popularity_score=75.0,
            tier=PopularityTier.HIGH,
            youtube_subscribers=50_000_000,
            sources_available=["youtube_subscribers"],
        )
        service.cache.set("Cached Artist", cached_pop)

        # Get from cache
        result = service.get_artist_popularity("Cached Artist")

        assert result.name == "Cached Artist"
        assert result.popularity_score == 75.0
        assert result.tier == PopularityTier.HIGH

    def test_get_artist_popularity_cache_miss(self, tmp_path):
        """Test get_artist_popularity with cache miss (fetch from APIs)."""
        mock_youtube = MagicMock(spec=YouTubePopularity)
        mock_youtube.available = True
        mock_youtube.get_artist_metrics.return_value = YouTubeMetrics(
            channel_id="UC_abc",
            name="New Artist",
            subscriber_count=10_000_000,
            view_count=5_000_000_000,
        )

        mock_lastfm = MagicMock(spec=LastfmPopularity)
        mock_lastfm.available = True
        mock_lastfm.get_artist_metrics.return_value = LastfmMetrics(
            name="New Artist",
            listener_count=5_000_000,
            play_count=100_000_000,
            tags=["pop"],
        )

        with patch(
            "ticket_price_predictor.popularity.service.YouTubePopularity",
            return_value=mock_youtube,
        ):
            service = PopularityService(
                lastfm_api_key="test_key",
                cache_dir=tmp_path,
            )
            # Manually set lastfm mock (env var fallback creates real one)
            service.lastfm = mock_lastfm

            result = service.get_artist_popularity("New Artist")

        assert result.name == "New Artist"
        assert result.popularity_score > 0
        assert result.youtube_subscribers == 10_000_000
        assert result.youtube_views == 5_000_000_000
        assert result.lastfm_listeners == 5_000_000
        assert result.lastfm_play_count == 100_000_000

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
        mock_youtube = MagicMock(spec=YouTubePopularity)
        mock_youtube.available = True

        def mock_yt_metrics(artist_name):
            if "Taylor Swift" in artist_name:
                return YouTubeMetrics(
                    channel_id="UC_ts",
                    name="Taylor Swift",
                    subscriber_count=80_000_000,
                    view_count=40_000_000_000,
                )
            elif "BTS" in artist_name:
                return YouTubeMetrics(
                    channel_id="UC_bts",
                    name="BTS",
                    subscriber_count=75_000_000,
                    view_count=35_000_000_000,
                )
            return None

        mock_youtube.get_artist_metrics = mock_yt_metrics

        mock_lastfm = MagicMock(spec=LastfmPopularity)
        mock_lastfm.available = True

        def mock_lfm_metrics(artist_name):
            if "Taylor Swift" in artist_name:
                return LastfmMetrics(
                    name="Taylor Swift",
                    listener_count=25_000_000,
                    play_count=500_000_000,
                    tags=["pop"],
                )
            elif "BTS" in artist_name:
                return LastfmMetrics(
                    name="BTS",
                    listener_count=15_000_000,
                    play_count=300_000_000,
                    tags=["k-pop"],
                )
            return None

        mock_lastfm.get_artist_metrics = mock_lfm_metrics

        with patch(
            "ticket_price_predictor.popularity.service.YouTubePopularity",
            return_value=mock_youtube,
        ):
            service = PopularityService(
                lastfm_api_key="test_key",
                cache_dir=tmp_path,
            )
            service.lastfm = mock_lastfm

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
        mock_youtube = MagicMock(spec=YouTubePopularity)
        mock_youtube.available = True
        mock_youtube.get_artist_metrics.return_value = None

        with patch(
            "ticket_price_predictor.popularity.service.YouTubePopularity",
            return_value=mock_youtube,
        ):
            service = PopularityService(cache_dir=tmp_path)

            result = service.get_artist_popularity("Unknown Artist")

        # Should return low tier with zero score
        assert result.tier == PopularityTier.LOW
        assert result.popularity_score == 0
        assert len(result.sources_available) == 0


# ============================================================================
# LastfmPopularity Discovery Tests
# ============================================================================


class TestLastfmGetTopArtists:
    """Tests for LastfmPopularity.get_top_artists."""

    def test_returns_list_of_strings_on_success(self):
        """get_top_artists returns a list of artist name strings."""
        from unittest.mock import MagicMock, patch

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "artists": {
                "artist": [
                    {"name": "Taylor Swift"},
                    {"name": "Drake"},
                    {"name": "Bad Bunny"},
                ]
            }
        }
        mock_response.raise_for_status.return_value = None

        mock_client = MagicMock()
        mock_client.__enter__ = lambda _: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response

        with patch(
            "ticket_price_predictor.popularity.lastfm.httpx.Client", return_value=mock_client
        ):
            client = LastfmPopularity(api_key="fake_key")
            result = client.get_top_artists(limit=3)

        assert isinstance(result, list)
        assert result == ["Taylor Swift", "Drake", "Bad Bunny"]

    def test_returns_empty_list_on_http_error(self):
        """get_top_artists returns [] when the HTTP call raises."""
        from unittest.mock import MagicMock, patch

        mock_client = MagicMock()
        mock_client.__enter__ = lambda _: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = Exception("connection refused")

        with patch(
            "ticket_price_predictor.popularity.lastfm.httpx.Client", return_value=mock_client
        ):
            client = LastfmPopularity(api_key="fake_key")
            result = client.get_top_artists()

        assert result == []

    def test_returns_empty_list_when_unavailable(self):
        """get_top_artists returns [] immediately when api_key is empty."""
        client = LastfmPopularity(api_key="")
        result = client.get_top_artists()
        assert result == []


class TestLastfmGetTopArtistsByTag:
    """Tests for LastfmPopularity.get_top_artists_by_tag."""

    def test_returns_list_of_strings_on_success(self):
        """get_top_artists_by_tag returns a list of artist name strings for a tag."""
        from unittest.mock import MagicMock, patch

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "topartists": {
                "artist": [
                    {"name": "Dave Chappelle"},
                    {"name": "Kevin Hart"},
                ]
            }
        }
        mock_response.raise_for_status.return_value = None

        mock_client = MagicMock()
        mock_client.__enter__ = lambda _: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response

        with patch(
            "ticket_price_predictor.popularity.lastfm.httpx.Client", return_value=mock_client
        ):
            client = LastfmPopularity(api_key="fake_key")
            result = client.get_top_artists_by_tag("comedy", limit=2)

        assert isinstance(result, list)
        assert result == ["Dave Chappelle", "Kevin Hart"]

    def test_returns_empty_list_on_http_error(self):
        """get_top_artists_by_tag returns [] when the HTTP call raises."""
        from unittest.mock import MagicMock, patch

        mock_client = MagicMock()
        mock_client.__enter__ = lambda _: mock_client
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = Exception("timeout")

        with patch(
            "ticket_price_predictor.popularity.lastfm.httpx.Client", return_value=mock_client
        ):
            client = LastfmPopularity(api_key="fake_key")
            result = client.get_top_artists_by_tag("pop")

        assert result == []

    def test_returns_empty_list_when_unavailable(self):
        """get_top_artists_by_tag returns [] immediately when api_key is empty."""
        client = LastfmPopularity(api_key="")
        result = client.get_top_artists_by_tag("rock")
        assert result == []
