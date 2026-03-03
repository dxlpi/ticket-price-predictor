"""Tests for cold start handler."""

from unittest.mock import MagicMock

from ticket_price_predictor.ml.inference.cold_start import ColdStartEstimate, ColdStartHandler
from ticket_price_predictor.normalization.seat_zones import SeatZone
from ticket_price_predictor.popularity.aggregator import ArtistPopularity, PopularityTier

# ============================================================================
# ColdStartEstimate Tests
# ============================================================================


class TestColdStartEstimate:
    """Tests for ColdStartEstimate dataclass."""

    def test_creation(self):
        """Test creating a cold start estimate."""
        estimate = ColdStartEstimate(
            prices_by_zone={
                SeatZone.FLOOR_VIP: 500.0,
                SeatZone.LOWER_TIER: 250.0,
            },
            confidence=0.7,
            source="performer_history",
        )

        assert estimate.prices_by_zone[SeatZone.FLOOR_VIP] == 500.0
        assert estimate.confidence == 0.7
        assert estimate.source == "performer_history"


# ============================================================================
# ColdStartHandler Tests - Basic Functionality
# ============================================================================


class TestColdStartHandler:
    """Tests for ColdStartHandler basic functionality."""

    def test_default_zone_prices(self):
        """Test default zone prices are set."""
        handler = ColdStartHandler()
        defaults = handler.GLOBAL_ZONE_DEFAULTS

        assert SeatZone.FLOOR_VIP in defaults
        assert SeatZone.LOWER_TIER in defaults
        assert SeatZone.UPPER_TIER in defaults
        assert SeatZone.BALCONY in defaults

        # VIP should be most expensive
        assert defaults[SeatZone.FLOOR_VIP] > defaults[SeatZone.LOWER_TIER]
        assert defaults[SeatZone.LOWER_TIER] > defaults[SeatZone.UPPER_TIER]
        assert defaults[SeatZone.UPPER_TIER] > defaults[SeatZone.BALCONY]

    def test_get_estimate_with_historical_prices(self):
        """Test estimate uses historical prices when available."""
        historical = {
            "taylor swift": {
                SeatZone.FLOOR_VIP: 1000.0,
                SeatZone.LOWER_TIER: 500.0,
            }
        }

        handler = ColdStartHandler(historical_prices=historical)
        estimate = handler.get_estimate("Taylor Swift")

        assert estimate.source == "performer_history"
        assert estimate.confidence == 0.7
        assert estimate.prices_by_zone[SeatZone.FLOOR_VIP] == 1000.0

    def test_get_estimate_falls_back_to_heuristic(self):
        """Test estimate uses heuristic when no historical data."""
        handler = ColdStartHandler()
        estimate = handler.get_estimate("Unknown Artist")

        assert "heuristic" in estimate.source
        assert estimate.confidence < 0.7

    def test_update_historical_prices(self):
        """Test updating historical prices."""
        handler = ColdStartHandler()

        # Initially no history
        estimate1 = handler.get_estimate("New Artist")
        assert "heuristic" in estimate1.source

        # Add history
        handler.update_historical_prices(
            "New Artist", {SeatZone.FLOOR_VIP: 600.0, SeatZone.LOWER_TIER: 300.0}
        )

        # Should now use history
        estimate2 = handler.get_estimate("New Artist")
        assert estimate2.source == "performer_history"
        assert estimate2.prices_by_zone[SeatZone.FLOOR_VIP] == 600.0


# ============================================================================
# ColdStartHandler Tests - Keyword Detection (Fallback)
# ============================================================================


class TestColdStartHandlerKeywordDetection:
    """Tests for keyword-based category detection."""

    def test_kpop_detection(self):
        """Test K-pop artists get high multiplier."""
        handler = ColdStartHandler()

        for artist in ["BTS", "BLACKPINK", "Twice", "Stray Kids"]:
            estimate = handler.get_estimate(artist)
            # K-pop should have highest prices
            assert estimate.prices_by_zone[SeatZone.FLOOR_VIP] > 1000.0

    def test_major_artist_detection(self):
        """Test major artists get appropriate multiplier."""
        handler = ColdStartHandler()

        for artist in ["Taylor Swift", "Beyonce", "Ed Sheeran"]:
            estimate = handler.get_estimate(artist)
            assert estimate.prices_by_zone[SeatZone.FLOOR_VIP] > 800.0

    def test_country_artist_detection(self):
        """Test country artists get appropriate multiplier."""
        handler = ColdStartHandler()

        for artist in ["Morgan Wallen", "Luke Combs", "Zach Bryan"]:
            estimate = handler.get_estimate(artist)
            # Country artists should have moderate pricing
            floor_price = estimate.prices_by_zone[SeatZone.FLOOR_VIP]
            assert 600.0 < floor_price < 900.0

    def test_unknown_artist_default(self):
        """Test unknown artists get default multiplier."""
        handler = ColdStartHandler()
        estimate = handler.get_estimate("Local Band Nobody Knows")

        # Should use default multiplier (1.0)
        assert estimate.prices_by_zone[SeatZone.FLOOR_VIP] == 450.0

    def test_case_insensitive(self):
        """Test artist detection is case insensitive."""
        handler = ColdStartHandler()

        estimate1 = handler.get_estimate("BTS")
        estimate2 = handler.get_estimate("bts")
        estimate3 = handler.get_estimate("Bts")

        assert estimate1.prices_by_zone == estimate2.prices_by_zone
        assert estimate2.prices_by_zone == estimate3.prices_by_zone


# ============================================================================
# ColdStartHandler Tests - City and Event Adjustments
# ============================================================================


class TestColdStartHandlerAdjustments:
    """Tests for city tier and event type adjustments."""

    def test_city_tier_adjustment(self):
        """Test city tier affects prices."""
        handler = ColdStartHandler()

        # Same artist, different city tiers
        estimate_tier1 = handler.get_estimate("Unknown Artist", city_tier=1)
        estimate_tier2 = handler.get_estimate("Unknown Artist", city_tier=2)
        estimate_tier3 = handler.get_estimate("Unknown Artist", city_tier=3)

        # Tier 1 cities should have highest prices
        assert (
            estimate_tier1.prices_by_zone[SeatZone.FLOOR_VIP]
            > estimate_tier2.prices_by_zone[SeatZone.FLOOR_VIP]
        )
        assert (
            estimate_tier2.prices_by_zone[SeatZone.FLOOR_VIP]
            > estimate_tier3.prices_by_zone[SeatZone.FLOOR_VIP]
        )

    def test_event_type_adjustment(self):
        """Test event type affects prices."""
        handler = ColdStartHandler()

        concert = handler.get_estimate("Artist", event_type="CONCERT")
        sports = handler.get_estimate("Artist", event_type="SPORTS")
        theater = handler.get_estimate("Artist", event_type="THEATER")

        # Theater should be highest, sports lowest
        assert (
            theater.prices_by_zone[SeatZone.FLOOR_VIP] > concert.prices_by_zone[SeatZone.FLOOR_VIP]
        )
        assert (
            concert.prices_by_zone[SeatZone.FLOOR_VIP] > sports.prices_by_zone[SeatZone.FLOOR_VIP]
        )

    def test_combined_adjustments(self):
        """Test combined city and event adjustments."""
        handler = ColdStartHandler()

        # Premium scenario: K-pop in tier 1 city
        premium = handler.get_estimate("BTS", event_type="CONCERT", city_tier=1)

        # Budget scenario: unknown artist in tier 3 city
        budget = handler.get_estimate("Unknown", event_type="SPORTS", city_tier=3)

        # Premium should be much higher
        assert (
            premium.prices_by_zone[SeatZone.FLOOR_VIP]
            > budget.prices_by_zone[SeatZone.FLOOR_VIP] * 2
        )


# ============================================================================
# ColdStartHandler Tests - PopularityService Integration
# ============================================================================


class TestColdStartHandlerWithPopularityService:
    """Tests for PopularityService integration."""

    def test_uses_popularity_service_when_available(self):
        """Test handler uses PopularityService for tier detection."""
        # Create mock PopularityService
        mock_service = MagicMock()
        mock_service.get_artist_popularity.return_value = ArtistPopularity(
            name="New Artist",
            popularity_score=85.0,
            tier=PopularityTier.HIGH,
            sources_available=["youtube_subscribers"],
        )

        handler = ColdStartHandler(popularity_service=mock_service)
        estimate = handler.get_estimate("New Artist")

        # Should use popularity service
        assert "popularity_service" in estimate.source
        mock_service.get_artist_popularity.assert_called_once_with("New Artist")

    def test_high_tier_gets_high_multiplier(self):
        """Test HIGH tier from PopularityService gets 2.5x multiplier."""
        mock_service = MagicMock()
        mock_service.get_artist_popularity.return_value = ArtistPopularity(
            name="Popular Artist",
            popularity_score=90.0,
            tier=PopularityTier.HIGH,
        )

        handler = ColdStartHandler(popularity_service=mock_service)
        estimate = handler.get_estimate("Popular Artist")

        # HIGH tier = 2.5x multiplier
        # Base FLOOR_VIP is 450, so 450 * 2.5 = 1125
        assert estimate.prices_by_zone[SeatZone.FLOOR_VIP] == 1125.0

    def test_medium_tier_gets_medium_multiplier(self):
        """Test MEDIUM tier from PopularityService gets 1.5x multiplier."""
        mock_service = MagicMock()
        mock_service.get_artist_popularity.return_value = ArtistPopularity(
            name="Medium Artist",
            popularity_score=55.0,
            tier=PopularityTier.MEDIUM,
        )

        handler = ColdStartHandler(popularity_service=mock_service)
        estimate = handler.get_estimate("Medium Artist")

        # MEDIUM tier = 1.5x multiplier
        assert estimate.prices_by_zone[SeatZone.FLOOR_VIP] == 675.0

    def test_low_tier_gets_base_multiplier(self):
        """Test LOW tier from PopularityService gets 1.0x multiplier."""
        mock_service = MagicMock()
        mock_service.get_artist_popularity.return_value = ArtistPopularity(
            name="New Artist",
            popularity_score=20.0,
            tier=PopularityTier.LOW,
        )

        handler = ColdStartHandler(popularity_service=mock_service)
        estimate = handler.get_estimate("New Artist")

        # LOW tier = 1.0x multiplier
        assert estimate.prices_by_zone[SeatZone.FLOOR_VIP] == 450.0

    def test_falls_back_to_keywords_on_service_error(self):
        """Test handler falls back to keywords when PopularityService fails."""
        mock_service = MagicMock()
        mock_service.get_artist_popularity.side_effect = Exception("API Error")

        handler = ColdStartHandler(popularity_service=mock_service)

        # BTS should still get K-pop pricing via keyword fallback
        estimate = handler.get_estimate("BTS")

        assert "keyword_fallback" in estimate.source
        # K-pop multiplier is 2.5x
        assert estimate.prices_by_zone[SeatZone.FLOOR_VIP] == 1125.0

    def test_higher_confidence_with_popularity_service(self):
        """Test estimates have higher confidence when using PopularityService."""
        mock_service = MagicMock()
        mock_service.get_artist_popularity.return_value = ArtistPopularity(
            name="Artist",
            popularity_score=50.0,
            tier=PopularityTier.MEDIUM,
        )

        handler_with = ColdStartHandler(popularity_service=mock_service)
        handler_without = ColdStartHandler()

        estimate_with = handler_with.get_estimate("Artist")
        estimate_without = handler_without.get_estimate("Artist")

        assert estimate_with.confidence > estimate_without.confidence

    def test_no_service_uses_keywords(self):
        """Test handler uses keywords when no PopularityService provided."""
        handler = ColdStartHandler(popularity_service=None)
        estimate = handler.get_estimate("Taylor Swift")

        # Should use keyword fallback for known major artist
        assert "keyword_fallback" in estimate.source


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestColdStartHandlerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_artist_name(self):
        """Test handling of empty artist name."""
        handler = ColdStartHandler()
        estimate = handler.get_estimate("")

        # Should use default pricing
        assert estimate.prices_by_zone[SeatZone.FLOOR_VIP] == 450.0

    def test_whitespace_artist_name(self):
        """Test handling of whitespace artist name."""
        handler = ColdStartHandler()
        estimate = handler.get_estimate("   ")

        # Should use default pricing
        assert estimate.prices_by_zone[SeatZone.FLOOR_VIP] == 450.0

    def test_artist_name_with_special_characters(self):
        """Test handling of artist names with special characters."""
        handler = ColdStartHandler()
        estimate = handler.get_estimate("AC/DC & Friends!")

        # Should handle gracefully
        assert estimate is not None
        assert SeatZone.FLOOR_VIP in estimate.prices_by_zone

    def test_invalid_city_tier(self):
        """Test handling of invalid city tier."""
        handler = ColdStartHandler()
        estimate = handler.get_estimate("Artist", city_tier=99)

        # Should fall back to default tier
        assert estimate is not None

    def test_invalid_event_type(self):
        """Test handling of invalid event type."""
        handler = ColdStartHandler()
        estimate = handler.get_estimate("Artist", event_type="UNKNOWN")

        # Should fall back to default
        assert estimate is not None

    def test_all_zones_have_prices(self):
        """Test all seat zones have prices in estimate."""
        handler = ColdStartHandler()
        estimate = handler.get_estimate("Any Artist")

        for zone in SeatZone:
            assert zone in estimate.prices_by_zone
            assert estimate.prices_by_zone[zone] > 0
