"""Tests for normalization module."""

import pytest

from ticket_price_predictor.normalization import SeatZoneMapper
from ticket_price_predictor.schemas import SeatZone


class TestSeatZoneMapper:
    """Tests for SeatZoneMapper."""

    @pytest.fixture
    def mapper(self) -> SeatZoneMapper:
        """Create a mapper instance."""
        return SeatZoneMapper()

    def test_map_price_range_basic(self, mapper: SeatZoneMapper):
        """Test basic price range mapping."""
        zone_prices = mapper.map_price_range_to_zones(50.0, 200.0)

        # Floor/VIP should be close to max
        assert zone_prices[SeatZone.FLOOR_VIP] == 200.0

        # Balcony should be close to min
        assert zone_prices[SeatZone.BALCONY] == 87.5  # 50 + (200-50) * 0.25

        # Lower tier should be in between
        assert zone_prices[SeatZone.LOWER_TIER] == 155.0  # 50 + (200-50) * 0.70

        # Upper tier should be in between
        assert zone_prices[SeatZone.UPPER_TIER] == 117.5  # 50 + (200-50) * 0.45

    def test_map_price_range_same_min_max(self, mapper: SeatZoneMapper):
        """Test mapping when min equals max."""
        zone_prices = mapper.map_price_range_to_zones(100.0, 100.0)

        # All zones should have the same price
        assert zone_prices[SeatZone.FLOOR_VIP] == 100.0
        assert zone_prices[SeatZone.BALCONY] == 100.0

    def test_map_price_range_inverted(self, mapper: SeatZoneMapper):
        """Test that inverted min/max is handled."""
        zone_prices = mapper.map_price_range_to_zones(200.0, 50.0)

        # Should swap and work correctly
        assert zone_prices[SeatZone.FLOOR_VIP] == 200.0

    def test_normalize_floor_keywords(self, mapper: SeatZoneMapper):
        """Test normalizing floor/VIP section names."""
        assert mapper.normalize_zone_name("Floor Section") == SeatZone.FLOOR_VIP
        assert mapper.normalize_zone_name("VIP Box") == SeatZone.FLOOR_VIP
        assert mapper.normalize_zone_name("Pit") == SeatZone.FLOOR_VIP
        assert mapper.normalize_zone_name("Courtside") == SeatZone.FLOOR_VIP
        assert mapper.normalize_zone_name("Premium Seats") == SeatZone.FLOOR_VIP

    def test_normalize_lower_keywords(self, mapper: SeatZoneMapper):
        """Test normalizing lower tier section names."""
        assert mapper.normalize_zone_name("Lower Bowl") == SeatZone.LOWER_TIER
        assert mapper.normalize_zone_name("Section 100") == SeatZone.LOWER_TIER
        assert mapper.normalize_zone_name("Orchestra") == SeatZone.LOWER_TIER
        assert mapper.normalize_zone_name("Loge") == SeatZone.LOWER_TIER

    def test_normalize_upper_keywords(self, mapper: SeatZoneMapper):
        """Test normalizing upper tier section names."""
        assert mapper.normalize_zone_name("Upper Deck") == SeatZone.UPPER_TIER
        assert mapper.normalize_zone_name("Section 200") == SeatZone.UPPER_TIER
        assert mapper.normalize_zone_name("Mezzanine") == SeatZone.UPPER_TIER
        assert mapper.normalize_zone_name("Section 300") == SeatZone.UPPER_TIER

    def test_normalize_balcony_keywords(self, mapper: SeatZoneMapper):
        """Test normalizing balcony section names."""
        assert mapper.normalize_zone_name("Balcony") == SeatZone.BALCONY
        assert mapper.normalize_zone_name("Section 400") == SeatZone.BALCONY
        assert mapper.normalize_zone_name("Nosebleed Section") == SeatZone.BALCONY
        assert mapper.normalize_zone_name("Limited View") == SeatZone.BALCONY

    def test_normalize_section_number_pattern(self, mapper: SeatZoneMapper):
        """Test normalizing by section number."""
        assert mapper.normalize_zone_name("Section 50") == SeatZone.FLOOR_VIP
        assert mapper.normalize_zone_name("Section 120") == SeatZone.LOWER_TIER
        assert mapper.normalize_zone_name("Section 250") == SeatZone.UPPER_TIER
        assert mapper.normalize_zone_name("Section 450") == SeatZone.BALCONY

    def test_normalize_unknown_defaults_to_upper(self, mapper: SeatZoneMapper):
        """Test that unknown sections default to upper tier."""
        assert mapper.normalize_zone_name("Random Area") == SeatZone.UPPER_TIER
        assert mapper.normalize_zone_name("XYZ123") == SeatZone.UPPER_TIER

    def test_get_zone_price_ratio(self, mapper: SeatZoneMapper):
        """Test getting price ratios."""
        assert mapper.get_zone_price_ratio(SeatZone.FLOOR_VIP) == 1.0
        assert mapper.get_zone_price_ratio(SeatZone.LOWER_TIER) == 0.70
        assert mapper.get_zone_price_ratio(SeatZone.UPPER_TIER) == 0.45
        assert mapper.get_zone_price_ratio(SeatZone.BALCONY) == 0.25

    def test_custom_ratios(self):
        """Test using custom price ratios."""
        custom_ratios = {
            SeatZone.FLOOR_VIP: 1.0,
            SeatZone.LOWER_TIER: 0.80,
            SeatZone.UPPER_TIER: 0.50,
            SeatZone.BALCONY: 0.30,
        }
        mapper = SeatZoneMapper(price_ratios=custom_ratios)

        zone_prices = mapper.map_price_range_to_zones(0.0, 100.0)
        assert zone_prices[SeatZone.LOWER_TIER] == 80.0
        assert zone_prices[SeatZone.UPPER_TIER] == 50.0
        assert zone_prices[SeatZone.BALCONY] == 30.0
