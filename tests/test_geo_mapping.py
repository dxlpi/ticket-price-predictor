"""Tests for geo mapping module."""

from ticket_price_predictor.ml.features.geo_mapping import (
    CITY_COUNTRY_MAP,
    get_country,
    get_region_key,
)


class TestGetCountry:
    """Tests for get_country function."""

    def test_us_city(self):
        """Test US city returns US."""
        assert get_country("new york") == "US"
        assert get_country("los angeles") == "US"
        assert get_country("chicago") == "US"

    def test_case_insensitive(self):
        """Test case insensitive lookup."""
        assert get_country("New York") == "US"
        assert get_country("NEW YORK") == "US"
        assert get_country("Bangkok") == "TH"

    def test_international_cities(self):
        """Test international city mappings."""
        assert get_country("bangkok") == "TH"
        assert get_country("seoul") == "KR"
        assert get_country("london") == "GB"
        assert get_country("tokyo") == "JP"
        assert get_country("paris") == "FR"

    def test_unknown_city_defaults_to_us(self):
        """Test unknown city defaults to US."""
        assert get_country("some random city") == "US"
        assert get_country("") == "US"

    def test_whitespace_stripped(self):
        """Test whitespace is stripped."""
        assert get_country("  new york  ") == "US"
        assert get_country("  bangkok  ") == "TH"

    def test_all_tier1_cities_mapped(self):
        """Test all tier 1 cities are in the map."""
        tier1 = [
            "new york",
            "los angeles",
            "chicago",
            "houston",
            "phoenix",
            "philadelphia",
            "san antonio",
            "san diego",
            "dallas",
            "san jose",
            "las vegas",
            "miami",
            "boston",
            "atlanta",
            "san francisco",
        ]
        for city in tier1:
            assert get_country(city) == "US", f"{city} not mapped to US"

    def test_all_tier2_cities_mapped(self):
        """Test all tier 2 cities are in the map."""
        tier2 = [
            "seattle",
            "denver",
            "washington",
            "nashville",
            "austin",
            "detroit",
            "portland",
            "charlotte",
            "orlando",
            "minneapolis",
            "tampa",
            "pittsburgh",
            "st. louis",
            "baltimore",
            "salt lake city",
        ]
        for city in tier2:
            assert get_country(city) == "US", f"{city} not mapped to US"


class TestGetRegionKey:
    """Tests for get_region_key function."""

    def test_us_city(self):
        """Test US city region key."""
        assert get_region_key("new york") == "US:new york"
        assert get_region_key("Los Angeles") == "US:los angeles"

    def test_international_city(self):
        """Test international city region key."""
        assert get_region_key("bangkok") == "TH:bangkok"
        assert get_region_key("Seoul") == "KR:seoul"

    def test_unknown_city(self):
        """Test unknown city gets US prefix."""
        assert get_region_key("random town") == "US:random town"


class TestCityCountryMap:
    """Tests for the CITY_COUNTRY_MAP dictionary."""

    def test_map_not_empty(self):
        """Test map has entries."""
        assert len(CITY_COUNTRY_MAP) >= 50  # 30 US + 20 international

    def test_all_keys_lowercase(self):
        """Test all keys are lowercase."""
        for key in CITY_COUNTRY_MAP:
            assert key == key.lower(), f"Key '{key}' is not lowercase"

    def test_all_values_are_valid_country_codes(self):
        """Test all values are 2-letter codes."""
        for city, code in CITY_COUNTRY_MAP.items():
            assert len(code) == 2, f"Country code for '{city}' is not 2 chars: {code}"
            assert code == code.upper(), f"Country code for '{city}' is not uppercase: {code}"
