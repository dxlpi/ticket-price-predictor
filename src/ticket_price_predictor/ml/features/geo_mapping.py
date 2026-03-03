"""Static city-to-country mapping for regional popularity features.

This module provides geographic lookup utilities for the ML feature pipeline,
mapping cities to ISO 3166-1 alpha-2 country codes and generating composite
region keys for feature extraction.
"""

CITY_COUNTRY_MAP = {
    # US Cities - Tier 1 (Major metros)
    "new york": "US",
    "los angeles": "US",
    "chicago": "US",
    "houston": "US",
    "phoenix": "US",
    "philadelphia": "US",
    "san antonio": "US",
    "san diego": "US",
    "dallas": "US",
    "san jose": "US",
    "las vegas": "US",
    "miami": "US",
    "boston": "US",
    "atlanta": "US",
    "san francisco": "US",
    # US Cities - Tier 2 (Secondary markets)
    "seattle": "US",
    "denver": "US",
    "washington": "US",
    "nashville": "US",
    "austin": "US",
    "detroit": "US",
    "portland": "US",
    "charlotte": "US",
    "orlando": "US",
    "minneapolis": "US",
    "tampa": "US",
    "pittsburgh": "US",
    "st. louis": "US",
    "baltimore": "US",
    "salt lake city": "US",
    # US Cities - Tier 3 (Additional markets from dataset)
    "inglewood": "US",
    "glendale": "US",
    "oakland": "US",
    "fort worth": "US",
    "el paso": "US",
    "arlington": "US",
    "tuscaloosa": "US",
    "thackerville": "US",
    "uncasville": "US",
    "indio": "US",
    "napa": "US",
    "fort lauderdale": "US",
    # International Cities
    "bangkok": "TH",
    "seoul": "KR",
    "london": "GB",
    "tokyo": "JP",
    "paris": "FR",
    "berlin": "DE",
    "sydney": "AU",
    "toronto": "CA",
    "mexico city": "MX",
    "são paulo": "BR",
    "mumbai": "IN",
    "singapore": "SG",
    "manila": "PH",
    "jakarta": "ID",
    "ho chi minh city": "VN",
    "taipei": "TW",
    "hong kong": "HK",
    "kuala lumpur": "MY",
    "dubai": "AE",
    "amsterdam": "NL",
    "dusseldorf": "DE",
    "glasgow": "GB",
    "manchester": "GB",
}

# US state abbreviations for normalizing "City, STATE" format
_US_STATE_ABBREVS = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "DC",
}


def _normalize_city(city: str) -> str:
    """Normalize city name by lowercasing, stripping, and removing state suffix.

    Handles formats like "Las Vegas, NV" -> "las vegas".

    Args:
        city: Raw city name

    Returns:
        Normalized city name for lookup
    """
    normalized = city.lower().strip()
    # Strip "City, STATE" format (e.g., "Las Vegas, NV" -> "las vegas")
    if ", " in normalized:
        parts = normalized.rsplit(", ", 1)
        suffix = parts[1].strip().upper()
        if suffix in _US_STATE_ABBREVS:
            normalized = parts[0].strip()
    return normalized


def get_country(city: str) -> str:
    """Get ISO 3166-1 alpha-2 country code for a city.

    Args:
        city: City name (case-insensitive, will be normalized).
              Handles "City, STATE" format (e.g., "Las Vegas, NV").

    Returns:
        Two-letter country code (e.g., "US", "GB", "JP"). Defaults to "US"
        if city is not found in the mapping.

    Examples:
        >>> get_country("New York")
        'US'
        >>> get_country("TOKYO")
        'JP'
        >>> get_country("Las Vegas, NV")
        'US'
        >>> get_country("Unknown City")
        'US'
    """
    normalized_city = _normalize_city(city)
    return CITY_COUNTRY_MAP.get(normalized_city, "US")


def get_region_key(city: str) -> str:
    """Generate composite region key for geographic feature extraction.

    Args:
        city: City name (case-insensitive, will be normalized).
              Handles "City, STATE" format (e.g., "Las Vegas, NV").

    Returns:
        Composite key in format "{country}:{city_lower}" (e.g., "US:new york",
        "JP:tokyo"). Used for region-specific feature aggregation.

    Examples:
        >>> get_region_key("New York")
        'US:new york'
        >>> get_region_key("  TOKYO  ")
        'JP:tokyo'
        >>> get_region_key("Las Vegas, NV")
        'US:las vegas'
        >>> get_region_key("Unknown City")
        'US:unknown city'
    """
    normalized_city = _normalize_city(city)
    country = get_country(city)
    return f"{country}:{normalized_city}"
