"""Cold-start handling for new events and performers."""

from dataclasses import dataclass

from ticket_price_predictor.normalization.seat_zones import SeatZone


@dataclass
class ColdStartEstimate:
    """Estimated prices for cold-start scenarios."""

    prices_by_zone: dict[SeatZone, float]
    confidence: float  # 0-1, lower for cold-start
    source: str  # "cluster", "performer_history", "heuristic", "global_default"


class ColdStartHandler:
    """Handle predictions for new events and performers."""

    # Global default prices by zone (based on typical market data)
    GLOBAL_ZONE_DEFAULTS = {
        SeatZone.FLOOR_VIP: 450.0,
        SeatZone.LOWER_TIER: 250.0,
        SeatZone.UPPER_TIER: 150.0,
        SeatZone.BALCONY: 75.0,
    }

    # Demand multipliers by artist category
    DEMAND_MULTIPLIERS = {
        "kpop": 2.5,
        "major": 2.0,
        "country": 1.5,
        "default": 1.0,
    }

    # K-pop keywords for detection
    KPOP_KEYWORDS = frozenset([
        "blackpink", "bts", "twice", "stray kids", "aespa", "newjeans",
        "seventeen", "nct", "exo", "red velvet", "itzy", "ive", "le sserafim",
    ])

    # Major artist keywords
    MAJOR_ARTISTS = frozenset([
        "taylor swift", "beyonce", "coldplay", "ed sheeran", "eagles",
        "the weeknd", "drake", "bad bunny", "harry styles", "adele",
        "bruno mars", "lady gaga", "billie eilish", "post malone",
    ])

    # Country artist keywords
    COUNTRY_ARTISTS = frozenset([
        "morgan wallen", "zach bryan", "luke combs", "chris stapleton",
        "kenny chesney", "jason aldean", "thomas rhett", "carrie underwood",
    ])

    def __init__(
        self,
        historical_prices: dict[str, dict[SeatZone, float]] | None = None,
    ) -> None:
        """Initialize cold-start handler.

        Args:
            historical_prices: Dictionary mapping artist names to zone prices
        """
        self._historical_prices = historical_prices or {}

    def get_estimate(
        self,
        artist_name: str,
        event_type: str = "CONCERT",
        city_tier: int = 2,
    ) -> ColdStartEstimate:
        """Get price estimate for a new event.

        Args:
            artist_name: Artist or team name
            event_type: Type of event (CONCERT, SPORTS, THEATER)
            city_tier: City tier (1=major, 2=medium, 3=smaller)

        Returns:
            ColdStartEstimate with prices by zone
        """
        artist_lower = artist_name.lower().strip()

        # Try performer historical prices first
        if artist_lower in self._historical_prices:
            return ColdStartEstimate(
                prices_by_zone=self._historical_prices[artist_lower],
                confidence=0.7,
                source="performer_history",
            )

        # Fall back to heuristic-based estimation
        return self._heuristic_estimate(artist_lower, event_type, city_tier)

    def _heuristic_estimate(
        self,
        artist_lower: str,
        event_type: str,
        city_tier: int,
    ) -> ColdStartEstimate:
        """Generate heuristic-based price estimate.

        Args:
            artist_lower: Lowercase artist name
            event_type: Event type
            city_tier: City tier

        Returns:
            ColdStartEstimate
        """
        # Determine artist category
        category = self._detect_category(artist_lower)
        multiplier = self.DEMAND_MULTIPLIERS[category]

        # City tier adjustment
        city_multiplier = {1: 1.2, 2: 1.0, 3: 0.8}.get(city_tier, 1.0)

        # Event type adjustment
        event_multiplier = {"CONCERT": 1.0, "SPORTS": 0.9, "THEATER": 1.1}.get(
            event_type, 1.0
        )

        # Calculate prices
        total_multiplier = multiplier * city_multiplier * event_multiplier
        prices = {
            zone: price * total_multiplier
            for zone, price in self.GLOBAL_ZONE_DEFAULTS.items()
        }

        return ColdStartEstimate(
            prices_by_zone=prices,
            confidence=0.3,  # Low confidence for heuristic
            source="heuristic",
        )

    def _detect_category(self, artist_lower: str) -> str:
        """Detect artist category from name.

        Args:
            artist_lower: Lowercase artist name

        Returns:
            Category string
        """
        if any(kw in artist_lower for kw in self.KPOP_KEYWORDS):
            return "kpop"
        elif any(major in artist_lower for major in self.MAJOR_ARTISTS):
            return "major"
        elif any(country in artist_lower for country in self.COUNTRY_ARTISTS):
            return "country"
        else:
            return "default"

    def update_historical_prices(
        self,
        artist_name: str,
        prices: dict[SeatZone, float],
    ) -> None:
        """Update historical prices for an artist.

        Args:
            artist_name: Artist name
            prices: Dictionary of zone prices
        """
        self._historical_prices[artist_name.lower().strip()] = prices
