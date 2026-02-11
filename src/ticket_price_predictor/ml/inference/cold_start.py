"""Cold-start handling for new events and performers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ticket_price_predictor.config import get_ml_config
from ticket_price_predictor.normalization.seat_zones import SeatZone

if TYPE_CHECKING:
    from ticket_price_predictor.popularity.service import PopularityService

logger = logging.getLogger(__name__)
_config = get_ml_config()


@dataclass
class ColdStartEstimate:
    """Estimated prices for cold-start scenarios."""

    prices_by_zone: dict[SeatZone, float]
    confidence: float  # 0-1, lower for cold-start
    source: str  # "cluster", "performer_history", "heuristic", "global_default"


class ColdStartHandler:
    """Handle predictions for new events and performers."""

    # Global default prices by zone (from centralized config)
    GLOBAL_ZONE_DEFAULTS = {
        SeatZone.FLOOR_VIP: _config.default_floor_vip_price,
        SeatZone.LOWER_TIER: _config.default_lower_tier_price,
        SeatZone.UPPER_TIER: _config.default_upper_tier_price,
        SeatZone.BALCONY: _config.default_balcony_price,
    }

    # Demand multipliers by popularity tier (data-driven, from config)
    TIER_MULTIPLIERS = _config.get_tier_multipliers()

    # Legacy demand multipliers by artist category (fallback, from config)
    DEMAND_MULTIPLIERS = _config.get_category_multipliers()

    # K-pop keywords for detection (fallback when PopularityService unavailable)
    KPOP_KEYWORDS = frozenset(
        [
            "blackpink",
            "bts",
            "twice",
            "stray kids",
            "aespa",
            "newjeans",
            "seventeen",
            "nct",
            "exo",
            "red velvet",
            "itzy",
            "ive",
            "le sserafim",
        ]
    )

    # Major artist keywords (fallback)
    MAJOR_ARTISTS = frozenset(
        [
            "taylor swift",
            "beyonce",
            "coldplay",
            "ed sheeran",
            "eagles",
            "the weeknd",
            "drake",
            "bad bunny",
            "harry styles",
            "adele",
            "bruno mars",
            "lady gaga",
            "billie eilish",
            "post malone",
        ]
    )

    # Country artist keywords (fallback)
    COUNTRY_ARTISTS = frozenset(
        [
            "morgan wallen",
            "zach bryan",
            "luke combs",
            "chris stapleton",
            "kenny chesney",
            "jason aldean",
            "thomas rhett",
            "carrie underwood",
        ]
    )

    def __init__(
        self,
        historical_prices: dict[str, dict[SeatZone, float]] | None = None,
        popularity_service: PopularityService | None = None,
    ) -> None:
        """Initialize cold-start handler.

        Args:
            historical_prices: Dictionary mapping artist names to zone prices
            popularity_service: Optional PopularityService for data-driven tier detection
        """
        self._historical_prices = historical_prices or {}
        self._popularity_service = popularity_service

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
                confidence=_config.confidence_performer_history,
                source="performer_history",
            )

        # Fall back to heuristic-based estimation (pass original name for PopularityService)
        return self._heuristic_estimate(artist_name, event_type, city_tier)

    def _heuristic_estimate(
        self,
        artist_name: str,
        event_type: str,
        city_tier: int,
    ) -> ColdStartEstimate:
        """Generate heuristic-based price estimate.

        Args:
            artist_name: Artist name (original case preserved for PopularityService)
            event_type: Event type
            city_tier: City tier

        Returns:
            ColdStartEstimate
        """
        # Get demand multiplier (data-driven or keyword fallback)
        multiplier, source = self._get_tier_multiplier(artist_name)

        # City tier adjustment (from config)
        city_multiplier = _config.get_city_multipliers().get(
            city_tier, _config.city_multiplier_tier2
        )

        # Event type adjustment (from config)
        event_multiplier = _config.get_event_multipliers().get(
            event_type, _config.event_multiplier_concert
        )

        # Calculate prices
        total_multiplier = multiplier * city_multiplier * event_multiplier
        prices = {
            zone: price * total_multiplier for zone, price in self.GLOBAL_ZONE_DEFAULTS.items()
        }

        # Higher confidence when using PopularityService (from config)
        confidence = (
            _config.confidence_popularity_service
            if source == "popularity_service"
            else _config.confidence_keyword_fallback
        )

        return ColdStartEstimate(
            prices_by_zone=prices,
            confidence=confidence,
            source=f"heuristic_{source}",
        )

    def _get_tier_multiplier(self, artist_name: str) -> tuple[float, str]:
        """Get demand multiplier based on artist popularity tier.

        Uses PopularityService for data-driven tier detection, with fallback
        to keyword matching when service is unavailable.

        Args:
            artist_name: Artist name (will be lowercased)

        Returns:
            Tuple of (multiplier, source) where source is "popularity_service" or "keyword_fallback"
        """
        artist_lower = artist_name.lower().strip()

        # Try PopularityService first (data-driven approach)
        if self._popularity_service is not None:
            try:
                popularity = self._popularity_service.get_artist_popularity(artist_name)
                tier = popularity.tier.value  # "high", "medium", or "low"
                multiplier = self.TIER_MULTIPLIERS.get(tier, 1.0)
                logger.debug(f"PopularityService tier for '{artist_name}': {tier} -> {multiplier}x")
                return multiplier, "popularity_service"
            except Exception as e:
                logger.warning(
                    f"PopularityService lookup failed for '{artist_name}': {e}, "
                    "falling back to keyword matching"
                )

        # Fallback to keyword-based detection
        category = self._detect_category_keywords(artist_lower)
        multiplier = self.DEMAND_MULTIPLIERS[category]
        return multiplier, "keyword_fallback"

    def _detect_category_keywords(self, artist_lower: str) -> str:
        """Detect artist category from name using keyword matching (fallback).

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
