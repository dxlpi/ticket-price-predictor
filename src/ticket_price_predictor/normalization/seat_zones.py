"""Seat zone normalization and price mapping."""

import re

from ticket_price_predictor.schemas import SeatZone

__all__ = ["SeatZone", "SeatZoneMapper", "ZONE_PRICE_RATIOS", "ZONE_KEYWORDS"]

# Price ratios for deriving zone prices from min/max price range
# These approximate typical venue pricing distribution
ZONE_PRICE_RATIOS: dict[SeatZone, float] = {
    SeatZone.FLOOR_VIP: 1.0,  # 100% of max price
    SeatZone.LOWER_TIER: 0.70,  # 70% of max price
    SeatZone.UPPER_TIER: 0.45,  # 45% of max price
    SeatZone.BALCONY: 0.25,  # 25% of max price (or close to min)
}

# Keywords for mapping raw section names to standardized zones
ZONE_KEYWORDS: dict[SeatZone, list[str]] = {
    SeatZone.FLOOR_VIP: [
        "floor",
        "vip",
        "pit",
        "field",
        "courtside",
        "front row",
        "premium",
        "platinum",
        "club",
        "suite",
    ],
    SeatZone.LOWER_TIER: [
        "lower",
        "100",
        "orchestra",
        "main floor",
        "loge",
        "terrace",
        "box",
    ],
    SeatZone.UPPER_TIER: [
        "upper",
        "200",
        "300",
        "400",
        "mezzanine",
        "club level",
        "press level",
    ],
    SeatZone.BALCONY: [
        "balcony",
        "gallery",
        "nosebleed",
        "500",
        "rear",
        "obstructed",
        "limited view",
    ],
}


class SeatZoneMapper:
    """Maps price ranges and section names to standardized seat zones."""

    def __init__(
        self,
        price_ratios: dict[SeatZone, float] | None = None,
    ) -> None:
        """Initialize the mapper.

        Args:
            price_ratios: Optional custom price ratios for zones
        """
        self._price_ratios = price_ratios or ZONE_PRICE_RATIOS

    def map_price_range_to_zones(
        self,
        price_min: float,
        price_max: float,
    ) -> dict[SeatZone, float]:
        """Derive zone prices from an event's price range.

        Uses the configured ratios to estimate prices for each zone
        based on the overall min/max price range.

        Args:
            price_min: Minimum price from price range
            price_max: Maximum price from price range

        Returns:
            Dictionary mapping each zone to its estimated price
        """
        if price_max <= 0:
            price_max = price_min

        # Ensure min <= max
        if price_min > price_max:
            price_min, price_max = price_max, price_min

        zone_prices: dict[SeatZone, float] = {}

        for zone, ratio in self._price_ratios.items():
            # Calculate price as weighted combination of min and max
            # Higher ratios weight toward max, lower toward min
            zone_price = price_min + (price_max - price_min) * ratio
            zone_prices[zone] = round(zone_price, 2)

        return zone_prices

    def normalize_zone_name(self, raw_name: str) -> SeatZone:
        """Normalize a raw section/zone name to a standardized zone.

        Args:
            raw_name: Raw section name from venue data

        Returns:
            Normalized SeatZone enum value
        """
        name_lower = raw_name.lower().strip()

        # Check keywords for each zone (in priority order)
        for zone in [
            SeatZone.FLOOR_VIP,
            SeatZone.LOWER_TIER,
            SeatZone.UPPER_TIER,
            SeatZone.BALCONY,
        ]:
            keywords = ZONE_KEYWORDS[zone]
            for keyword in keywords:
                if keyword in name_lower:
                    return zone

        # Try to match section numbers
        section_match = re.search(r"section\s*(\d+)", name_lower)
        if section_match:
            section_num = int(section_match.group(1))
            if section_num < 100:
                return SeatZone.FLOOR_VIP
            elif section_num < 200:
                return SeatZone.LOWER_TIER
            elif section_num < 500:
                return SeatZone.UPPER_TIER
            else:
                return SeatZone.BALCONY

        # Default to upper tier for unknown sections
        return SeatZone.UPPER_TIER

    def get_zone_price_ratio(self, zone: SeatZone) -> float:
        """Get the price ratio for a zone.

        Args:
            zone: The seat zone

        Returns:
            Price ratio (0.0 to 1.0)
        """
        return self._price_ratios.get(zone, 0.5)
