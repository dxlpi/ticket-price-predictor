"""Seating and zone feature extraction."""

import re

import pandas as pd

from ticket_price_predictor.ml.features.base import FeatureExtractor
from ticket_price_predictor.normalization.seat_zones import SeatZone, SeatZoneMapper

# Keywords that indicate premium/high-value seating within any zone
_PREMIUM_KEYWORDS = frozenset(
    {"vip", "platinum", "premium", "pit", "field", "courtside", "front row", "suite"}
)
# Exclude GA sections — even if they mention "pit", GA floor is not premium-priced
_PREMIUM_EXCLUDE = frozenset({"general admission"})


def _is_premium_section(section: str) -> int:
    """Return 1 if section name indicates a premium seat, else 0."""
    s = str(section).lower()
    if any(excl in s for excl in _PREMIUM_EXCLUDE):
        return 0
    return int(any(kw in s for kw in _PREMIUM_KEYWORDS))


class SeatingFeatureExtractor(FeatureExtractor):
    """Extract features related to seating and zones."""

    # Zone price ratios (from SeatZoneMapper)
    ZONE_PRICE_RATIOS = {
        SeatZone.FLOOR_VIP: 1.0,
        SeatZone.LOWER_TIER: 0.70,
        SeatZone.UPPER_TIER: 0.45,
        SeatZone.BALCONY: 0.25,
    }

    # Zone encoding for ML
    ZONE_ENCODING = {
        SeatZone.FLOOR_VIP: 3,
        SeatZone.LOWER_TIER: 2,
        SeatZone.UPPER_TIER: 1,
        SeatZone.BALCONY: 0,
    }

    def __init__(self) -> None:
        """Initialize with SeatZoneMapper."""
        self._mapper = SeatZoneMapper()

    @property
    def feature_names(self) -> list[str]:
        """Return list of feature names."""
        return [
            "seat_zone_encoded",
            "zone_price_ratio",
            "row_numeric",
            "is_floor",
            "is_ga",
            "is_premium",
        ]

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract seating features.

        Expects columns: section, row (optional)
        """
        result = pd.DataFrame(index=df.index)

        # Normalize sections to zones
        zones = df["section"].apply(self._mapper.normalize_zone_name)
        result["seat_zone_encoded"] = zones.map(self.ZONE_ENCODING)
        result["zone_price_ratio"] = zones.map(self.ZONE_PRICE_RATIOS)

        # Row features
        if "row" in df.columns:
            result["row_numeric"] = df["row"].apply(self._parse_row)
        else:
            result["row_numeric"] = 10  # Default mid-row

        # Binary features
        section_lower = df["section"].str.lower()
        result["is_floor"] = (
            section_lower.str.contains("floor|pit|field|courtside", regex=True)
        ).astype(int)
        result["is_ga"] = (section_lower.str.contains("ga|general admission", regex=True)).astype(
            int
        )

        # Premium section detection: keyword matching on raw section name
        result["is_premium"] = df["section"].fillna("").apply(_is_premium_section)

        return result

    def _parse_row(self, row: str | None) -> int:
        """Parse row string to numeric value."""
        if pd.isna(row) or not row:
            return 10  # Default mid-row

        row_str = str(row).strip().upper()

        # Handle GA
        if row_str in ("GA", "GENERAL", "STANDING"):
            return 0

        # Try numeric extraction
        numbers = re.findall(r"\d+", row_str)
        if numbers:
            return int(numbers[0])

        # Handle letter rows (A=1, B=2, etc.)
        if len(row_str) == 1 and row_str.isalpha():
            return ord(row_str) - ord("A") + 1
        if len(row_str) == 2 and row_str.isalpha():
            # AA, BB, etc. = premium rows
            return (ord(row_str[0]) - ord("A")) * 26 + (ord(row_str[1]) - ord("A")) + 1

        return 10  # Default
