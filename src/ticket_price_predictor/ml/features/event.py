"""Event and location feature extraction."""

import pandas as pd

from ticket_price_predictor.config import get_ml_config
from ticket_price_predictor.ml.features.base import FeatureExtractor

_config = get_ml_config()


class EventFeatureExtractor(FeatureExtractor):
    """Extract features related to events and locations."""

    # City tiers based on market size and ticket demand
    TIER_1_CITIES = frozenset([
        "new york", "los angeles", "chicago", "houston", "phoenix",
        "philadelphia", "san antonio", "san diego", "dallas", "san jose",
        "las vegas", "miami", "boston", "atlanta", "san francisco",
    ])

    TIER_2_CITIES = frozenset([
        "seattle", "denver", "washington", "nashville", "austin",
        "detroit", "portland", "charlotte", "orlando", "minneapolis",
        "tampa", "pittsburgh", "st. louis", "baltimore", "salt lake city",
    ])

    # Event type encoding
    EVENT_TYPE_MAP = {
        "CONCERT": 0,
        "SPORTS": 1,
        "THEATER": 2,
    }

    @property
    def feature_names(self) -> list[str]:
        """Return list of feature names."""
        return [
            "event_type_encoded",
            "city_tier",
            "day_of_week",
            "is_weekend",
            "month",
            "is_summer",
            "is_holiday_season",
            "venue_capacity_bucket",
        ]

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract event features.

        Expects columns: event_type, city, event_datetime, venue_capacity (optional)
        """
        result = pd.DataFrame(index=df.index)

        # Event type encoding
        if "event_type" in df.columns:
            result["event_type_encoded"] = (
                df["event_type"].map(self.EVENT_TYPE_MAP).fillna(0).astype(int)
            )
        else:
            result["event_type_encoded"] = 0

        # City tier
        if "city" in df.columns:
            city_lower = df["city"].str.lower().str.strip()
            result["city_tier"] = city_lower.apply(self._get_city_tier)
        else:
            result["city_tier"] = 3

        # DateTime features
        if "event_datetime" in df.columns:
            dt = pd.to_datetime(df["event_datetime"])
            result["day_of_week"] = dt.dt.dayofweek
            result["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
            result["month"] = dt.dt.month
            result["is_summer"] = dt.dt.month.isin([6, 7, 8]).astype(int)
            result["is_holiday_season"] = dt.dt.month.isin([11, 12]).astype(int)
        else:
            result["day_of_week"] = 5
            result["is_weekend"] = 1
            result["month"] = 6
            result["is_summer"] = 1
            result["is_holiday_season"] = 0

        # Venue capacity bucket
        if "venue_capacity" in df.columns:
            result["venue_capacity_bucket"] = df["venue_capacity"].apply(
                self._capacity_bucket
            )
        else:
            result["venue_capacity_bucket"] = 2  # Medium default

        return result

    def _get_city_tier(self, city: str) -> int:
        """Get city tier (1=major, 2=medium, 3=smaller)."""
        if city in self.TIER_1_CITIES:
            return 1
        elif city in self.TIER_2_CITIES:
            return 2
        else:
            return 3

    def _capacity_bucket(self, capacity: float | None) -> int:
        """Bucket venue capacity into categories (thresholds from config)."""
        if pd.isna(capacity):
            return 2  # Unknown -> medium
        elif capacity < _config.venue_small_capacity:
            return 0  # Small/intimate
        elif capacity < _config.venue_medium_capacity:
            return 1  # Medium arena
        elif capacity < _config.venue_large_capacity:
            return 2  # Large arena
        else:
            return 3  # Stadium
