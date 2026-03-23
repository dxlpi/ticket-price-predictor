"""Event and location feature extraction."""

import numpy as np
import pandas as pd

from ticket_price_predictor.config import get_ml_config
from ticket_price_predictor.ml.features.base import FeatureExtractor
from ticket_price_predictor.ml.features.geo_mapping import _normalize_city

_config = get_ml_config()


class EventFeatureExtractor(FeatureExtractor):
    """Extract features related to events and locations."""

    # City tiers based on market size and ticket demand
    TIER_1_CITIES = frozenset(
        [
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
    )

    TIER_2_CITIES = frozenset(
        [
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
    )

    # Event type encoding
    EVENT_TYPE_MAP = {
        "concert": 0,
        "sports": 1,
        "theater": 2,
        "comedy": 3,
    }

    def __init__(self) -> None:
        """Initialize with empty city-week counts (populated in fit)."""
        # Maps (city, iso_year, iso_week) → distinct event count in training split
        self._city_week_counts: dict[tuple[str, int, int], int] = {}

    def fit(self, df: pd.DataFrame) -> "EventFeatureExtractor":
        """Compute backward-looking city-week event counts from training data.

        Uses ISO calendar (year, week) with city to count concurrent events.
        The (city, year, week) key prevents cross-year week-number collisions.
        Because fit() is only called on training data, market_saturation is
        backward-looking and does not leak future event counts.

        Args:
            df: Training DataFrame (raw listings)

        Returns:
            self
        """
        self._city_week_counts = {}
        if "city" not in df.columns or "event_datetime" not in df.columns:
            return self
        if "event_id" not in df.columns:
            # Cannot count distinct events without an event_id — skip saturation
            return self

        dt = pd.to_datetime(df["event_datetime"])
        iso = dt.dt.isocalendar()  # DataFrame with year, week, day columns
        city_norm = df["city"].apply(_normalize_city)

        temp = pd.DataFrame(
            {
                "city": city_norm,
                "event_id": df["event_id"],
                "iso_year": iso["year"].astype(int),
                "iso_week": iso["week"].astype(int),
            }
        )

        for (city, year, week), group in temp.groupby(["city", "iso_year", "iso_week"]):
            self._city_week_counts[(str(city), int(year), int(week))] = int(
                group["event_id"].nunique()
            )

        return self

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
            "market_saturation",
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
            city_normalized = df["city"].apply(_normalize_city)
            result["city_tier"] = city_normalized.apply(self._get_city_tier)
        else:
            result["city_tier"] = 3

        # Compute datetime series once; reused by both datetime features and market saturation
        dt = pd.to_datetime(df["event_datetime"]) if "event_datetime" in df.columns else None

        # DateTime features
        if dt is not None:
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
            result["venue_capacity_bucket"] = df["venue_capacity"].apply(self._capacity_bucket)
        else:
            result["venue_capacity_bucket"] = 2  # Medium default

        # Market saturation: log1p(number of distinct events in same city/week)
        # Counts come from fit() on training data only → no temporal leakage
        if "city" in df.columns and dt is not None and self._city_week_counts:
            iso = dt.dt.isocalendar()
            city_norm = df["city"].apply(_normalize_city)
            result["market_saturation"] = [
                np.log1p(self._city_week_counts.get((str(city), int(year), int(week)), 0))
                for city, year, week in zip(
                    city_norm, iso["year"].astype(int), iso["week"].astype(int), strict=False
                )
            ]
        else:
            result["market_saturation"] = 0.0

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
        if capacity is None or pd.isna(capacity):
            return 2  # Unknown -> medium
        elif capacity < _config.venue_small_capacity:
            return 0  # Small/intimate
        elif capacity < _config.venue_medium_capacity:
            return 1  # Medium arena
        elif capacity < _config.venue_large_capacity:
            return 2  # Large arena
        else:
            return 3  # Stadium
