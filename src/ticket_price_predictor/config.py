"""Configuration settings for ticket price predictor."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class MLConfig:
    """Machine learning configuration - centralized magic numbers.

    All ML-related thresholds, multipliers, and weights should be defined here
    to enable easy tuning and prevent scattered hardcoded values.
    """

    # === Cold Start Handler ===
    # Default zone prices (baseline for new events)
    default_floor_vip_price: float = 450.0
    default_lower_tier_price: float = 250.0
    default_upper_tier_price: float = 150.0
    default_balcony_price: float = 75.0

    # Tier multipliers (data-driven from PopularityService)
    tier_multiplier_high: float = 2.5
    tier_multiplier_medium: float = 1.5
    tier_multiplier_low: float = 1.0

    # Legacy category multipliers (fallback keyword matching)
    category_multiplier_kpop: float = 2.5
    category_multiplier_major: float = 2.0
    category_multiplier_country: float = 1.5
    category_multiplier_default: float = 1.0

    # City tier multipliers
    city_multiplier_tier1: float = 1.2
    city_multiplier_tier2: float = 1.0
    city_multiplier_tier3: float = 0.8

    # Event type multipliers
    event_multiplier_concert: float = 1.0
    event_multiplier_sports: float = 0.9
    event_multiplier_theater: float = 1.1

    # Confidence scores
    confidence_performer_history: float = 0.7
    confidence_popularity_service: float = 0.4
    confidence_keyword_fallback: float = 0.3

    # === Artist Stats ===
    premium_price_threshold: float = 200.0
    default_avg_price: float = 150.0
    default_price_std: float = 50.0
    default_premium_ratio: float = 0.5

    # === Predictor ===
    # Price bound estimation (for non-quantile models)
    price_bound_margin: float = 0.15  # ±15%

    # Direction thresholds
    direction_up_threshold: float = 1.05  # >5% above baseline
    direction_down_threshold: float = 0.95  # <5% below baseline
    direction_max_probability: float = 0.9
    direction_stable_probability: float = 0.6

    # Default confidence when interval calculation fails
    default_confidence: float = 0.5

    # === Venue Capacity Buckets ===
    venue_small_capacity: int = 5000
    venue_medium_capacity: int = 15000
    venue_large_capacity: int = 40000

    # === Popularity Aggregator ===
    popularity_high_threshold: float = 70.0
    popularity_medium_threshold: float = 40.0

    # Source weights for popularity scoring
    weight_youtube_subscribers: float = 0.30
    weight_youtube_views: float = 0.20
    weight_lastfm_listeners: float = 0.30
    weight_lastfm_play_count: float = 0.20

    # Normalization constants
    max_youtube_subscribers: int = 100_000_000
    max_youtube_views: int = 10_000_000_000
    max_lastfm_listeners: int = 50_000_000
    max_lastfm_play_count: int = 500_000_000

    # === Training ===
    default_train_ratio: float = 0.7
    default_val_ratio: float = 0.15
    default_test_ratio: float = 0.15

    # === Regional Popularity Defaults ===
    regional_default_avg_price: float = 150.0
    regional_default_listing_count: float = 0.0
    regional_default_price_ratio: float = 1.0
    regional_default_market_strength: float = 0.5

    def get_zone_defaults(self) -> dict[str, float]:
        """Get default prices by zone name."""
        return {
            "floor_vip": self.default_floor_vip_price,
            "lower_tier": self.default_lower_tier_price,
            "upper_tier": self.default_upper_tier_price,
            "balcony": self.default_balcony_price,
        }

    def get_tier_multipliers(self) -> dict[str, float]:
        """Get tier multipliers dict."""
        return {
            "high": self.tier_multiplier_high,
            "medium": self.tier_multiplier_medium,
            "low": self.tier_multiplier_low,
        }

    def get_category_multipliers(self) -> dict[str, float]:
        """Get legacy category multipliers dict."""
        return {
            "kpop": self.category_multiplier_kpop,
            "major": self.category_multiplier_major,
            "country": self.category_multiplier_country,
            "default": self.category_multiplier_default,
        }

    def get_city_multipliers(self) -> dict[int, float]:
        """Get city tier multipliers dict."""
        return {
            1: self.city_multiplier_tier1,
            2: self.city_multiplier_tier2,
            3: self.city_multiplier_tier3,
        }

    def get_event_multipliers(self) -> dict[str, float]:
        """Get event type multipliers dict."""
        return {
            "CONCERT": self.event_multiplier_concert,
            "SPORTS": self.event_multiplier_sports,
            "THEATER": self.event_multiplier_theater,
        }


@lru_cache
def get_ml_config() -> MLConfig:
    """Get cached ML configuration."""
    return MLConfig()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Ticketmaster API
    ticketmaster_api_key: str = ""
    ticketmaster_base_url: str = "https://app.ticketmaster.com/discovery/v2"

    # Setlist.fm API (for historical concert data)
    setlistfm_api_key: str = ""

    # Last.fm API (for artist popularity data)
    lastfm_api_key: str = ""

    # Data storage
    data_dir: Path = Path("./data")

    # Collection settings
    snapshot_interval_hours: int = 8

    @property
    def raw_data_dir(self) -> Path:
        """Directory for raw snapshot data."""
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """Directory for processed feature data."""
        return self.data_dir / "processed"


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
