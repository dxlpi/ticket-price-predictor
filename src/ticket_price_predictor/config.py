"""Configuration settings for ticket price predictor."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # Data storage
    data_dir: Path = Path("./data")

    # Collection settings
    snapshot_interval_hours: int = 6

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
