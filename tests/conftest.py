"""Pytest fixtures for ticket price predictor tests."""

import json
from pathlib import Path
from typing import Any

import pytest

from ticket_price_predictor.config import Settings


@pytest.fixture
def sample_events() -> dict[str, Any]:
    """Load sample events from fixtures."""
    fixtures_path = Path(__file__).parent.parent / "fixtures" / "sample_events.json"
    with open(fixtures_path) as f:
        return json.load(f)


@pytest.fixture
def concert_event(sample_events: dict[str, Any]) -> dict[str, Any]:
    """Get sample concert event."""
    return sample_events["concert_event"]


@pytest.fixture
def sports_event(sample_events: dict[str, Any]) -> dict[str, Any]:
    """Get sample sports event."""
    return sample_events["sports_event"]


@pytest.fixture
def theater_event(sample_events: dict[str, Any]) -> dict[str, Any]:
    """Get sample theater event."""
    return sample_events["theater_event"]


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings with mock API key."""
    return Settings(ticketmaster_api_key="test_api_key_12345")
