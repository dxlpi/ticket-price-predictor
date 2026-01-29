"""Tests for Ticketmaster API client."""

from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from ticket_price_predictor.api.ticketmaster import (
    RateLimitError,
    TicketmasterClient,
    TicketmasterError,
)
from ticket_price_predictor.config import Settings
from ticket_price_predictor.schemas import EventType


class TestTicketmasterClient:
    """Tests for TicketmasterClient."""

    def test_client_initialization(self, test_settings: Settings):
        """Test client initializes with settings."""
        client = TicketmasterClient(test_settings)

        assert client.api_key == "test_api_key_12345"
        assert "ticketmaster.com" in client.base_url

    def test_client_requires_context_manager(self, test_settings: Settings):
        """Test client requires async context manager for requests."""
        client = TicketmasterClient(test_settings)

        with pytest.raises(TicketmasterError, match="not initialized"):
            import asyncio

            asyncio.run(client._request("/events.json"))

    @pytest.mark.asyncio
    async def test_client_context_manager(self, test_settings: Settings):
        """Test client works with async context manager."""
        async with TicketmasterClient(test_settings) as client:
            assert client._client is not None

    @pytest.mark.asyncio
    async def test_missing_api_key_raises_error(self):
        """Test missing API key raises error."""
        settings = Settings(ticketmaster_api_key="")

        async with TicketmasterClient(settings) as client:
            with pytest.raises(TicketmasterError, match="not configured"):
                await client._request("/events.json")


class TestParseEventMetadata:
    """Tests for event metadata parsing."""

    def test_parse_concert_event(self, test_settings: Settings, concert_event: dict[str, Any]):
        """Test parsing concert event."""
        client = TicketmasterClient(test_settings)
        metadata = client.parse_event_metadata(concert_event)

        assert metadata.event_id == "G5vzZ4kH3eGJf"
        assert metadata.event_type == EventType.CONCERT
        assert metadata.artist_or_team == "Taylor Swift"
        assert metadata.venue_name == "SoFi Stadium"
        assert metadata.city == "Inglewood"

    def test_parse_sports_event(self, test_settings: Settings, sports_event: dict[str, Any]):
        """Test parsing sports event."""
        client = TicketmasterClient(test_settings)
        metadata = client.parse_event_metadata(sports_event)

        assert metadata.event_id == "G5diZfkn0B-bP"
        assert metadata.event_type == EventType.SPORTS
        assert metadata.artist_or_team == "Los Angeles Lakers"
        assert metadata.venue_name == "Crypto.com Arena"
        assert metadata.city == "Los Angeles"

    def test_parse_theater_event(self, test_settings: Settings, theater_event: dict[str, Any]):
        """Test parsing theater event."""
        client = TicketmasterClient(test_settings)
        metadata = client.parse_event_metadata(theater_event)

        assert metadata.event_id == "G5eYZ4kQ2Rp1d"
        assert metadata.event_type == EventType.THEATER
        assert metadata.artist_or_team == "Hamilton"
        assert metadata.venue_name == "Richard Rodgers Theatre"
        assert metadata.city == "New York"


class TestApiResponses:
    """Tests for API response handling."""

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, test_settings: Settings):
        """Test rate limit response raises RateLimitError."""
        mock_response = httpx.Response(429, text="Rate limit exceeded")

        async with TicketmasterClient(test_settings) as client:
            with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = mock_response

                with pytest.raises(RateLimitError):
                    await client._request("/events.json")

    @pytest.mark.asyncio
    async def test_api_error_response(self, test_settings: Settings):
        """Test non-200 response raises TicketmasterError."""
        mock_response = httpx.Response(500, text="Internal Server Error")

        async with TicketmasterClient(test_settings) as client:
            with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = mock_response

                with pytest.raises(TicketmasterError, match="500"):
                    await client._request("/events.json")

    @pytest.mark.asyncio
    async def test_empty_search_results(self, test_settings: Settings):
        """Test empty search results return empty list."""
        mock_response = httpx.Response(200, json={"page": {"totalElements": 0}})

        async with TicketmasterClient(test_settings) as client:
            with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = mock_response

                results = await client.search_events(keyword="nonexistent")
                assert results == []

    @pytest.mark.asyncio
    async def test_search_events_returns_list(
        self, test_settings: Settings, concert_event: dict[str, Any]
    ):
        """Test search_events returns list of events."""
        mock_response = httpx.Response(200, json={"_embedded": {"events": [concert_event]}})

        async with TicketmasterClient(test_settings) as client:
            with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = mock_response

                results = await client.search_events(keyword="Taylor Swift")
                assert len(results) == 1
                assert results[0]["id"] == "G5vzZ4kH3eGJf"
