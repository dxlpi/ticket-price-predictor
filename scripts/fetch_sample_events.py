#!/usr/bin/env python3
"""Fetch sample events from Ticketmaster API for testing."""

import asyncio
import json
from datetime import datetime, timedelta

from ticket_price_predictor.api import TicketmasterClient
from ticket_price_predictor.config import get_settings


async def main() -> None:
    """Fetch and display sample events."""
    settings = get_settings()

    if not settings.ticketmaster_api_key:
        print("Error: TICKETMASTER_API_KEY not set")
        print("Set the environment variable or create a .env file")
        return

    print("Fetching sample events from Ticketmaster...")
    print(f"API Key: {settings.ticketmaster_api_key[:8]}...")

    async with TicketmasterClient(settings) as client:
        # Search for upcoming concerts
        start_date = datetime.now()
        end_date = start_date + timedelta(days=90)

        print("\n--- Concerts ---")
        concerts = await client.search_events(
            classification_name="Music",
            start_date=start_date,
            end_date=end_date,
            size=3,
        )
        for event in concerts:
            metadata = client.parse_event_metadata(event)
            print(f"  {metadata.artist_or_team} @ {metadata.venue_name}, {metadata.city}")
            print(f"    Date: {metadata.event_datetime}")
            print(f"    Event ID: {metadata.event_id}")

        print("\n--- Sports ---")
        sports = await client.search_events(
            classification_name="Sports",
            start_date=start_date,
            end_date=end_date,
            size=3,
        )
        for event in sports:
            metadata = client.parse_event_metadata(event)
            print(f"  {metadata.artist_or_team} @ {metadata.venue_name}, {metadata.city}")
            print(f"    Date: {metadata.event_datetime}")
            print(f"    Event ID: {metadata.event_id}")

        print("\n--- Theater ---")
        theater = await client.search_events(
            classification_name="Arts & Theatre",
            start_date=start_date,
            end_date=end_date,
            size=3,
        )
        for event in theater:
            metadata = client.parse_event_metadata(event)
            print(f"  {metadata.artist_or_team} @ {metadata.venue_name}, {metadata.city}")
            print(f"    Date: {metadata.event_datetime}")
            print(f"    Event ID: {metadata.event_id}")

        # Save raw responses for inspection
        output = {
            "fetched_at": datetime.now().isoformat(),
            "concerts": concerts[:1] if concerts else [],
            "sports": sports[:1] if sports else [],
            "theater": theater[:1] if theater else [],
        }

        output_path = settings.raw_data_dir / "sample_api_response.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\nRaw API response saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
