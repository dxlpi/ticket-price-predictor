#!/usr/bin/env python3
"""Monitor top popular events from Vivid Seats.

Collects seat-level pricing data from the most popular upcoming events.
Designed to run hourly via cron or scheduler.

Usage:
    # One-time collection
    python scripts/monitor_popular.py

    # Cron job (every hour)
    0 * * * * cd /path/to/ticket-price-predictor && python scripts/monitor_popular.py >> logs/monitor.log 2>&1
"""

import asyncio
from datetime import datetime
from pathlib import Path

from ticket_price_predictor.ingestion import DataSource, ListingCollector
from ticket_price_predictor.popularity import PopularityService
from ticket_price_predictor.storage import ListingRepository


async def get_popular_events(
    max_events: int = 20,
    popularity_service: PopularityService | None = None,
) -> list[dict]:
    """Fetch top popular events from Vivid Seats using search.

    Uses search_events() for each performer instead of navigating to
    performer pages, which VividSeats manipulates for scrapers.

    Args:
        max_events: Maximum number of events to return
        popularity_service: Optional PopularityService for ranking performers

    Returns:
        List of event dictionaries with name, url, etc.
    """
    from ticket_price_predictor.scrapers import VividSeatsScraper

    print("Fetching popular performers and their events...")

    # Pre-defined list of popular performers to search
    # These are known high-ticket-volume artists
    popular_performers = [
        "Bruno Mars",
        "Lady Gaga",
        "Morgan Wallen",
        "BTS",
        "Taylor Swift",
        "Beyonce",
        "The Eagles",
        "Chris Stapleton",
        "Ariana Grande",
        "Harry Styles",
        "Backstreet Boys",
        "George Strait",
        "Megan Moroney",
        "Olivia Dean",
        "Rush",
    ]

    # Use popularity service to rank if available
    if popularity_service:
        ranked = popularity_service.rank_performers(popular_performers)
        performer_names = [p.name for p in ranked]
        tier_lookup = {p.name.lower(): p.tier_allocation for p in ranked}
        print(f"  Using popularity-ranked selection: {len(performer_names)} performers")
    else:
        performer_names = popular_performers[:10]
        tier_lookup = {}

    events = []

    # Use single browser session for all searches
    async with VividSeatsScraper(headless=True, delay_seconds=2.0) as scraper:
        for performer_name in performer_names:
            if len(events) >= max_events:
                break

            # Get tier-based event allocation (default to 2 if no popularity data)
            events_to_collect = tier_lookup.get(performer_name.lower(), 2)

            try:
                # Use search which works correctly
                scraped_events = await scraper.search_events(
                    performer_name,
                    max_results=events_to_collect,
                )

                for e in scraped_events:
                    if len(events) >= max_events:
                        break

                    events.append({
                        "id": e.stubhub_event_id,
                        "name": e.event_name,
                        "url": e.event_url,
                        "venue": e.venue_name,
                        "city": e.city,
                        "min_price": e.min_price,
                        "ticket_count": e.ticket_count,
                        "performer": performer_name,
                    })

                print(f"  {performer_name}: {len(scraped_events)} events, added {min(len(scraped_events), events_to_collect)}")

            except Exception as e:
                print(f"  Error fetching {performer_name}: {e}")

    return events[:max_events]


async def collect_listings_for_events(
    events: list[dict],
    data_dir: Path,
    max_listings_per_event: int = 100,
) -> dict:
    """Collect listings for a list of events.

    Uses a SINGLE browser session for all events to avoid repeated
    browser launches and reduce timeouts.

    Args:
        events: List of event dictionaries
        data_dir: Data storage directory
        max_listings_per_event: Max listings per event

    Returns:
        Summary statistics
    """
    from datetime import UTC, datetime

    from ticket_price_predictor.schemas import ScrapedEvent, create_listing_from_scraped
    from ticket_price_predictor.scrapers import VividSeatsScraper
    from ticket_price_predictor.storage import ListingRepository

    repository = ListingRepository(data_dir)
    timestamp = datetime.now(UTC)

    stats = {
        "events_attempted": len(events),
        "events_succeeded": 0,
        "total_listings": 0,
        "errors": [],
    }

    # Use a SINGLE browser session for all events
    async with VividSeatsScraper(headless=True, delay_seconds=3.0) as scraper:
        for i, event in enumerate(events, 1):
            print(f"\n[{i}/{len(events)}] {event['name'][:50]}...")

            try:
                # Get listings using persistent browser session
                scraped_listings = await scraper.get_event_listings(
                    event["url"],
                    max_listings=max_listings_per_event,
                )

                if scraped_listings:
                    # Create event object for conversion
                    scraped_event = ScrapedEvent(
                        stubhub_event_id=event.get("id", ""),
                        event_name=event["name"],
                        artist_or_team=event.get("performer", "Unknown Artist"),
                        venue_name=event["venue"],
                        city=event["city"],
                        event_datetime=datetime.now(UTC),
                        event_url=event["url"],
                    )

                    # Convert and save listings
                    listings = [
                        create_listing_from_scraped(s, scraped_event, timestamp)
                        for s in scraped_listings
                    ]
                    saved = repository.save_listings(listings)

                    stats["events_succeeded"] += 1
                    stats["total_listings"] += saved
                    if saved > 0:
                        print(f"    Collected {saved} new listings (from {len(scraped_listings)} found)")
                    else:
                        print(f"    Found {len(scraped_listings)} listings (no price changes)")
                else:
                    print("    No listings found")

            except Exception as e:
                error_msg = f"{event['name']}: {e}"
                stats["errors"].append(error_msg)
                print(f"    Error: {e}")

    return stats


async def main() -> None:
    """Main entry point."""
    data_dir = Path("data")
    max_listings_per_event = 100

    # Initialize popularity service
    pop_service = PopularityService()

    # Default max events (actual count determined by tier allocations in get_popular_events)
    max_events = 30

    print("=" * 60)
    print("POPULAR EVENTS MONITOR")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Max events: {max_events}")
    print(f"Max listings per event: {max_listings_per_event}")
    print()

    # Get popular events with popularity-based ranking
    events = await get_popular_events(
        max_events=max_events,
        popularity_service=pop_service,
    )
    print(f"\nFound {len(events)} popular events to monitor:")
    for i, event in enumerate(events, 1):
        price_str = f"${event['min_price']}" if event['min_price'] else "N/A"
        performer = event.get('performer', event.get('category', 'Unknown'))
        print(f"  {i:2}. [{performer[:15]}] {event['name'][:40]} - {price_str} ({event['ticket_count']} tix)")

    if not events:
        print("\nNo events found to monitor!")
        return

    # Collect listings
    print("\n" + "=" * 60)
    print("COLLECTING LISTINGS")
    print("=" * 60)

    stats = await collect_listings_for_events(
        events=events,
        data_dir=data_dir,
        max_listings_per_event=max_listings_per_event,
    )

    # Summary
    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Events attempted: {stats['events_attempted']}")
    print(f"Events succeeded: {stats['events_succeeded']}")
    print(f"Total listings collected: {stats['total_listings']}")

    if stats["errors"]:
        print(f"\nErrors ({len(stats['errors'])}):")
        for error in stats["errors"][:5]:
            print(f"  - {error[:80]}")
        if len(stats["errors"]) > 5:
            print(f"  ... and {len(stats['errors']) - 5} more")

    # Show current data stats
    print("\n" + "=" * 60)
    print("CURRENT DATA STATS")
    print("=" * 60)

    repo = ListingRepository(data_dir)
    all_listings = repo.get_listings()
    event_ids = repo.list_event_ids()

    print(f"Total events being monitored: {len(event_ids)}")
    print(f"Total listings in database: {len(all_listings)}")

    print(f"\nCompleted at: {datetime.now().isoformat()}")


if __name__ == "__main__":
    asyncio.run(main())
