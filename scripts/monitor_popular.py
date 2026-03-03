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
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from ticket_price_predictor.popularity import PopularityService
from ticket_price_predictor.storage import ListingRepository

# Tier-based event allocation (events per artist per collection run)
TIER_EVENT_LIMITS = {
    "stadium": 5,
    "arena": 3,
    "theater": 2,
    "emerging": 1,
}

WATCHLIST_PATH = Path("data/artist_watchlist.json")


def load_artist_watchlist(path: Path = WATCHLIST_PATH) -> dict[str, list[str]]:
    """Load artist watchlist from JSON config.

    Args:
        path: Path to watchlist JSON file

    Returns:
        Dict mapping tier name to list of artist names
    """
    if not path.exists():
        print(f"  Warning: watchlist not found at {path}, using empty list")
        return {}

    with open(path) as f:
        data = json.load(f)

    # Filter out metadata keys (start with _)
    return {k: v for k, v in data.items() if not k.startswith("_")}


async def get_popular_events(
    max_events: int = 50,
    popularity_service: PopularityService | None = None,
) -> list[dict]:
    """Fetch top popular events from Vivid Seats using search.

    Uses search_events() for each performer instead of navigating to
    performer pages, which VividSeats manipulates for scrapers.

    Artists are loaded from data/artist_watchlist.json organized by
    popularity tier. Each tier has a different event allocation.

    Args:
        max_events: Maximum number of events to return
        popularity_service: Optional PopularityService for ranking performers

    Returns:
        List of event dictionaries with name, url, etc.
    """
    from ticket_price_predictor.scrapers import VividSeatsScraper

    print("Fetching popular performers and their events...")

    # Load artists from config file, organized by tier
    watchlist = load_artist_watchlist()

    if not watchlist:
        print("  No artists in watchlist!")
        return []

    # Build flat performer list with tier-based event limits
    performer_configs: list[tuple[str, int]] = []
    for tier, artists in watchlist.items():
        events_per_artist = TIER_EVENT_LIMITS.get(tier, 1)
        for artist in artists:
            performer_configs.append((artist, events_per_artist))

    total_artists = len(performer_configs)
    print(f"  Loaded {total_artists} artists from watchlist across {len(watchlist)} tiers")

    # Use popularity service to re-rank within tiers if available
    if popularity_service:
        all_names = [name for name, _ in performer_configs]
        ranked = popularity_service.rank_performers(all_names)
        # Rebuild configs with ranked order but keep tier-based limits
        tier_limits = {name.lower(): limit for name, limit in performer_configs}
        performer_configs = [
            (p.name, tier_limits.get(p.name.lower(), 1)) for p in ranked
        ]
        print(f"  Re-ranked by popularity: {len(performer_configs)} performers")

    events = []

    # Use single browser session for all searches
    async with VividSeatsScraper(headless=True, delay_seconds=2.0) as scraper:
        for performer_name, events_to_collect in performer_configs:
            if len(events) >= max_events:
                break

            try:
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
                        "event_datetime": e.event_datetime,
                    })

                print(f"  {performer_name}: {len(scraped_events)} events, added {min(len(scraped_events), events_to_collect)}")

            except Exception as e:
                print(f"  Error fetching {performer_name}: {e}")

    return events[:max_events]


def create_snapshots_from_listings(
    listings: list,
    event_id: str,
    timestamp: datetime,
    days_to_event: int,
) -> list:
    """Aggregate listings into zone-level price snapshots.

    Groups listings by normalized seat zone and computes
    min/avg/max price and inventory count per zone.

    Args:
        listings: List of TicketListing objects
        event_id: Event identifier
        timestamp: Snapshot capture time
        days_to_event: Days until event

    Returns:
        List of PriceSnapshot objects
    """
    from collections import defaultdict

    from ticket_price_predictor.normalization import SeatZoneMapper
    from ticket_price_predictor.schemas import PriceSnapshot

    mapper = SeatZoneMapper()

    # Group prices by zone
    zone_prices: dict[str, list[float]] = defaultdict(list)
    zone_quantities: dict[str, int] = defaultdict(int)

    for listing in listings:
        zone = mapper.normalize_zone_name(listing.section)
        zone_prices[zone.value].append(listing.listing_price)
        zone_quantities[zone.value] += listing.quantity

    # Create snapshots
    snapshots = []
    for zone_value, prices in zone_prices.items():
        if not prices:
            continue

        from ticket_price_predictor.schemas import SeatZone

        snapshots.append(
            PriceSnapshot(
                event_id=event_id,
                seat_zone=SeatZone(zone_value),
                timestamp=timestamp,
                price_min=min(prices),
                price_avg=sum(prices) / len(prices),
                price_max=max(prices),
                inventory_remaining=zone_quantities[zone_value],
                days_to_event=max(0, days_to_event),
            )
        )

    return snapshots


async def collect_listings_for_events(
    events: list[dict],
    data_dir: Path,
    max_listings_per_event: int = 100,
) -> dict:
    """Collect listings for a list of events and create price snapshots.

    Uses a SINGLE browser session for all events to avoid repeated
    browser launches and reduce timeouts. After collecting listings,
    aggregates them into zone-level price snapshots for longitudinal
    tracking.

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
    from ticket_price_predictor.storage import ListingRepository, SnapshotRepository

    repository = ListingRepository(data_dir)
    snapshot_repo = SnapshotRepository(data_dir)
    timestamp = datetime.now(UTC)

    stats = {
        "events_attempted": len(events),
        "events_succeeded": 0,
        "total_listings": 0,
        "total_snapshots": 0,
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

                    # Create and save zone-level price snapshots
                    if listings:
                        days_to = listings[0].days_to_event
                        snapshots = create_snapshots_from_listings(
                            listings, event.get("id", ""), timestamp, days_to
                        )
                        if snapshots:
                            snap_saved = snapshot_repo.save_snapshots(snapshots)
                            stats["total_snapshots"] += snap_saved

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
    import argparse

    parser = argparse.ArgumentParser(description="Monitor popular event ticket prices")
    parser.add_argument(
        "--urgent",
        action="store_true",
        help="Only collect events within 14 days (for high-frequency runs)",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=100,
        help="Maximum events to collect (default: 100)",
    )
    args = parser.parse_args()

    data_dir = Path("data")
    max_listings_per_event = 100

    # Initialize popularity service
    pop_service = PopularityService()

    max_events = args.max_events
    mode = "URGENT (≤14 days)" if args.urgent else "FULL"

    print("=" * 60)
    print(f"POPULAR EVENTS MONITOR [{mode}]")
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
    # Filter to near-term events if --urgent
    if args.urgent:
        from datetime import UTC, timedelta

        cutoff = datetime.now(UTC) + timedelta(days=14)
        before_count = len(events)
        events = [
            e for e in events
            if e.get("event_datetime") and e["event_datetime"] < cutoff
        ]
        print(f"\nUrgent filter: {before_count} → {len(events)} events (within 14 days)")

    print(f"\nFound {len(events)} events to monitor:")
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
    print(f"Total snapshots saved: {stats.get('total_snapshots', 0)}")

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
