#!/usr/bin/env python3
"""Monitor top popular events from Vivid Seats.

Collects seat-level pricing data from the most popular upcoming events.
Designed to run hourly via cron or scheduler.

Discovery uses two complementary strategies:
1. Watchlist: search for specific artists by name (priority artists)
2. Browse: discover concerts via the hermes productions API (diverse coverage)

Usage:
    # One-time collection
    python scripts/monitor_popular.py

    # Cron job (every hour)
    0 * * * * cd /path/to/ticket-price-predictor && python scripts/monitor_popular.py >> logs/monitor.log 2>&1
"""

import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from ticket_price_predictor.popularity import PopularityService
from ticket_price_predictor.storage import ListingRepository

# Tier-based event allocation (events per artist per collection run)
TIER_EVENT_LIMITS = {
    "stadium": 3,
    "arena": 2,
    "theater": 1,
    "emerging": 1,
}

WATCHLIST_PATH = Path("data/artist_watchlist.json")
FESTIVAL_KEYWORDS_PATH = Path("data/festival_keywords.json")

# Default festival keywords (used if config file not found)
DEFAULT_FESTIVAL_KEYWORDS = [
    "festival", "music fest", "season ticket", "day pass",
    "3 day", "2 day", "weekend pass", "ga pass",
    "bottlerock", "stagecoach", "tortuga", "coachella",
    "lollapalooza", "bonnaroo", "two step inn", "cma fest",
]

# Safety limits
MAX_CONSECUTIVE_FAILURES = 5
TIME_BUDGET_SECONDS = 45 * 60  # 45 minutes


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


def load_festival_keywords(path: Path = FESTIVAL_KEYWORDS_PATH) -> list[str]:
    """Load festival/bundle keywords from config file.

    Args:
        path: Path to festival keywords JSON file

    Returns:
        List of keywords (lowercase)
    """
    if not path.exists():
        return [k.lower() for k in DEFAULT_FESTIVAL_KEYWORDS]

    with open(path) as f:
        keywords = json.load(f)

    return [k.lower() for k in keywords]


def is_festival_or_bundle(event_name: str, keywords: list[str]) -> bool:
    """Check if an event is a festival, season ticket, or multi-day pass.

    Args:
        event_name: Event name to check
        keywords: List of lowercase keywords to match against

    Returns:
        True if the event matches any festival/bundle keyword
    """
    name_lower = event_name.lower()
    return any(kw in name_lower for kw in keywords)


async def get_popular_events(
    max_events: int = 150,
    popularity_service: PopularityService | None = None,
) -> list[dict[str, Any]]:
    """Fetch popular events using watchlist searches + browse discovery.

    Two-phase discovery:
    1. Search for each watchlist artist (priority coverage)
    2. Browse concerts via hermes API (diverse coverage)

    Events are deduplicated by ID and (venue, date) tuple.
    Festival/bundle events are filtered out.

    Args:
        max_events: Maximum number of events to return
        popularity_service: Optional PopularityService for ranking performers

    Returns:
        List of event dictionaries with name, url, etc.
    """
    from ticket_price_predictor.scrapers import VividSeatsScraper

    start_time = time.monotonic()
    festival_keywords = load_festival_keywords()
    consecutive_failures = 0

    print("Fetching popular performers and their events...")

    # Load artists from config file, organized by tier
    watchlist = load_artist_watchlist()

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

    events: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_venue_dates: set[tuple[str, str]] = set()

    def _add_event(event_dict: dict[str, Any]) -> bool:
        """Add event if not a duplicate or festival. Returns True if added."""
        event_id = event_dict.get("id", "")
        venue = event_dict.get("venue", "")
        event_dt = event_dict.get("event_datetime")
        date_str = event_dt.isoformat()[:10] if event_dt else ""
        venue_date_key = (venue.lower(), date_str)

        # Dedup by ID
        if event_id and event_id in seen_ids:
            return False
        # Dedup by (venue, date)
        if venue and date_str and venue_date_key in seen_venue_dates:
            return False
        # Festival filter
        if is_festival_or_bundle(event_dict.get("name", ""), festival_keywords):
            return False

        events.append(event_dict)
        if event_id:
            seen_ids.add(event_id)
        if venue and date_str:
            seen_venue_dates.add(venue_date_key)
        return True

    # === Phase 1: Watchlist searches ===
    print("\n  Phase 1: Watchlist artist searches...")
    async with VividSeatsScraper(headless=True, delay_seconds=2.0) as scraper:
        for performer_name, events_to_collect in performer_configs:
            if len(events) >= max_events:
                break

            # Time budget check
            elapsed = time.monotonic() - start_time
            if elapsed > TIME_BUDGET_SECONDS:
                print(f"  Time budget reached ({elapsed:.0f}s). Stopping watchlist search.")
                break

            # Circuit breaker
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"  Circuit breaker: {MAX_CONSECUTIVE_FAILURES} consecutive failures. Pausing 60s...")
                await asyncio.sleep(60)
                consecutive_failures = 0
                # Try one more
                try:
                    test_events = await scraper.search_events(performer_name, max_results=1)
                    if not test_events:
                        print("  Circuit breaker: still failing after pause. Stopping watchlist search.")
                        break
                except Exception:
                    print("  Circuit breaker: still failing after pause. Stopping watchlist search.")
                    break

            try:
                scraped_events = await scraper.search_events(
                    performer_name,
                    max_results=events_to_collect,
                )

                added = 0
                for e in scraped_events:
                    if len(events) >= max_events:
                        break
                    if _add_event({
                        "id": e.stubhub_event_id,
                        "name": e.event_name,
                        "url": e.event_url,
                        "venue": e.venue_name,
                        "city": e.city,
                        "min_price": e.min_price,
                        "ticket_count": e.ticket_count,
                        "performer": performer_name,
                        "event_datetime": e.event_datetime,
                    }):
                        added += 1

                if scraped_events:
                    consecutive_failures = 0
                    print(f"  {performer_name}: {len(scraped_events)} found, {added} added")
                else:
                    consecutive_failures += 1

            except Exception as e:
                consecutive_failures += 1
                print(f"  Error fetching {performer_name}: {e}")

        # === Phase 2: Browse discovery (same browser session) ===
        elapsed = time.monotonic() - start_time
        remaining_budget = max_events - len(events)

        if remaining_budget > 0 and elapsed < TIME_BUDGET_SECONDS:
            print(f"\n  Phase 2: Browse concert discovery (budget: {remaining_budget} events)...")
            consecutive_failures = 0

            try:
                browse_events = await scraper.browse_concerts(
                    max_pages=10,
                    max_events=remaining_budget + 50,  # fetch extra to account for dedup/filter losses
                )

                browse_added = 0
                for evt in browse_events:
                    if len(events) >= max_events:
                        break
                    if _add_event({
                        "id": evt.stubhub_event_id,
                        "name": evt.event_name,
                        "url": evt.event_url,
                        "venue": evt.venue_name,
                        "city": evt.city,
                        "min_price": evt.min_price,
                        "ticket_count": evt.ticket_count,
                        "performer": evt.artist_or_team,
                        "event_datetime": evt.event_datetime,
                    }):
                        browse_added += 1

                print(f"  Browse: {len(browse_events)} found, {browse_added} new after dedup/filter")

            except Exception as browse_err:
                print(f"  Browse discovery error: {browse_err}")
        else:
            if remaining_budget <= 0:
                print("\n  Phase 2: Skipped (event budget full)")
            else:
                print(f"\n  Phase 2: Skipped (time budget: {elapsed:.0f}s)")

    print(f"\n  Total events discovered: {len(events)} (dedup: {len(seen_ids)} IDs, {len(seen_venue_dates)} venue-dates)")
    return events[:max_events]


def create_snapshots_from_listings(
    listings: list[Any],
    event_id: str,
    timestamp: datetime,
    days_to_event: int,
) -> list[Any]:
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
                source="vividseats",
            )
        )

    return snapshots


async def collect_listings_for_events(
    events: list[dict[str, Any]],
    data_dir: Path,
    max_listings_per_event: int = 100,
    start_time: float | None = None,
) -> dict[str, Any]:
    """Collect listings for a list of events and create price snapshots.

    Uses a SINGLE browser session for all events to avoid repeated
    browser launches and reduce timeouts. After collecting listings,
    aggregates them into zone-level price snapshots for longitudinal
    tracking.

    Args:
        events: List of event dictionaries
        data_dir: Data storage directory
        max_listings_per_event: Max listings per event
        start_time: Monotonic start time for wall-clock budget (optional)

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

    stats: dict[str, Any] = {
        "events_attempted": len(events),
        "events_succeeded": 0,
        "events_skipped_time": 0,
        "total_listings": 0,
        "total_snapshots": 0,
        "empty_responses": 0,
        "errors": [],
    }

    consecutive_failures = 0

    # Use a SINGLE browser session for all events
    async with VividSeatsScraper(headless=True, delay_seconds=3.0) as scraper:
        for i, event in enumerate(events, 1):
            # Wall-clock time guard
            if start_time is not None:
                elapsed = time.monotonic() - start_time
                if elapsed > TIME_BUDGET_SECONDS:
                    remaining = len(events) - i + 1
                    stats["events_skipped_time"] = remaining
                    print(f"\n  Time budget reached ({elapsed:.0f}s). Skipping {remaining} remaining events.")
                    break

            # Circuit breaker
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"\n  Circuit breaker: {MAX_CONSECUTIVE_FAILURES} consecutive failures. Pausing 60s...")
                await asyncio.sleep(60)
                consecutive_failures = 0

                # Try one more event as a test
                try:
                    test = await scraper.get_event_listings(event["url"], max_listings=1)
                    if not test:
                        remaining = len(events) - i + 1
                        print(f"  Circuit breaker: still failing. Aborting ({remaining} events skipped).")
                        stats["events_skipped_time"] = remaining
                        break
                except Exception:
                    remaining = len(events) - i + 1
                    print(f"  Circuit breaker: still failing. Aborting ({remaining} events skipped).")
                    stats["events_skipped_time"] = remaining
                    break

            print(f"\n[{i}/{len(events)}] {event['name'][:50]}...")

            try:
                # Get listings using persistent browser session
                scraped_listings = await scraper.get_event_listings(
                    event["url"],
                    max_listings=max_listings_per_event,
                )

                if scraped_listings:
                    consecutive_failures = 0

                    # Create event object for conversion
                    scraped_event = ScrapedEvent(
                        stubhub_event_id=event.get("id", ""),
                        event_name=event["name"],
                        artist_or_team=event.get("performer", "Unknown Artist"),
                        venue_name=event["venue"],
                        city=event["city"],
                        event_datetime=event.get("event_datetime") or datetime.now(UTC),
                        event_url=event["url"],
                    )

                    # Convert and save listings (with validation)
                    from ticket_price_predictor.validation.quality import DataValidator
                    validator = DataValidator()
                    listings = []
                    for s in scraped_listings:
                        listing = create_listing_from_scraped(s, scraped_event, timestamp)
                        result = validator.validate_listing(listing)
                        if result.is_valid:
                            listings.append(listing)
                        else:
                            print(f"    Rejected invalid listing: {result.errors}")
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
                    consecutive_failures += 1
                    stats["empty_responses"] += 1
                    print("    No listings found")

            except Exception as e:
                consecutive_failures += 1
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
        default=150,
        help="Maximum events to collect (default: 150)",
    )
    args = parser.parse_args()

    run_start = time.monotonic()
    data_dir = Path("data")
    max_listings_per_event = 100

    # Initialize popularity service
    pop_service = PopularityService()

    max_events = args.max_events
    mode = "URGENT (≤14 days)" if args.urgent else "FULL"

    print("=" * 60)
    print(f"POPULAR EVENTS MONITOR [{mode}]")
    print("=" * 60)
    print(f"Timestamp: {datetime.now(UTC).isoformat()}")
    print(f"Max events: {max_events}")
    print(f"Max listings per event: {max_listings_per_event}")
    print(f"Time budget: {TIME_BUDGET_SECONDS // 60} minutes")
    print()

    # Get popular events with popularity-based ranking
    events = await get_popular_events(
        max_events=max_events,
        popularity_service=pop_service,
    )
    # Filter to near-term events if --urgent
    if args.urgent:
        from datetime import timedelta

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
        start_time=run_start,
    )

    # Summary
    elapsed_total = time.monotonic() - run_start
    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Events attempted: {stats['events_attempted']}")
    print(f"Events succeeded: {stats['events_succeeded']}")
    print(f"Events skipped (time): {stats.get('events_skipped_time', 0)}")
    print(f"Total listings collected: {stats['total_listings']}")
    print(f"Total snapshots saved: {stats.get('total_snapshots', 0)}")
    print(f"Empty responses: {stats.get('empty_responses', 0)}")
    print(f"Total time: {elapsed_total:.0f}s ({elapsed_total / 60:.1f} min)")

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

    print(f"\nCompleted at: {datetime.now(UTC).isoformat()}")


if __name__ == "__main__":
    asyncio.run(main())
