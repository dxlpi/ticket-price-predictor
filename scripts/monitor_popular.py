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
import re
from datetime import datetime
from pathlib import Path

from playwright.async_api import async_playwright
from playwright_stealth import Stealth

from ticket_price_predictor.ingestion import DataSource, ListingCollector
from ticket_price_predictor.storage import ListingRepository


async def get_popular_events(max_events: int = 20) -> list[dict]:
    """Fetch top popular events from Vivid Seats.

    Gets events from popular performers (not parking passes or small events).

    Args:
        max_events: Maximum number of events to return

    Returns:
        List of event dictionaries with name, url, etc.
    """
    print("Fetching popular performers and their events...")

    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=True)
    context = await browser.new_context(
        viewport={"width": 1920, "height": 1080},
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    )
    page = await context.new_page()
    stealth = Stealth()
    await stealth.apply_stealth_async(page)

    events = []

    # Get popular performers from concerts page
    await page.goto("https://www.vividseats.com/concerts", wait_until="load")
    await asyncio.sleep(3)

    html = await page.content()
    match = re.search(r'<script id="__NEXT_DATA__"[^>]*>([^<]+)</script>', html)

    performers = []
    if match:
        data = json.loads(match.group(1))
        page_props = data.get("props", {}).get("pageProps", {})

        # Get popular performers
        pop_data = page_props.get("initialPopularPerformersData", {})
        if isinstance(pop_data, dict):
            performers = pop_data.get("items", [])
        elif isinstance(pop_data, list):
            performers = pop_data

        print(f"  Found {len(performers)} popular performers")

    # For each popular performer, get their upcoming events
    for performer in performers[:10]:  # Top 10 performers
        if len(events) >= max_events:
            break

        performer_name = performer.get("name", "Unknown")
        web_path = performer.get("webPath", "")

        if not web_path:
            continue

        try:
            performer_url = f"https://www.vividseats.com{web_path}"
            await page.goto(performer_url, wait_until="load")
            await asyncio.sleep(2)

            html = await page.content()
            match = re.search(r'<script id="__NEXT_DATA__"[^>]*>([^<]+)</script>', html)

            if match:
                data = json.loads(match.group(1))
                page_props = data.get("props", {}).get("pageProps", {})

                # Get production list (events)
                prod_data = page_props.get("initialProductionListData", {})
                items = prod_data.get("items", [])

                # Get events with tickets
                for item in items[:3]:  # Up to 3 events per performer
                    if len(events) >= max_events:
                        break

                    event_web_path = item.get("webPath", "")
                    ticket_count = item.get("ticketCount", 0)

                    # Skip events with no tickets
                    if not event_web_path or ticket_count < 10:
                        continue

                    events.append({
                        "id": str(item.get("id", "")),
                        "name": item.get("name", "Unknown"),
                        "url": f"https://www.vividseats.com{event_web_path}",
                        "venue": item.get("venue", {}).get("name", "Unknown"),
                        "city": item.get("venue", {}).get("city", "Unknown"),
                        "min_price": item.get("minPrice"),
                        "ticket_count": ticket_count,
                        "performer": performer_name,
                    })

                print(f"  {performer_name}: {len(items)} events, added {min(len(items), 3)}")

        except Exception as e:
            print(f"  Error fetching {performer_name}: {e}")

    await browser.close()
    await playwright.stop()

    return events[:max_events]


async def collect_listings_for_events(
    events: list[dict],
    data_dir: Path,
    max_listings_per_event: int = 100,
) -> dict:
    """Collect listings for a list of events.

    Args:
        events: List of event dictionaries
        data_dir: Data storage directory
        max_listings_per_event: Max listings per event

    Returns:
        Summary statistics
    """
    collector = ListingCollector(
        data_dir=data_dir,
        delay_seconds=3.0,
        source=DataSource.VIVIDSEATS,
    )

    stats = {
        "events_attempted": len(events),
        "events_succeeded": 0,
        "total_listings": 0,
        "errors": [],
    }

    for i, event in enumerate(events, 1):
        print(f"\n[{i}/{len(events)}] {event['name'][:50]}...")

        try:
            result = await collector.collect_for_event_url(
                event_url=event["url"],
                event_name=event["name"],
                artist_name=event.get("performer", "Unknown Artist"),
                venue_name=event["venue"],
                city=event["city"],
                max_listings=max_listings_per_event,
            )

            if result.listings_saved > 0:
                stats["events_succeeded"] += 1
                stats["total_listings"] += result.listings_saved
                print(f"    Collected {result.listings_saved} listings")
            else:
                print(f"    No listings found")

            if result.errors:
                stats["errors"].extend(result.errors)

        except Exception as e:
            error_msg = f"{event['name']}: {e}"
            stats["errors"].append(error_msg)
            print(f"    Error: {e}")

    return stats


async def main() -> None:
    """Main entry point."""
    data_dir = Path("data")
    max_events = 20
    max_listings_per_event = 100

    print("=" * 60)
    print("POPULAR EVENTS MONITOR")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Max events: {max_events}")
    print(f"Max listings per event: {max_listings_per_event}")
    print()

    # Get popular events
    events = await get_popular_events(max_events=max_events)
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
