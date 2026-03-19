#!/usr/bin/env python3
"""Collect real seat-level ticket listings from Vivid Seats.

This script scrapes ticket listings with full seat-level detail:
- Section, row, and seat numbers
- Current listing price
- Quantity available

Usage:
    python scripts/collect_listings.py --artist "Eagles"
    python scripts/collect_listings.py --artist "Taylor Swift" --max-events 5
    python scripts/collect_listings.py --event-url "https://www.vividseats.com/..."
"""

import argparse
import asyncio
from pathlib import Path

from ticket_price_predictor.ingestion import DataSource, ListingCollector
from ticket_price_predictor.storage import ListingRepository


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect real seat-level ticket listings from Vivid Seats"
    )

    parser.add_argument(
        "--artist",
        help="Artist name to search for (e.g., 'Blackpink', 'Taylor Swift')",
    )

    parser.add_argument(
        "--event-url",
        help="Direct URL to a Vivid Seats event page",
    )

    parser.add_argument(
        "--max-events",
        type=int,
        default=5,
        help="Maximum number of events to process per artist (default: 5)",
    )

    parser.add_argument(
        "--max-listings",
        type=int,
        default=500,
        help="Maximum listings per event (default: 500)",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between requests in seconds (default: 2.0)",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory (default: data)",
    )

    parser.add_argument(
        "--show-sample",
        action="store_true",
        help="Show sample of collected listings",
    )

    return parser.parse_args()


async def main() -> None:
    """Main entry point."""
    args = parse_args()

    if not args.artist and not args.event_url:
        print("Error: Must provide either --artist or --event-url")
        return

    print("=" * 50)
    print("TICKET LISTING COLLECTOR")
    print("=" * 50)
    print()
    print(f"Data source: VIVIDSEATS")
    print(f"Data directory: {args.data_dir}")
    print(f"Request delay: {args.delay}s")
    print()

    collector = ListingCollector(
        data_dir=args.data_dir,
        delay_seconds=args.delay,
        source=DataSource.VIVIDSEATS,
    )

    if args.artist:
        print(f"Searching for: {args.artist}")
        print(f"Max events: {args.max_events}")
        print()

        result = await collector.collect_for_artist(
            artist_name=args.artist,
            max_events=args.max_events,
            max_listings_per_event=args.max_listings,
        )

    elif args.event_url:
        print(f"Event URL: {args.event_url}")
        print()

        result = await collector.collect_for_event_url(
            event_url=args.event_url,
            max_listings=args.max_listings,
        )

    else:
        print("No search criteria provided")
        return

    # Print results
    print()
    print("=" * 50)
    print("COLLECTION RESULTS")
    print("=" * 50)
    print(f"Events found: {result.events_found}")
    print(f"Events processed: {result.events_processed}")
    print(f"Listings collected: {result.listings_collected}")
    print(f"Listings saved: {result.listings_saved}")

    if result.errors:
        print()
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")

    # Show sample if requested
    if args.show_sample and result.listings_saved > 0:
        print()
        print("=" * 50)
        print("SAMPLE LISTINGS")
        print("=" * 50)

        repo = ListingRepository(args.data_dir)
        listings = repo.get_listings()[:10]

        for listing in listings:
            markup = f"{listing.markup_ratio:.1f}x" if listing.markup_ratio else "N/A"
            print(
                f"  {listing.section:15} | Row {listing.row:5} | "
                f"${listing.listing_price:>8.2f} | Markup: {markup}"
            )

    print()
    if result.success:
        print("Collection completed successfully!")
    else:
        print("Collection completed with errors.")


if __name__ == "__main__":
    asyncio.run(main())
