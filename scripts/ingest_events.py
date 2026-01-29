#!/usr/bin/env python3
"""CLI script for ingesting event metadata from Ticketmaster."""

import argparse
import asyncio
import sys
from pathlib import Path

from ticket_price_predictor.config import get_settings
from ticket_price_predictor.ingestion import EventIngestionService
from ticket_price_predictor.schemas import EventType
from ticket_price_predictor.storage import EventRepository


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Ingest event metadata from Ticketmaster API")

    parser.add_argument(
        "--days-ahead",
        type=int,
        default=90,
        help="Number of days into the future to search (default: 90)",
    )

    parser.add_argument(
        "--event-types",
        nargs="+",
        choices=["concert", "sports", "theater"],
        help="Event types to ingest (default: all)",
    )

    parser.add_argument(
        "--cities",
        nargs="+",
        help="Cities to filter by (default: all)",
    )

    parser.add_argument(
        "--max-events",
        type=int,
        help="Maximum events to fetch per category",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Data directory (default: from settings)",
    )

    return parser.parse_args()


async def main() -> int:
    """Run the event ingestion."""
    args = parse_args()
    settings = get_settings()

    # Determine data directory
    data_dir = args.data_dir or settings.data_dir

    print(f"Data directory: {data_dir}")
    print(f"Days ahead: {args.days_ahead}")

    # Parse event types
    event_types = None
    if args.event_types:
        type_map = {
            "concert": EventType.CONCERT,
            "sports": EventType.SPORTS,
            "theater": EventType.THEATER,
        }
        event_types = [type_map[t] for t in args.event_types]
        print(f"Event types: {[t.value for t in event_types]}")

    if args.cities:
        print(f"Cities: {args.cities}")

    # Create repository and service
    repository = EventRepository(data_dir)
    service = EventIngestionService(repository, settings)

    # Run ingestion
    print("\nStarting event ingestion...")
    result = await service.ingest_upcoming_events(
        days_ahead=args.days_ahead,
        event_types=event_types,
        cities=args.cities,
        max_events=args.max_events,
    )

    # Print results
    print("\n--- Ingestion Results ---")
    print(f"Events fetched: {result.events_fetched}")
    print(f"Events saved: {result.events_saved}")
    print(f"Events skipped (duplicates): {result.events_skipped}")

    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for error in result.errors:
            print(f"  - {error}")
        return 1

    print("\nIngestion complete!")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
