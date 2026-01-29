#!/usr/bin/env python3
"""Combined pipeline runner for event ingestion and snapshot collection."""

import argparse
import asyncio
import sys
from pathlib import Path

from ticket_price_predictor.config import get_settings
from ticket_price_predictor.ingestion import EventIngestionService, SnapshotCollector
from ticket_price_predictor.schemas import EventType
from ticket_price_predictor.storage import EventRepository, SnapshotRepository


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run full data pipeline: ingest events + collect snapshots"
    )

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
        "--skip-ingestion",
        action="store_true",
        help="Skip event ingestion, only collect snapshots",
    )

    parser.add_argument(
        "--skip-snapshots",
        action="store_true",
        help="Skip snapshot collection, only ingest events",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Data directory (default: from settings)",
    )

    return parser.parse_args()


async def main() -> int:
    """Run the full pipeline."""
    args = parse_args()
    settings = get_settings()

    # Determine data directory
    data_dir = args.data_dir or settings.data_dir

    print("=" * 50)
    print("TICKET PRICE PREDICTOR - DATA PIPELINE")
    print("=" * 50)
    print(f"\nData directory: {data_dir}")

    # Create repositories
    event_repo = EventRepository(data_dir)
    snapshot_repo = SnapshotRepository(data_dir)

    errors = []

    # Step 1: Event Ingestion
    if not args.skip_ingestion:
        print("\n--- STEP 1: Event Ingestion ---")

        # Parse event types
        event_types = None
        if args.event_types:
            type_map = {
                "concert": EventType.CONCERT,
                "sports": EventType.SPORTS,
                "theater": EventType.THEATER,
            }
            event_types = [type_map[t] for t in args.event_types]

        ingestion_service = EventIngestionService(event_repo, settings)
        ingestion_result = await ingestion_service.ingest_upcoming_events(
            days_ahead=args.days_ahead,
            event_types=event_types,
            cities=args.cities,
            max_events=args.max_events,
        )

        print(f"  Events fetched: {ingestion_result.events_fetched}")
        print(f"  Events saved: {ingestion_result.events_saved}")
        print(f"  Events skipped: {ingestion_result.events_skipped}")

        if ingestion_result.errors:
            errors.extend(ingestion_result.errors)
            print(f"  Errors: {len(ingestion_result.errors)}")
    else:
        print("\n--- STEP 1: Event Ingestion (SKIPPED) ---")

    # Step 2: Snapshot Collection
    if not args.skip_snapshots:
        print("\n--- STEP 2: Snapshot Collection ---")

        event_ids = event_repo.list_event_ids()
        print(f"  Tracked events: {len(event_ids)}")

        if event_ids:
            collector = SnapshotCollector(event_repo, snapshot_repo, settings)
            collection_result = await collector.collect_snapshots(event_ids=event_ids)

            print(f"  Events processed: {collection_result.events_processed}")
            print(f"  Snapshots created: {collection_result.snapshots_created}")
            print(f"  Snapshots saved: {collection_result.snapshots_saved}")

            if collection_result.errors:
                errors.extend(collection_result.errors)
                print(f"  Errors: {len(collection_result.errors)}")
        else:
            print("  No events to process")
    else:
        print("\n--- STEP 2: Snapshot Collection (SKIPPED) ---")

    # Summary
    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)

    if errors:
        print(f"\nTotal errors: {len(errors)}")
        for error in errors[:5]:
            print(f"  - {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")
        return 1

    print("\nAll steps completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
