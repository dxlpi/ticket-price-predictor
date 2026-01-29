#!/usr/bin/env python3
"""CLI script for collecting price snapshots."""

import argparse
import asyncio
import sys
from pathlib import Path

from ticket_price_predictor.config import get_settings
from ticket_price_predictor.ingestion import SnapshotCollector
from ticket_price_predictor.storage import EventRepository, SnapshotRepository


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Collect price snapshots for tracked events")

    parser.add_argument(
        "--event-ids",
        nargs="+",
        help="Specific event IDs to collect snapshots for",
    )

    parser.add_argument(
        "--all-tracked",
        action="store_true",
        help="Collect snapshots for all tracked events",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Data directory (default: from settings)",
    )

    return parser.parse_args()


async def main() -> int:
    """Run the snapshot collection."""
    args = parse_args()
    settings = get_settings()

    # Determine data directory
    data_dir = args.data_dir or settings.data_dir

    print(f"Data directory: {data_dir}")

    # Create repositories and collector
    event_repo = EventRepository(data_dir)
    snapshot_repo = SnapshotRepository(data_dir)
    collector = SnapshotCollector(event_repo, snapshot_repo, settings)

    # Determine which events to process
    event_ids = args.event_ids
    if args.all_tracked or event_ids is None:
        event_ids = event_repo.list_event_ids()
        print(f"Found {len(event_ids)} tracked events")
    else:
        print(f"Processing {len(event_ids)} specified events")

    if not event_ids:
        print("No events to process. Run ingest_events.py first.")
        return 0

    # Run collection
    print("\nCollecting snapshots...")
    result = await collector.collect_snapshots(event_ids=event_ids)

    # Print results
    print("\n--- Collection Results ---")
    print(f"Events processed: {result.events_processed}")
    print(f"Snapshots created: {result.snapshots_created}")
    print(f"Snapshots saved: {result.snapshots_saved}")

    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for error in result.errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(result.errors) > 10:
            print(f"  ... and {len(result.errors) - 10} more errors")
        return 1

    print("\nCollection complete!")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
