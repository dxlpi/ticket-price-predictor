#!/usr/bin/env python3
"""Generate historical training data from setlist.fm + synthetic prices.

This script:
1. Fetches real historical concerts from setlist.fm
2. Generates realistic synthetic price trajectories
3. Saves everything to Parquet for ML training

Usage:
    python scripts/generate_historical_data.py --artist "Blackpink" --max-concerts 50
    python scripts/generate_historical_data.py --artist "Taylor Swift" --year-from 2023
"""

import argparse
import asyncio
from pathlib import Path

from ticket_price_predictor.api import SetlistFMClient
from ticket_price_predictor.storage import EventRepository, SnapshotRepository
from ticket_price_predictor.synthetic import SyntheticPriceGenerator


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate historical training data from setlist.fm"
    )

    parser.add_argument(
        "--artist",
        required=True,
        help="Artist name to search for (e.g., 'Blackpink', 'Taylor Swift')",
    )

    parser.add_argument(
        "--max-concerts",
        type=int,
        default=100,
        help="Maximum number of concerts to fetch (default: 100)",
    )

    parser.add_argument(
        "--year-from",
        type=int,
        help="Only include concerts from this year onwards",
    )

    parser.add_argument(
        "--year-to",
        type=int,
        help="Only include concerts up to this year",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory (default: data)",
    )

    parser.add_argument(
        "--api-key",
        help="Setlist.fm API key (or set SETLISTFM_API_KEY env var)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible price generation (default: 42)",
    )

    return parser.parse_args()


async def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Get API key
    api_key = args.api_key
    if not api_key:
        import os

        api_key = os.environ.get("SETLISTFM_API_KEY")

    if not api_key:
        print("Error: Setlist.fm API key required.")
        print("Get one free at: https://www.setlist.fm/settings/api")
        print("Then set SETLISTFM_API_KEY env var or use --api-key")
        return

    print(f"Artist: {args.artist}")
    print(f"Data directory: {args.data_dir}")
    print()

    # Fetch concerts from setlist.fm
    print("Fetching historical concerts from setlist.fm...")

    async with SetlistFMClient(api_key=api_key) as client:
        concerts = await client.get_artist_concerts(
            artist_name=args.artist,
            max_concerts=args.max_concerts,
            year_from=args.year_from,
            year_to=args.year_to,
        )

    if not concerts:
        print(f"No concerts found for '{args.artist}'")
        return

    print(f"Found {len(concerts)} concerts")

    # Show sample
    print("\nSample concerts:")
    for concert in concerts[:5]:
        print(
            f"  - {concert.event_date.strftime('%Y-%m-%d')} | {concert.venue_name} ({concert.city})"
        )

    # Generate synthetic prices
    print("\nGenerating synthetic price trajectories...")

    generator = SyntheticPriceGenerator(seed=args.seed)
    trajectories = generator.generate_batch(concerts)

    # Count totals
    total_events = len(trajectories)
    total_snapshots = sum(len(t.snapshots) for t in trajectories)

    print(f"Generated {total_events} events with {total_snapshots} price snapshots")

    # Save to storage
    print("\nSaving to Parquet...")

    event_repo = EventRepository(args.data_dir)
    snapshot_repo = SnapshotRepository(args.data_dir)

    events_saved = 0
    snapshots_saved = 0

    for trajectory in trajectories:
        saved = event_repo.save_events([trajectory.event])
        events_saved += saved

        saved = snapshot_repo.save_snapshots(trajectory.snapshots)
        snapshots_saved += saved

    print(f"Saved {events_saved} events and {snapshots_saved} snapshots")

    # Summary
    print("\n" + "=" * 50)
    print("GENERATION COMPLETE")
    print("=" * 50)
    print(f"Artist: {args.artist}")
    print(f"Concerts processed: {len(concerts)}")
    print(f"Events saved: {events_saved}")
    print(f"Snapshots saved: {snapshots_saved}")

    # Show price range info
    if trajectories:
        all_prices = [s.price_avg for t in trajectories for s in t.snapshots]
        print(f"\nPrice range: ${min(all_prices):.2f} - ${max(all_prices):.2f}")


if __name__ == "__main__":
    asyncio.run(main())
