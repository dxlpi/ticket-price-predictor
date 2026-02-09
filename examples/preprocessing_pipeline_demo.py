"""Demo script for preprocessing pipeline functionality.

This script demonstrates:
1. Preset pipelines (listings, events, snapshots)
2. Custom pipeline construction
3. Checkpoint support
4. Error handling and metrics aggregation
"""

import sys
from pathlib import Path

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ticket_price_predictor.preprocessing import (
    PipelineBuilder,
    PreprocessingConfig,
    PreprocessingPipeline,
)
from ticket_price_predictor.preprocessing.cleaners import (
    PriceOutlierHandler,
    TextNormalizer,
)
from ticket_price_predictor.preprocessing.validators import SchemaValidator
from ticket_price_predictor.schemas import SeatZone


def demo_listings_pipeline():
    """Demonstrate listings preprocessing pipeline."""
    print("\n=== LISTINGS PIPELINE DEMO ===")

    # Create sample listings data
    listings_df = pd.DataFrame(
        {
            "listing_id": ["L1", "L2", "L3"],
            "event_id": ["E1", "E1", "E2"],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "event_datetime": pd.to_datetime(
                ["2024-02-01", "2024-02-01", "2024-02-15"]
            ),
            "section": ["Floor 1", "Upper 201", "Balcony A"],
            "row": ["A", "B", "C"],
            "quantity": [2, 4, 2],
            "listing_price": [150.0, 75.0, 50.0],
            "total_price": [300.0, 300.0, 100.0],
            "days_to_event": [31, 30, 43],
            "artist_or_team": ["Bruno Mars", "Bruno Mars", "Taylor Swift"],
            "venue_name": ["MGM Grand", "MGM Grand", "Madison Square Garden"],
            "city": ["Vegas", "Las Vegas", "NYC"],
        }
    )

    # Create events data for joining
    events_df = pd.DataFrame(
        {
            "event_id": ["E1", "E2"],
            "venue_capacity": [17000, 20000],
        }
    )

    # Build and run pipeline
    pipeline = PipelineBuilder.build_listings_pipeline(events_df=events_df)
    result = pipeline.process(listings_df)

    print(f"Input rows: {len(listings_df)}")
    print(f"Output rows: {len(result.data)}")
    print(f"Columns added: {len(result.data.columns) - len(listings_df.columns)}")
    print(f"Issues raised: {len(result.issues)}")
    print(f"Pipeline stages: {len(result.metrics['stages'])}")

    # Show added columns
    new_cols = set(result.data.columns) - set(listings_df.columns)
    print(f"New columns: {sorted(new_cols)}")

    return result


def demo_events_pipeline():
    """Demonstrate events preprocessing pipeline."""
    print("\n=== EVENTS PIPELINE DEMO ===")

    events_df = pd.DataFrame(
        {
            "event_id": ["E1", "E2", "E3"],
            "event_type": ["concert", "sports", "concert"],
            "event_datetime": pd.to_datetime(
                ["2024-02-01", "2024-02-02", "2024-03-15"]
            ),
            "artist_or_team": ["Bruno Mars", "Lakers", "Taylor Swift"],
            "venue_id": ["V1", "V2", "V3"],
            "venue_name": ["MGM Grand", "Crypto.com Arena", "Madison Square Garden"],
            "city": ["Las Vegas", "LA", "New York"],
            "venue_capacity": [17000, 19000, 20000],
        }
    )

    pipeline = PipelineBuilder.build_events_pipeline()
    result = pipeline.process(events_df)

    print(f"Input rows: {len(events_df)}")
    print(f"Output rows: {len(result.data)}")
    print(f"Issues raised: {len(result.issues)}")

    return result


def demo_snapshots_pipeline():
    """Demonstrate snapshots preprocessing pipeline."""
    print("\n=== SNAPSHOTS PIPELINE DEMO ===")

    snapshots_df = pd.DataFrame(
        {
            "event_id": ["E1", "E1", "E2"],
            "seat_zone": [
                SeatZone.FLOOR_VIP.value,
                SeatZone.LOWER_TIER.value,
                SeatZone.UPPER_TIER.value,
            ],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
            "price_min": [100.0, 75.0, 50.0],
            "price_avg": [150.0, 100.0, 75.0],
            "price_max": [200.0, 150.0, 100.0],
            "inventory_remaining": [50, 100, 200],
            "days_to_event": [31, 31, 45],
        }
    )

    pipeline = PipelineBuilder.build_snapshots_pipeline()
    result = pipeline.process(snapshots_df)

    print(f"Input rows: {len(snapshots_df)}")
    print(f"Output rows: {len(result.data)}")
    print(f"Issues raised: {len(result.issues)}")

    return result


def demo_custom_pipeline():
    """Demonstrate custom pipeline construction."""
    print("\n=== CUSTOM PIPELINE DEMO ===")

    # Create minimal data
    df = pd.DataFrame(
        {
            "artist_or_team": ["Artist A", "Artist B"],
            "venue_name": ["Venue 1", "Venue 2"],
            "city": ["NYC", "LA"],
            "listing_price": [150.0, 75.0],
        }
    )

    # Build custom pipeline with specific stages
    config = PreprocessingConfig(iqr_multiplier=2.0)  # Custom config
    pipeline = PipelineBuilder.build_custom_pipeline(
        stages=[
            TextNormalizer(config),
            PriceOutlierHandler(config),
        ],
        name="minimal_cleaning",
    )

    result = pipeline.process(df)

    print(f"Custom pipeline '{pipeline.name}'")
    print(f"Input rows: {len(df)}")
    print(f"Output rows: {len(result.data)}")
    print(f"Stages: {list(result.metrics['stages'].keys())}")

    return result


def demo_checkpoints():
    """Demonstrate checkpoint functionality."""
    print("\n=== CHECKPOINT DEMO ===")

    listings_df = pd.DataFrame(
        {
            "listing_id": ["L1", "L2"],
            "event_id": ["E1", "E2"],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "event_datetime": pd.to_datetime(["2024-02-01", "2024-02-02"]),
            "section": ["Floor 1", "Upper 201"],
            "row": ["A", "B"],
            "quantity": [2, 4],
            "listing_price": [150.0, 75.0],
            "total_price": [300.0, 300.0],
            "days_to_event": [31, 31],
            "artist_or_team": ["Artist A", "Artist B"],
            "venue_name": ["Venue 1", "Venue 2"],
            "city": ["NYC", "LA"],
        }
    )

    checkpoint_dir = Path("data/.checkpoints")
    pipeline = PipelineBuilder.build_listings_pipeline(checkpoint_dir=checkpoint_dir)

    print(f"Running pipeline with checkpoints in: {checkpoint_dir}")
    result = pipeline.process(listings_df)

    # Check created checkpoints
    checkpoint_files = list(checkpoint_dir.glob("listings_stage_*.parquet"))
    print(f"Checkpoints created: {len(checkpoint_files)}")

    # Demonstrate resume
    if checkpoint_files:
        print(f"Resuming from stage 1...")
        resumed_df = pipeline.resume_from_checkpoint(1)
        print(f"Resumed DataFrame: {len(resumed_df)} rows")

    # Cleanup
    import shutil

    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    print("Checkpoints cleaned up")

    return result


def main():
    """Run all demos."""
    print("=" * 60)
    print("PREPROCESSING PIPELINE DEMO")
    print("=" * 60)

    demo_listings_pipeline()
    demo_events_pipeline()
    demo_snapshots_pipeline()
    demo_custom_pipeline()
    demo_checkpoints()

    print("\n" + "=" * 60)
    print("ALL DEMOS COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
