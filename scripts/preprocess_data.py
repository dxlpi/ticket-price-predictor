#!/usr/bin/env python3
"""Batch preprocessing of ticket data.

Usage:
    python scripts/preprocess_data.py --input data/listings --output data/preprocessed
    python scripts/preprocess_data.py --dataset listings --dry-run
    python scripts/preprocess_data.py --dataset events --incremental
    python scripts/preprocess_data.py --config preprocessing_config.json
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ticket_price_predictor.preprocessing import (
    PreprocessingConfig,
    PreprocessingPipeline,
    PipelineBuilder,
    QualityReporter,
)
from ticket_price_predictor.storage import EventRepository, ListingRepository


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Batch preprocessing of ticket data")

    parser.add_argument(
        "--input",
        type=Path,
        help="Input directory or parquet file",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for preprocessed data",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["listings", "events", "snapshots"],
        help="Dataset type (uses preset pipeline)",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory (default: data)",
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to preprocessing config JSON",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate report only, don't write output",
    )

    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Process only new data since last run",
    )

    parser.add_argument(
        "--watermark-file",
        type=Path,
        default=Path(".omc/state/preprocessing_watermark.json"),
        help="Watermark file for incremental processing",
    )

    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate quality report without preprocessing",
    )

    return parser.parse_args()


def load_watermark(watermark_file: Path) -> dict:
    """Load processing watermark for incremental runs."""
    if watermark_file.exists():
        with open(watermark_file) as f:
            return json.load(f)
    return {"listings": {}, "events": {}, "snapshots": {}}


def save_watermark(watermark_file: Path, watermark: dict) -> None:
    """Save processing watermark."""
    watermark_file.parent.mkdir(parents=True, exist_ok=True)
    with open(watermark_file, "w") as f:
        json.dump(watermark, f, indent=2, default=str)


def get_new_partitions(
    data_dir: Path,
    dataset_type: str,
    last_partition: str | None,
) -> list[Path]:
    """Get partitions newer than the watermark.

    Args:
        data_dir: Base data directory
        dataset_type: Type of dataset (listings, events, snapshots)
        last_partition: Last processed partition (e.g., "2025/01/15")

    Returns:
        List of partition paths to process
    """
    if dataset_type == "listings":
        # Listings partitioned as data/listings/year=YYYY/month=MM/day=DD/
        listings_dir = data_dir / "listings"
        if not listings_dir.exists():
            return []

        partitions = []
        for year_dir in sorted(listings_dir.glob("year=*")):
            for month_dir in sorted(year_dir.glob("month=*")):
                for day_dir in sorted(month_dir.glob("day=*")):
                    partition_key = f"{year_dir.name.split('=')[1]}/{month_dir.name.split('=')[1]}/{day_dir.name.split('=')[1]}"
                    if last_partition is None or partition_key > last_partition:
                        partitions.append(day_dir)

        return partitions

    elif dataset_type == "events":
        # Events stored as single file
        events_file = data_dir / "events.parquet"
        if events_file.exists():
            return [events_file]
        return []

    elif dataset_type == "snapshots":
        # Snapshots partitioned as data/snapshots/year=YYYY/month=MM/
        snapshots_dir = data_dir / "snapshots"
        if not snapshots_dir.exists():
            return []

        partitions = []
        for year_dir in sorted(snapshots_dir.glob("year=*")):
            for month_dir in sorted(year_dir.glob("month=*")):
                partition_key = f"{year_dir.name.split('=')[1]}/{month_dir.name.split('=')[1]}"
                if last_partition is None or partition_key > last_partition:
                    partitions.append(month_dir)

        return partitions

    return []


def load_data(
    input_path: Path | None,
    dataset_type: str | None,
    data_dir: Path,
    incremental: bool,
    watermark: dict,
) -> tuple[pd.DataFrame, str | None]:
    """Load data for preprocessing.

    Returns:
        Tuple of (DataFrame, latest_partition_key)
    """
    if input_path:
        # Load from specified path
        print(f"Loading data from: {input_path}")
        if input_path.is_file():
            return pd.read_parquet(input_path), None
        else:
            # Load directory
            return pd.read_parquet(input_path), None

    elif dataset_type:
        # Load using repository pattern
        if dataset_type == "listings":
            repo = ListingRepository(data_dir)

            if incremental:
                last_partition = watermark.get("listings", {}).get("last_partition")
                partitions = get_new_partitions(data_dir, "listings", last_partition)

                if not partitions:
                    print("No new partitions to process.")
                    return pd.DataFrame(), None

                print(f"Processing {len(partitions)} new partition(s)")
                dfs = []
                latest_partition = None

                for partition in partitions:
                    df = pd.read_parquet(partition)
                    dfs.append(df)
                    # Extract partition key
                    parts = str(partition).split("/")
                    latest_partition = "/".join(
                        [p.split("=")[1] for p in parts if "=" in p]
                    )

                return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(), latest_partition

            else:
                # Load all listings
                return repo.load_all_listings(), None

        elif dataset_type == "events":
            repo = EventRepository(data_dir)
            return repo.load_all_events(), None

        elif dataset_type == "snapshots":
            # Snapshots don't have a dedicated repository method yet
            snapshots_dir = data_dir / "snapshots"
            if not snapshots_dir.exists():
                return pd.DataFrame(), None
            return pd.read_parquet(snapshots_dir), None

    raise ValueError("Must specify either --input or --dataset")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)
    print()

    # Load configuration
    if args.config:
        print(f"Loading config from: {args.config}")
        with open(args.config) as f:
            config_dict = json.load(f)
        config = PreprocessingConfig(**config_dict)
    else:
        config = PreprocessingConfig()

    # Load watermark for incremental processing
    watermark = load_watermark(args.watermark_file) if args.incremental else {}

    # Load data
    df, latest_partition = load_data(
        args.input,
        args.dataset,
        args.data_dir,
        args.incremental,
        watermark,
    )

    if df.empty:
        print("No data to process.")
        return

    print(f"Loaded {len(df):,} rows")
    print()

    # Build pipeline
    if args.dataset:
        print(f"Using '{args.dataset}' preset pipeline")
        builder = PipelineBuilder(config)
        pipeline = builder.build_preset(args.dataset)  # type: ignore
    else:
        print("Using default pipeline")
        builder = PipelineBuilder(config)
        pipeline = builder.build_default()

    # Report-only mode
    if args.report_only:
        print("Generating quality report (no preprocessing)...")
        from ticket_price_predictor.preprocessing.base import ProcessingResult

        result = ProcessingResult(
            data=df,
            metrics={"input_rows": len(df)},
        )
    else:
        # Run preprocessing
        if args.dry_run:
            print("DRY RUN: Preprocessing (no output will be written)")
        else:
            print("Running preprocessing...")

        result = pipeline.process(df)

        print(f"Processed {len(result.data):,} rows")
        print(f"Issues found: {len(result.issues)}")
        print()

    # Generate quality report
    print("Generating quality report...")
    reporter = QualityReporter(config)
    metrics = reporter.extract_metrics(result)

    print()
    print(reporter.generate_text_summary(metrics))
    print()

    # Save outputs
    if not args.dry_run and not args.report_only:
        if args.output:
            output_path = args.output
            output_path.mkdir(parents=True, exist_ok=True)

            # Write preprocessed data
            data_file = output_path / "preprocessed.parquet"
            result.data.to_parquet(data_file, index=False)
            print(f"Preprocessed data saved to: {data_file}")

            # Write quality report
            report_file = output_path / "quality_report.json"
            with open(report_file, "w") as f:
                f.write(reporter.generate_json_export(metrics))
            print(f"Quality report saved to: {report_file}")

            # Write issues log
            if result.issues:
                issues_file = output_path / "issues.log"
                with open(issues_file, "w") as f:
                    f.write("\n".join(result.issues))
                print(f"Issues log saved to: {issues_file}")

        # Update watermark for incremental processing
        if args.incremental and latest_partition and args.dataset:
            watermark[args.dataset]["last_partition"] = latest_partition
            watermark[args.dataset]["last_processed_ts"] = datetime.now(UTC).isoformat()
            save_watermark(args.watermark_file, watermark)
            print(f"Watermark updated: {latest_partition}")

    print()
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
