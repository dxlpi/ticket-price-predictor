"""Train the sale probability (CVR) classifier."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ticket_price_predictor.ml.training.data_loader import DataLoader
from ticket_price_predictor.ml.training.sale_trainer import SaleProbabilityTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sale probability classifier")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--output", default="data/models", help="Output directory")
    parser.add_argument("--version", default="v1", help="Model version")
    parser.add_argument("--window-hours", type=int, default=48, help="Label window hours")
    parser.add_argument("--min-absent-scrapes", type=int, default=2, help="Min consecutive absences for sold (disappearance strategy only)")
    parser.add_argument(
        "--label-strategy",
        choices=["inventory_depletion", "disappearance"],
        default="inventory_depletion",
        help=(
            "Label construction strategy. "
            "'inventory_depletion' (default): uses aggregate inventory depletion from snapshot data. "
            "'disappearance': uses individual listing disappearance (legacy, unreliable with partial scraping)."
        ),
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SALE PROBABILITY MODEL TRAINING")
    print("=" * 60)
    print(f"\nVersion:        {args.version}")
    print(f"Label strategy: {args.label_strategy}")
    print(f"Label window:   {args.window_hours}h")
    if args.label_strategy == "disappearance":
        print(f"Min absent scrapes: {args.min_absent_scrapes}")

    # Load listings data via DataLoader (uses repository pattern)
    data_dir = Path(args.data_dir)
    loader = DataLoader(data_dir)

    print(f"\nLoading listings from {data_dir / 'raw' / 'listings'}...")
    listings_df = loader.load_all_listings()

    if listings_df.empty:
        print("\nERROR: No listings found")
        sys.exit(1)

    print(f"Loaded {len(listings_df):,} listings")
    print(f"  - {listings_df['event_id'].nunique()} events")

    # Check required columns
    required = {"listing_id", "event_id", "timestamp", "event_datetime", "listing_price"}
    missing = required - set(listings_df.columns)
    if missing:
        print(f"\nERROR: Missing required columns: {missing}")
        print(f"Available columns: {sorted(listings_df.columns)}")
        sys.exit(1)

    # Ensure timestamp columns are datetime
    for col in ["timestamp", "event_datetime"]:
        if col in listings_df.columns and not pd.api.types.is_datetime64_any_dtype(listings_df[col]):
            listings_df[col] = pd.to_datetime(listings_df[col])

    # Load snapshots if using inventory depletion strategy
    snapshots_df = None
    if args.label_strategy == "inventory_depletion":
        print(f"\nLoading snapshots from {data_dir / 'raw' / 'snapshots'}...")
        snapshots_df = loader.load_snapshots()
        if snapshots_df.empty:
            print("\nERROR: No snapshot data found. Use --label-strategy disappearance or collect snapshots first.")
            sys.exit(1)
        print(f"Loaded {len(snapshots_df):,} snapshots")
        print(f"  - {snapshots_df['event_id'].nunique()} events")
        null_inv = snapshots_df["inventory_remaining"].isna().sum()
        if null_inv > 0:
            print(f"  - {null_inv} rows with null inventory_remaining (will be dropped)")

    # Train
    trainer = SaleProbabilityTrainer(
        label_window_hours=args.window_hours,
        min_absent_scrapes=args.min_absent_scrapes,
        label_strategy=args.label_strategy,
    )

    print("\nTraining sale probability classifier...")
    model, metrics = trainer.train(listings_df, model_version=args.version, snapshots_df=snapshots_df)

    # Print results
    print("\n" + "=" * 60)
    print(f"RESULTS (v{args.version})")
    print("=" * 60)
    print(f"Training samples:   {metrics.n_train_samples:,}")
    print(f"Validation samples: {metrics.n_val_samples:,}")
    print(f"Test samples:       {metrics.n_test_samples:,}")
    print(f"Features:           {metrics.n_features}")
    print(f"\nAUC-ROC:     {metrics.auc_roc:.4f}" if metrics.auc_roc else "\nAUC-ROC:     N/A")
    print(f"Precision:   {metrics.precision:.4f}" if metrics.precision else "Precision:   N/A")
    print(f"Recall:      {metrics.recall:.4f}" if metrics.recall else "Recall:      N/A")
    print(f"F1:          {metrics.f1:.4f}" if metrics.f1 else "F1:          N/A")
    print(f"ECE:         {metrics.calibration_error:.4f}" if metrics.calibration_error else "ECE:         N/A")
    print(f"Best iter:   {metrics.best_iteration}")
    print(f"Train time:  {metrics.training_time_seconds:.1f}s")

    if metrics.feature_importance:
        print("\nTop features:")
        for i, (name, imp) in enumerate(metrics.feature_importance.items(), 1):
            print(f"  {i}. {name}: {imp:.4f}")

    # Save
    output_dir = Path(args.output)
    trainer.save(model, metrics, output_dir, version=args.version)
    print(f"\nModel saved to: {output_dir}/sale_probability_{args.version}.joblib")
    print(f"Metrics saved to: {output_dir}/sale_probability_{args.version}_metrics.json")


if __name__ == "__main__":
    main()
