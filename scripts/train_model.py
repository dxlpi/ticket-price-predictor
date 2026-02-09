#!/usr/bin/env python3
"""Train a ticket price prediction model.

Usage:
    python scripts/train_model.py  # trains LightGBM model (default)
    python scripts/train_model.py --model quantile --version v3
    python scripts/train_model.py --model baseline --output data/models/
    python scripts/train_model.py --from-study lightgbm_aggressive --version v4
"""

import argparse
from pathlib import Path

import optuna

from ticket_price_predictor.ml.training.data_loader import DataLoader
from ticket_price_predictor.ml.training.trainer import ModelTrainer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a price prediction model")

    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline", "lightgbm", "quantile"],
        default="lightgbm",
        help="Model type to train (default: lightgbm)",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory (default: data)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/models"),
        help="Output directory for model (default: data/models)",
    )

    parser.add_argument(
        "--version",
        type=str,
        default="v3",
        help="Model version (default: v3)",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training data ratio (default: 0.7)",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation data ratio (default: 0.15)",
    )

    parser.add_argument(
        "--from-study",
        type=str,
        default=None,
        help="Load best hyperparameters from Optuna study",
    )

    parser.add_argument(
        "--trial-id",
        type=int,
        default=None,
        help="Use specific trial ID from study (requires --from-study)",
    )

    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Enable preprocessing pipeline before training",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("TICKET PRICE MODEL TRAINING")
    print("=" * 60)
    print()
    print(f"Model type: {args.model}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output}")
    print(f"Version: {args.version}")
    print(f"Preprocessing: {'enabled' if args.preprocess else 'disabled'}")

    # Load hyperparameters from Optuna study if specified
    params = None
    if args.from_study:
        print(f"\nLoading hyperparameters from study: {args.from_study}")
        storage = f"sqlite:///data/optuna/studies/{args.from_study}.db"
        study = optuna.load_study(study_name=args.from_study, storage=storage)

        if args.trial_id is not None:
            # Use specific trial
            trial = [t for t in study.trials if t.number == args.trial_id][0]
            print(f"  Using trial #{args.trial_id}")
            print(f"  Trial MAE: ${trial.value:.2f}")
        else:
            # Use best trial
            trial = study.best_trial
            print(f"  Using best trial #{trial.number}")
            print(f"  Best MAE: ${trial.value:.2f}")

        params = trial.params
        print(f"  Loaded {len(params)} hyperparameters")
        print()

    print()

    # Load data
    print("Loading data...")
    loader = DataLoader(args.data_dir)
    df = loader.load_all_listings()

    if df.empty:
        print("ERROR: No data found. Run data collection first.")
        return

    summary = loader.get_summary()
    print(f"Loaded {summary['n_listings']:,} listings")
    print(f"  - {summary['n_events']} events")
    print(f"  - {summary['n_artists']} artists")
    print(f"  - Price range: ${summary['price_min']:.2f} - ${summary['price_max']:.2f}")
    print()

    # Train model
    trainer = ModelTrainer(
        model_type=args.model,  # type: ignore
        model_version=args.version,
    )

    test_ratio = 1.0 - args.train_ratio - args.val_ratio

    # Use train_with_params if we have custom params, otherwise use regular train
    if params:
        # Need to create split and use train_with_params
        from ticket_price_predictor.ml.features.pipeline import FeaturePipeline
        from ticket_price_predictor.ml.training.splitter import TimeBasedSplitter

        print("Extracting features...")
        pipeline = FeaturePipeline(include_momentum=True)
        X = pipeline.fit_transform(df)
        y = df["listing_price"]

        print("Splitting data...")
        splitter = TimeBasedSplitter(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=test_ratio,
        )
        split = splitter.split(X, y, raw_df=df)

        metrics = trainer.train_with_params(split, params)
    else:
        metrics = trainer.train(
            df,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=test_ratio,
            preprocess=args.preprocess,
        )

    # Save model
    print()
    model_path = trainer.save(args.output)

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {model_path}")
    print()
    print("Summary:")
    print(f"  MAE:  ${metrics.mae:.2f}")
    print(f"  RMSE: ${metrics.rmse:.2f}")
    print(f"  R²:   {metrics.r2:.4f}")


if __name__ == "__main__":
    main()
