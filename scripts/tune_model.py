#!/usr/bin/env python3
"""Hyperparameter tuning CLI for LightGBM models."""

import argparse
from pathlib import Path

from ticket_price_predictor.ml.training.data_loader import DataLoader
from ticket_price_predictor.ml.features.pipeline import FeaturePipeline
from ticket_price_predictor.ml.training.splitter import TimeBasedSplitter
from ticket_price_predictor.ml.tuning.study_manager import StudyManager


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for LightGBM ticket price prediction"
    )

    # Tuning configuration
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of trials to run (default: 50)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Maximum time in seconds (default: None)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Name for the study (default: auto-generated)",
    )
    parser.add_argument(
        "--search-space",
        type=str,
        default="aggressive",
        choices=["conservative", "aggressive", "regularization_focus"],
        help="Search space strategy (default: aggressive)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume existing study if found",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel trials (-1 for all cores, default: 1)",
    )
    parser.add_argument(
        "--no-penalty",
        action="store_true",
        help="Disable feature dominance penalty",
    )

    # Data configuration
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory (default: data)",
    )

    args = parser.parse_args()

    # Generate study name if not provided
    if args.study_name is None:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.study_name = f"lightgbm_{args.search_space}_{timestamp}"

    print("=" * 70)
    print("HYPERPARAMETER TUNING - LightGBM Ticket Price Predictor")
    print("=" * 70)
    print(f"\nStudy name: {args.study_name}")
    print(f"Search space: {args.search_space}")
    print(f"Trials: {args.n_trials}")
    print(f"Parallel jobs: {args.n_jobs}")
    print(f"Feature dominance penalty: {not args.no_penalty}")
    print(f"Resume: {args.resume}")
    print()

    # Load data
    print("Loading training data...")
    data_dir = Path(args.data_dir)
    loader = DataLoader(data_dir)
    df = loader.load_all_listings()

    if df.empty:
        print("ERROR: No data found. Run data collection first.")
        return

    print(f"  Loaded {len(df)} listings")

    # Extract features
    print("Extracting features...")
    pipeline = FeaturePipeline(include_momentum=True)
    X = pipeline.fit_transform(df)
    y = df["listing_price"]
    print(f"  Features: {X.shape[1]} columns")

    # Create time-based split
    print("\nCreating time-based train/val/test split...")
    splitter = TimeBasedSplitter()
    split = splitter.split(X, y, raw_df=df)
    print(f"  Train: {len(split.X_train)} samples")
    print(f"  Validation: {len(split.X_val)} samples")
    print(f"  Test: {len(split.X_test)} samples")

    # Create study manager and run optimization
    print(f"\nInitializing optimization study...")
    manager = StudyManager(study_name=args.study_name)

    study = manager.optimize(
        split=split,
        n_trials=args.n_trials,
        timeout=args.timeout,
        search_space=args.search_space,
        penalize_dominance=not args.no_penalty,
        n_jobs=args.n_jobs,
    )

    # Save trial metadata for all trials
    print("\nSaving trial metadata...")
    for trial in study.trials:
        manager.save_trial_metadata(trial)

    # Print summary
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)

    best_trial = study.best_trial
    print(f"\nBest Trial: #{best_trial.number}")
    print(f"  Validation MAE: ${best_trial.value:.2f}")
    print(f"  R²: {best_trial.user_attrs.get('val_r2', 'N/A'):.4f}")
    print(f"  Max feature importance: {best_trial.user_attrs.get('max_feature_importance', 'N/A'):.3f}")
    print(f"  Top feature: {best_trial.user_attrs.get('top_feature', 'N/A')}")
    print(f"  Best iteration: {best_trial.user_attrs.get('best_iteration', 'N/A')}")

    print("\nBest Hyperparameters:")
    for param, value in best_trial.params.items():
        print(f"  {param}: {value}")

    print(f"\nStudy saved to: {manager.storage}")
    print(f"Trial metadata saved to: {manager.trials_dir / args.study_name}")

    print("\nNext steps:")
    print(f"  1. Train final model:")
    print(f"     python scripts/train_model.py --from-study {args.study_name} --version v4")
    print(f"  2. View study in Optuna Dashboard (optional):")
    print(f"     optuna-dashboard {manager.storage}")


if __name__ == "__main__":
    main()
