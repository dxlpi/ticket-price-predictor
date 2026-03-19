#!/usr/bin/env python3
"""Hyperparameter tuning CLI for LightGBM models.

Default mode uses leak-free tuning: raw data is split before feature
extraction, and features are re-extracted per trial to enable smoothing
factor tuning. Evaluation is in dollar-space (not log-space).

Legacy mode (--legacy) uses the old pre-extracted feature path, which
fits the pipeline on all data before splitting (data leakage risk).
"""

import argparse
from pathlib import Path

import numpy as np

from ticket_price_predictor.ml.training.data_loader import DataLoader
from ticket_price_predictor.ml.training.splitter import TimeBasedSplitter
from ticket_price_predictor.ml.tuning.study_manager import StudyManager


def main() -> None:
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
        choices=["conservative", "aggressive", "regularization_focus", "full", "phase5"],
        help="Search space (default: aggressive). phase5 adds max_bin, path_smooth, min_gain_to_split",
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
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy pre-extracted feature path (has data leakage risk)",
    )
    parser.add_argument(
        "--cv-tuning",
        action="store_true",
        help="Use TemporalGroupCV within each trial for more stable hyperparameter selection",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Number of CV folds when --cv-tuning is active (default: 3)",
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
        from datetime import UTC, datetime

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        mode = args.search_space if args.legacy else "tuning"
        args.study_name = f"lightgbm_{mode}_{timestamp}"

    print("=" * 70)
    print("HYPERPARAMETER TUNING - LightGBM Ticket Price Predictor")
    print("=" * 70)
    print(f"\nStudy name: {args.study_name}")
    print(f"Mode: {'legacy (pre-extracted)' if args.legacy else 'leak-free (re-extract per trial)'}")
    if args.legacy:
        print(f"Search space: {args.search_space}")
    if not args.legacy and args.cv_tuning:
        print(f"CV tuning: enabled ({args.cv_folds} folds per trial)")
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

    # Filter invalid prices and cap outliers (same as ModelTrainer)
    df = df[df["listing_price"] >= 10].copy()
    cap = float(np.percentile(df["listing_price"], 95))
    df["listing_price"] = df["listing_price"].clip(upper=cap)
    print(f"  After filtering: {len(df)} listings (price cap: ${cap:.0f})")

    # Normalize city names
    if "city" in df.columns:
        from ticket_price_predictor.ml.features.geo_mapping import _normalize_city

        df["city"] = df["city"].apply(_normalize_city)

    # Normalize artist names
    if "artist_or_team" in df.columns:
        from ticket_price_predictor.ml.training.trainer import ModelTrainer

        df = ModelTrainer._normalize_artist_names(df)

    # Create study manager
    manager = StudyManager(study_name=args.study_name)

    if args.legacy:
        # Legacy path: pre-extract features (data leakage risk)
        from ticket_price_predictor.ml.features.pipeline import FeaturePipeline

        print("\nExtracting features (legacy: fit on all data)...")
        pipeline = FeaturePipeline(include_momentum=False)
        X = pipeline.fit_transform(df)
        y = df["listing_price"]
        print(f"  Features: {X.shape[1]} columns")

        print("\nCreating time-based train/val/test split...")
        splitter = TimeBasedSplitter()
        split = splitter.split(X, y, raw_df=df)
        print(f"  Train: {len(split.X_train)} samples")
        print(f"  Validation: {len(split.X_val)} samples")
        print(f"  Test: {len(split.X_test)} samples")

        study = manager.optimize(
            split=split,
            n_trials=args.n_trials,
            timeout=args.timeout,
            search_space=args.search_space,
            penalize_dominance=not args.no_penalty,
            n_jobs=args.n_jobs,
        )
    else:
        # Leak-free path: split raw data first, re-extract per trial
        print("\nSplitting raw data (before feature extraction)...")
        splitter = TimeBasedSplitter(stratify_col="artist_or_team")
        raw_split = splitter.split_raw(df)
        print(f"  Train: {raw_split.n_train} samples")
        print(f"  Validation: {raw_split.n_val} samples")
        print(f"  Test: {raw_split.n_test} samples")

        study = manager.optimize_raw(
            raw_split=raw_split,
            n_trials=args.n_trials,
            timeout=args.timeout,
            penalize_dominance=not args.no_penalty,
            n_jobs=args.n_jobs,
            pipeline_kwargs={"include_listing": False},
            use_cv=args.cv_tuning,
            n_cv_folds=args.cv_folds,
            search_space=args.search_space,
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
    print(f"  R²: {best_trial.user_attrs.get('val_r2', 'N/A')}")
    print(f"  Max feature importance: {best_trial.user_attrs.get('max_feature_importance', 'N/A')}")
    print(f"  Top feature: {best_trial.user_attrs.get('top_feature', 'N/A')}")
    print(f"  Best iteration: {best_trial.user_attrs.get('best_iteration', 'N/A')}")

    print("\nBest Hyperparameters:")
    for param, value in best_trial.params.items():
        print(f"  {param}: {value}")

    print(f"\nStudy saved to: {manager.storage}")
    print(f"Trial metadata saved to: {manager.trials_dir / args.study_name}")

    print("\nNext steps:")
    print(f"  1. Train final model:")
    print(f"     python scripts/train_model.py --from-study {args.study_name} --version vXX")
    print(f"  2. View study in Optuna Dashboard (optional):")
    print(f"     optuna-dashboard {manager.storage}")


if __name__ == "__main__":
    main()
