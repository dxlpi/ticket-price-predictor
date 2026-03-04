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
from typing import Any

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from ticket_price_predictor.ml.models.lightgbm_model import LightGBMModel
from ticket_price_predictor.ml.training.data_loader import DataLoader
from ticket_price_predictor.ml.training.trainer import ModelTrainer
from ticket_price_predictor.popularity.service import PopularityService


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

    parser.add_argument(
        "--cv",
        action="store_true",
        help="Use temporal cross-validation instead of single split",
    )

    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5, requires --cv)",
    )

    parser.add_argument(
        "--sample-weight",
        type=str,
        choices=["none", "inverse_artist_freq", "recency"],
        default="none",
        help="Sample weighting strategy (default: none)",
    )

    parser.add_argument(
        "--no-listing",
        action="store_true",
        help="Disable listing context features (recommended: reduces noise)",
    )

    parser.add_argument(
        "--section-feature",
        action="store_true",
        help="Enable experimental section-level pricing feature",
    )

    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Disable temporal snapshot features",
    )

    parser.add_argument(
        "--loss",
        type=str,
        choices=["l2", "huber"],
        default="l2",
        help="Loss function: l2 (default, DART+MSE) or huber (GBDT+Huber, robust to outliers)",
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

    # Select base params from loss flag (before Optuna study overrides)
    params = None
    if args.loss == "huber" and args.model == "lightgbm":
        params = dict(LightGBMModel.GBDT_PARAMS)
        print(f"Loss: huber (GBDT+Huber, alpha={params['alpha']})")
    else:
        if args.loss == "huber" and args.model != "lightgbm":
            print(f"WARNING: --loss huber only applies to lightgbm model, ignoring for {args.model}")
        print("Loss: l2 (default DART+MSE)")

    # Load hyperparameters from Optuna study if specified (overrides --loss params)
    if args.from_study:
        import optuna

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

        raw_params = dict(trial.params)

        # Separate pipeline smoothing factors from model hyperparameters
        pipeline_smoothing_keys = {
            "event_pricing_smoothing",
            "artist_stats_smoothing",
            "regional_smoothing",
        }
        smoothing = {k: raw_params.pop(k) for k in pipeline_smoothing_keys if k in raw_params}

        # Handle meta params (use_huber → objective/alpha, dart_n_estimators → n_estimators)
        use_huber = raw_params.pop("use_huber", False)
        if use_huber:
            raw_params["objective"] = "huber"
            huber_alpha = raw_params.pop("huber_alpha", 1.0)
            raw_params["alpha"] = huber_alpha
        else:
            raw_params.pop("huber_alpha", None)

        dart_n_est = raw_params.pop("dart_n_estimators", None)
        if dart_n_est is not None and raw_params.get("boosting_type") == "dart":
            raw_params["n_estimators"] = dart_n_est

        params = raw_params
        print(f"  Loaded {len(params)} model hyperparameters")
        if smoothing:
            print(f"  Smoothing factors: {smoothing}")
        print()

    print()

    # Load data
    print("Loading data...")
    loader = DataLoader(args.data_dir)
    df = loader.load_all_listings()
    snapshot_df = pd.DataFrame() if args.no_snapshot else loader.load_snapshots()

    if df.empty:
        print("ERROR: No data found. Run data collection first.")
        return

    summary = loader.get_summary()
    print(f"Loaded {summary['n_listings']:,} listings")
    print(f"  - {summary['n_events']} events")
    print(f"  - {summary['n_artists']} artists")
    print(f"  - Price range: ${summary['price_min']:.2f} - ${summary['price_max']:.2f}")
    print()

    # Initialize popularity service (YouTube Music + Last.fm)
    popularity_service = PopularityService()

    # Build pipeline_kwargs from CLI flags and Optuna smoothing factors
    study_pipeline_kwargs: dict[str, Any] | None = None
    if args.no_listing:
        study_pipeline_kwargs = {"include_listing": False}
    if args.section_feature:
        if study_pipeline_kwargs is None:
            study_pipeline_kwargs = {}
        ep_params = study_pipeline_kwargs.setdefault("extractor_params", {})
        ep_params.setdefault("EventPricingFeatureExtractor", {})["include_section_feature"] = True
    if args.no_snapshot:
        if study_pipeline_kwargs is None:
            study_pipeline_kwargs = {}
        study_pipeline_kwargs["include_snapshot"] = False
    if args.from_study and smoothing:
        extractor_params: dict[str, dict[str, Any]] = {}
        if "event_pricing_smoothing" in smoothing:
            extractor_params["EventPricingFeatureExtractor"] = {
                "SMOOTHING_FACTOR": smoothing["event_pricing_smoothing"],
            }
        if "artist_stats_smoothing" in smoothing:
            extractor_params["PerformerFeatureExtractor"] = {
                "artist_stats_smoothing": smoothing["artist_stats_smoothing"],
            }
        if "regional_smoothing" in smoothing:
            extractor_params["RegionalPopularityFeatureExtractor"] = {
                "regional_smoothing": smoothing["regional_smoothing"],
            }
        study_pipeline_kwargs = {
            "extractor_params": extractor_params,
            "include_listing": False,
        }

    # Train model
    trainer = ModelTrainer(
        model_type=args.model,  # type: ignore
        model_version=args.version,
    )

    test_ratio = 1.0 - args.train_ratio - args.val_ratio

    if args.cv:
        from ticket_price_predictor.ml.training.splitter import TemporalGroupCV

        print(f"\nUsing temporal cross-validation with {args.n_folds} folds")
        cv = TemporalGroupCV(n_folds=args.n_folds)
        folds = cv.split(df)
        print(f"Generated {len(folds)} folds")

        fold_metrics = []
        for i, fold in enumerate(folds):
            print(f"\n{'=' * 40}")
            print(f"FOLD {i + 1}/{len(folds)}")
            print(f"{'=' * 40}")
            fold_trainer = ModelTrainer(
                model_type=args.model,  # type: ignore
                model_version=f"{args.version}_fold{i + 1}",
            )
            m = fold_trainer.train(
                fold.train_df,
                train_ratio=0.85,
                val_ratio=0.15,
                test_ratio=0.0,
                params=params,
                popularity_service=popularity_service,
                pipeline_kwargs=study_pipeline_kwargs,
                snapshot_df=snapshot_df,
            )
            fold_metrics.append(m)

        # Summary
        import numpy as np

        maes = [m.mae for m in fold_metrics]
        print(f"\n{'=' * 60}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Mean MAE:  ${np.mean(maes):.2f} (+/- ${np.std(maes):.2f})")
        print(f"  Best MAE:  ${np.min(maes):.2f}")
        print(f"  Worst MAE: ${np.max(maes):.2f}")

        # Use last fold's trainer for saving
        trainer = fold_trainer  # noqa: F841
        metrics = fold_metrics[-1]
    else:
        metrics = trainer.train(
            df,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=test_ratio,
            preprocess=args.preprocess,
            params=params,
            popularity_service=popularity_service,
            pipeline_kwargs=study_pipeline_kwargs,
            snapshot_df=snapshot_df,
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
