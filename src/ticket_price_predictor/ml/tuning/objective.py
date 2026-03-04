"""Optuna objective function for LightGBM tuning."""

from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ticket_price_predictor.ml.models.lightgbm_model import LightGBMModel
from ticket_price_predictor.ml.training.splitter import DataSplit, RawDataSplit, TemporalGroupCV
from ticket_price_predictor.ml.training.trainer import _LOG_EXCLUDE_SUFFIXES


def create_objective(
    split: DataSplit,
    search_space: str = "aggressive",
    penalize_dominance: bool = True,
) -> Any:
    """Factory to create objective function with data closure.

    Args:
        split: Pre-computed train/val/test split
        search_space: "conservative", "aggressive", or "regularization_focus"
        penalize_dominance: Add penalty for high feature dominance

    Returns:
        Objective function for Optuna
    """

    def objective(trial: optuna.Trial) -> float:
        """Objective: minimize validation MAE (with optional penalty)."""

        # Sample hyperparameters based on search space
        if search_space == "conservative":
            params = _sample_conservative(trial)
        elif search_space == "regularization_focus":
            params = _sample_regularization_focus(trial)
        elif search_space == "full":
            params = _sample_full(trial)
        else:  # aggressive
            params = _sample_aggressive(trial)

        # Train model
        try:
            model = LightGBMModel(params=params)
            model.fit(split.X_train, split.y_train, split.X_val, split.y_val)

            # Validation predictions
            val_preds = model.predict(split.X_val)
            val_mae = mean_absolute_error(split.y_val, val_preds)

            # Train predictions (for overfitting check)
            train_preds = model.predict(split.X_train)
            train_mae = mean_absolute_error(split.y_train, train_preds)

            # Secondary metrics
            val_r2 = r2_score(split.y_val, val_preds)
            val_rmse = np.sqrt(mean_squared_error(split.y_val, val_preds))

            # Feature importance
            importance = model.get_feature_importance()
            max_importance = max(importance.values()) if importance else 0.0
            top_feature = list(importance.keys())[0] if importance else ""

            # Store as user attributes
            trial.set_user_attr("val_r2", val_r2)
            trial.set_user_attr("val_rmse", val_rmse)
            trial.set_user_attr("train_mae", train_mae)
            trial.set_user_attr("best_iteration", model.best_iteration)
            trial.set_user_attr("max_feature_importance", max_importance)
            trial.set_user_attr("top_feature", top_feature)

            # Objective value (with optional penalty)
            objective_value = val_mae

            if penalize_dominance:
                # Penalize if any feature exceeds 50% importance
                dominance_penalty = max(0, max_importance - 0.5) * 100
                objective_value += dominance_penalty
                trial.set_user_attr("dominance_penalty", dominance_penalty)

            return float(objective_value)

        except (ValueError, RuntimeError, KeyError) as e:
            trial.set_user_attr("error", str(e))
            return float("inf")

    return objective


def _sample_conservative(trial: optuna.Trial) -> dict[str, Any]:
    """Conservative search space around current defaults."""
    return {
        "objective": "regression",
        "metric": ["rmse", "mae"],
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_categorical("num_leaves", [23, 31, 47]),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 7, 10]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000, step=100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.8),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.8),
        "bagging_freq": 5,
        "min_child_samples": trial.suggest_categorical("min_child_samples", [10, 20, 50]),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "early_stopping_rounds": 50,
        "verbose": -1,
    }


def _sample_aggressive(trial: optuna.Trial) -> dict[str, Any]:
    """Aggressive search space for exploration."""
    return {
        "objective": "regression",
        "metric": ["rmse", "mae"],
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_categorical("num_leaves", [15, 31, 63, 127, 255]),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 5, 7, 10, 15]),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000, step=100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 0.9),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.9),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1, 5, 10]),
        "min_child_samples": trial.suggest_categorical("min_child_samples", [5, 10, 20, 50, 100]),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "early_stopping_rounds": 50,
        "verbose": -1,
    }


def _sample_regularization_focus(trial: optuna.Trial) -> dict[str, Any]:
    """Search space focused on combating feature dominance."""
    return {
        "objective": "regression",
        "metric": ["rmse", "mae"],
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_categorical("num_leaves", [31, 63]),
        "max_depth": trial.suggest_categorical("max_depth", [7, 10]),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 500, 1500, step=100),
        # Heavy regularization
        "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 0.6),  # Lower!
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.7),
        "bagging_freq": 5,
        "min_child_samples": trial.suggest_categorical("min_child_samples", [50, 100, 200]),
        "reg_alpha": trial.suggest_float("reg_alpha", 1.0, 20.0),  # Higher!
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 20.0),  # Higher!
        "early_stopping_rounds": 50,
        "verbose": -1,
    }


def _sample_full(trial: optuna.Trial) -> dict[str, Any]:
    """Full search space with DART/GBDT selection and Huber loss option.

    Explores boosting type (GBDT vs DART), loss function (L2 vs Huber),
    and conditional parameters that depend on boosting type.

    When boosting_type="dart", n_estimators is fixed at 2000 because DART
    uses all iterations (no early stopping at a single best iteration).
    """
    boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "dart"])

    # DART uses all iterations; fix n_estimators rather than sampling
    if boosting_type == "dart":
        n_estimators = 2000
    else:
        n_estimators = trial.suggest_int("n_estimators", 500, 3000, step=100)

    params: dict[str, Any] = {
        "objective": "regression",
        "metric": ["rmse", "mae"],
        "boosting_type": boosting_type,
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 5, 7, 10, 15]),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "n_estimators": n_estimators,
        "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 0.9),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.9),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1, 5, 10]),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 20.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 20.0, log=True),
        "max_bin": trial.suggest_categorical("max_bin", [63, 127, 255, 511]),
        "path_smooth": trial.suggest_float("path_smooth", 0.0, 10.0),
        "early_stopping_rounds": 200,
        "verbose": -1,
    }

    # DART-specific conditional parameters
    if boosting_type == "dart":
        params["drop_rate"] = trial.suggest_float("drop_rate", 0.05, 0.3)
        params["skip_drop"] = trial.suggest_float("skip_drop", 0.3, 0.7)
        params["max_drop"] = trial.suggest_int("max_drop", 10, 50)

    # Huber loss option — ONLY with GBDT (DART+Huber causes catastrophic overfitting)
    if boosting_type == "gbdt":
        use_huber = trial.suggest_categorical("use_huber", [True, False])
        if use_huber:
            params["objective"] = "huber"
            params["alpha"] = trial.suggest_float("huber_alpha", 0.5, 2.0)

    return params


def create_raw_objective(
    raw_split: RawDataSplit,
    target_col: str = "listing_price",
    pipeline_kwargs: dict[str, Any] | None = None,
    penalize_dominance: bool = True,
    use_cv: bool = False,
    n_cv_folds: int = 3,
) -> Any:
    """Factory to create leak-free objective with dollar-space evaluation.

    Unlike create_objective which takes pre-extracted features (potential leakage),
    this function takes raw data splits and re-extracts features per trial.
    This enables tuning smoothing factors and other pipeline parameters.

    Evaluates MAE in dollar-space (np.expm1) since the acceptance criteria
    are in dollars, and log-space MAE is not monotonically equivalent.

    Args:
        raw_split: Pre-computed raw data split (before feature extraction)
        target_col: Target column name
        pipeline_kwargs: Base kwargs for FeaturePipeline (e.g. include_listing=False)
        penalize_dominance: Add penalty for high feature dominance
        use_cv: When True, use TemporalGroupCV on raw_split.train_df for more stable
            hyperparameter selection. Each fold evaluates on fold.val_df (matching the
            single-split pattern). Falls back to single-split when <2 folds produced.
        n_cv_folds: Number of CV folds when use_cv=True (default 3)

    Returns:
        Objective function for Optuna
    """
    from ticket_price_predictor.ml.features.pipeline import FeaturePipeline

    base_kwargs: dict[str, Any] = {"include_momentum": False}
    if pipeline_kwargs:
        base_kwargs.update(pipeline_kwargs)

    def _evaluate_single_split(
        params: dict[str, Any],
        extractor_params: dict[str, dict[str, Any]],
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> tuple[float, float, float, float, int, float, str]:
        """Extract features, train, and evaluate on one split. Returns (mae, r2, rmse, train_mae, n_features, max_importance, top_feature)."""
        kwargs = {**base_kwargs, "extractor_params": extractor_params}
        pipeline = FeaturePipeline(**kwargs)
        pipeline.fit(train_df)

        X_train = pipeline.transform(train_df).copy()
        X_val = pipeline.transform(val_df).copy()

        zero_var_cols = [c for c in X_train.columns if X_train[c].nunique() <= 1]
        if zero_var_cols:
            X_train = X_train.drop(columns=zero_var_cols)
            X_val = X_val.drop(columns=zero_var_cols)

        price_cols = [
            c
            for c in X_train.columns
            if ("price" in c.lower() or "avg" in c.lower() or "median" in c.lower())
            and not any(c.lower().endswith(suffix) for suffix in _LOG_EXCLUDE_SUFFIXES)
        ]
        for c in price_cols:
            X_train[c] = np.log1p(X_train[c].clip(lower=0))
            X_val[c] = np.log1p(X_val[c].clip(lower=0))

        y_train = pd.Series(np.log1p(train_df[target_col].values), name=target_col)
        y_val_raw = val_df[target_col].values
        y_val_log = pd.Series(np.log1p(y_val_raw), name=target_col)

        model = LightGBMModel(params=params)
        model.fit(X_train, y_train, X_val, y_val_log)

        val_preds_dollars = np.expm1(model.predict(X_val))
        val_mae = float(mean_absolute_error(y_val_raw, val_preds_dollars))
        val_r2 = float(r2_score(y_val_raw, val_preds_dollars))
        val_rmse = float(np.sqrt(mean_squared_error(y_val_raw, val_preds_dollars)))

        y_train_raw = train_df[target_col].values
        train_mae = float(mean_absolute_error(y_train_raw, np.expm1(model.predict(X_train))))

        importance = model.get_feature_importance()
        max_importance = max(importance.values()) if importance else 0.0
        top_feature = list(importance.keys())[0] if importance else ""

        return val_mae, val_r2, val_rmse, train_mae, X_train.shape[1], max_importance, top_feature

    def objective(trial: optuna.Trial) -> float:
        """Objective: minimize dollar-space validation MAE."""

        # Sample model hyperparameters
        params = _sample_tuning(trial)

        # Sample smoothing factors for feature extractors
        extractor_params: dict[str, dict[str, Any]] = {}

        event_smoothing = trial.suggest_int("event_pricing_smoothing", 5, 50)
        artist_smoothing = trial.suggest_int("artist_stats_smoothing", 20, 100)
        regional_smoothing = trial.suggest_int("regional_smoothing", 30, 150)

        extractor_params["EventPricingFeatureExtractor"] = {
            "SMOOTHING_FACTOR": event_smoothing,
        }
        extractor_params["PerformerFeatureExtractor"] = {
            "artist_stats_smoothing": artist_smoothing,
        }
        extractor_params["RegionalPopularityFeatureExtractor"] = {
            "regional_smoothing": regional_smoothing,
        }

        try:
            if use_cv:
                cv = TemporalGroupCV(n_folds=n_cv_folds)
                folds = cv.split(raw_split.train_df)

                if len(folds) < 2:
                    # Not enough data for CV; fall through to single-split
                    fold_maes = None
                else:
                    fold_results = []
                    for fold in folds:
                        result = _evaluate_single_split(
                            params, extractor_params, fold.train_df, fold.val_df
                        )
                        fold_results.append(result)

                    fold_maes_list = [r[0] for r in fold_results]
                    val_mae = float(np.mean(fold_maes_list))
                    val_r2 = float(np.mean([r[1] for r in fold_results]))
                    val_rmse = float(np.mean([r[2] for r in fold_results]))
                    train_mae = float(np.mean([r[3] for r in fold_results]))
                    n_features = fold_results[-1][4]
                    max_importance = float(np.mean([r[5] for r in fold_results]))
                    top_feature = fold_results[-1][6]
                    fold_maes = fold_maes_list

                if fold_maes is None:
                    (
                        val_mae,
                        val_r2,
                        val_rmse,
                        train_mae,
                        n_features,
                        max_importance,
                        top_feature,
                    ) = _evaluate_single_split(
                        params, extractor_params, raw_split.train_df, raw_split.val_df
                    )
            else:
                val_mae, val_r2, val_rmse, train_mae, n_features, max_importance, top_feature = (
                    _evaluate_single_split(
                        params, extractor_params, raw_split.train_df, raw_split.val_df
                    )
                )

            # Store as user attributes
            trial.set_user_attr("val_mae_dollars", val_mae)
            trial.set_user_attr("val_r2", val_r2)
            trial.set_user_attr("val_rmse", val_rmse)
            trial.set_user_attr("train_mae", train_mae)
            trial.set_user_attr("max_feature_importance", max_importance)
            trial.set_user_attr("top_feature", top_feature)
            trial.set_user_attr("n_features", n_features)

            # Objective value
            objective_value = val_mae

            if penalize_dominance:
                dominance_penalty = max(0, max_importance - 0.5) * val_mae * 0.1
                objective_value += dominance_penalty
                trial.set_user_attr("dominance_penalty", dominance_penalty)

            return objective_value

        except (ValueError, RuntimeError, KeyError) as e:
            trial.set_user_attr("error", str(e))
            return float("inf")

    return objective


def _sample_tuning(trial: optuna.Trial) -> dict[str, Any]:
    """Search space for Phase 4 tuning with DART/GBDT and safety guards.

    Uses MAE-first metric ordering for MAE-based early stopping.
    Guards against DART+Huber (catastrophic overfitting).
    """
    boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "dart"])

    if boosting_type == "dart":
        n_estimators = trial.suggest_categorical("dart_n_estimators", [1500, 2000, 2500])
    else:
        n_estimators = trial.suggest_int("n_estimators", 500, 3000, step=100)

    params: dict[str, Any] = {
        "objective": "regression",
        "metric": ["mae", "rmse"],
        "boosting_type": boosting_type,
        "num_leaves": trial.suggest_int("num_leaves", 31, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "n_estimators": n_estimators,
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 0.8),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.9),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1, 5, 10]),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 5.0, log=True),
        "early_stopping_rounds": 200,
        "verbose": -1,
    }

    if boosting_type == "dart":
        params["drop_rate"] = trial.suggest_float("drop_rate", 0.05, 0.2)
        params["skip_drop"] = trial.suggest_float("skip_drop", 0.3, 0.7)

    # Huber only with GBDT (DART+Huber causes catastrophic overfitting)
    if boosting_type == "gbdt":
        use_huber = trial.suggest_categorical("use_huber", [True, False])
        if use_huber:
            params["objective"] = "huber"
            params["alpha"] = trial.suggest_float("huber_alpha", 0.5, 2.0)

    return params
