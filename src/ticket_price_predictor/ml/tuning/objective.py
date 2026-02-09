"""Optuna objective function for LightGBM tuning."""

from typing import Any

import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ticket_price_predictor.ml.models.lightgbm_model import LightGBMModel
from ticket_price_predictor.ml.training.splitter import DataSplit


def create_objective(
    split: DataSplit,
    search_space: str = "aggressive",
    penalize_dominance: bool = True,
):
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

            return objective_value

        except Exception as e:
            # Log error and return worst possible value
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
        "min_child_samples": trial.suggest_categorical(
            "min_child_samples", [5, 10, 20, 50, 100]
        ),
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
