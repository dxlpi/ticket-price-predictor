"""LightGBM model for price prediction."""

from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from ticket_price_predictor.ml.models.base import PriceModel


class LightGBMModel(PriceModel):
    """LightGBM gradient boosting model.

    Advanced model with native handling of missing values,
    categorical features, and built-in feature importance.
    """

    DEFAULT_PARAMS = {
        "objective": "regression",
        "metric": ["rmse", "mae"],
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_estimators": 500,
        "early_stopping_rounds": 50,
    }

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        categorical_features: list[str] | None = None,
    ) -> None:
        """Initialize LightGBM model.

        Args:
            params: LightGBM parameters (merged with defaults)
            categorical_features: List of categorical feature names
        """
        self._params = {**self.DEFAULT_PARAMS, **(params or {})}
        self._categorical_features = categorical_features or []
        self._model: lgb.Booster | None = None
        self._fitted = False
        self._feature_names: list[str] = []
        self._best_iteration: int | None = None

    @property
    def name(self) -> str:
        """Return model name."""
        return "lightgbm"

    @property
    def is_fitted(self) -> bool:
        """Return True if model has been fitted."""
        return self._fitted

    @property
    def best_iteration(self) -> int | None:
        """Return best iteration from early stopping."""
        return self._best_iteration

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "LightGBMModel":
        """Fit the model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (used for early stopping)
            y_val: Validation target (used for early stopping)

        Returns:
            self
        """
        self._feature_names = list(X_train.columns)

        # Identify categorical features that exist in data
        cat_features = [c for c in self._categorical_features if c in X_train.columns]

        # Create datasets
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=cat_features if cat_features else "auto",
        )

        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                reference=train_data,
            )
            valid_sets.append(val_data)
            valid_names.append("valid")

        # Extract training params
        n_estimators = self._params.pop("n_estimators", 500)
        early_stopping_rounds = self._params.pop("early_stopping_rounds", 50)

        # Train
        callbacks = []
        if X_val is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))
        callbacks.append(lgb.log_evaluation(period=100))

        self._model = lgb.train(
            self._params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        # Restore params
        self._params["n_estimators"] = n_estimators
        self._params["early_stopping_rounds"] = early_stopping_rounds

        self._best_iteration = self._model.best_iteration
        self._fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.

        Args:
            X: Feature DataFrame

        Returns:
            Array of predictions
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("Model must be fitted before predicting")

        return self._model.predict(X, num_iteration=self._best_iteration)

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self._fitted or self._model is None:
            return {}

        importance = self._model.feature_importance(importance_type="gain")
        feature_names = self._model.feature_name()

        # Normalize
        total = importance.sum()
        if total > 0:
            importance = importance / total

        result = dict(zip(feature_names, importance))

        # Sort by importance
        return dict(sorted(result.items(), key=lambda x: -x[1])[:20])

    def save(self, path: Path) -> None:
        """Save model to disk.

        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "model": self._model,
                "params": self._params,
                "categorical_features": self._categorical_features,
                "feature_names": self._feature_names,
                "best_iteration": self._best_iteration,
                "fitted": self._fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "LightGBMModel":
        """Load model from disk.

        Args:
            path: Path to load model from

        Returns:
            Loaded model
        """
        data = joblib.load(path)

        model = cls(
            params=data["params"],
            categorical_features=data["categorical_features"],
        )
        model._model = data["model"]
        model._feature_names = data["feature_names"]
        model._best_iteration = data["best_iteration"]
        model._fitted = data["fitted"]

        return model

    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        return {
            "params": self._params,
            "categorical_features": self._categorical_features,
        }


class QuantileLightGBMModel(PriceModel):
    """LightGBM model with quantile regression for uncertainty.

    Trains three models for lower bound (2.5%), median (50%),
    and upper bound (97.5%) to provide 95% confidence intervals.
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        categorical_features: list[str] | None = None,
    ) -> None:
        """Initialize quantile model.

        Args:
            params: Base LightGBM parameters
            categorical_features: List of categorical feature names
        """
        base_params = LightGBMModel.DEFAULT_PARAMS.copy()
        base_params.update(params or {})
        base_params["objective"] = "quantile"

        self._base_params = base_params
        self._categorical_features = categorical_features or []

        self._model_lower: lgb.Booster | None = None
        self._model_median: lgb.Booster | None = None
        self._model_upper: lgb.Booster | None = None

        self._fitted = False
        self._feature_names: list[str] = []

    @property
    def name(self) -> str:
        """Return model name."""
        return "quantile_lightgbm"

    @property
    def is_fitted(self) -> bool:
        """Return True if model has been fitted."""
        return self._fitted

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "QuantileLightGBMModel":
        """Fit three quantile models.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target

        Returns:
            self
        """
        self._feature_names = list(X_train.columns)

        for quantile, attr in [
            (0.025, "_model_lower"),
            (0.5, "_model_median"),
            (0.975, "_model_upper"),
        ]:
            params = self._base_params.copy()
            params["alpha"] = quantile

            n_estimators = params.pop("n_estimators", 500)
            early_stopping = params.pop("early_stopping_rounds", 50)

            cat_features = [
                c for c in self._categorical_features if c in X_train.columns
            ]

            train_data = lgb.Dataset(
                X_train,
                label=y_train,
                categorical_feature=cat_features if cat_features else "auto",
            )

            valid_sets = [train_data]
            if X_val is not None and y_val is not None:
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                valid_sets.append(val_data)

            callbacks = []
            if X_val is not None:
                callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping))

            model = lgb.train(
                params,
                train_data,
                num_boost_round=n_estimators,
                valid_sets=valid_sets,
                callbacks=callbacks,
            )

            setattr(self, attr, model)

        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make median predictions.

        Args:
            X: Feature DataFrame

        Returns:
            Array of median predictions
        """
        if not self._fitted or self._model_median is None:
            raise RuntimeError("Model must be fitted before predicting")

        return self._model_median.predict(X)

    def predict_with_uncertainty(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make predictions with 95% confidence intervals.

        Args:
            X: Feature DataFrame

        Returns:
            Tuple of (median, lower_bound, upper_bound)
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predicting")

        median = self._model_median.predict(X)  # type: ignore
        lower = self._model_lower.predict(X)  # type: ignore
        upper = self._model_upper.predict(X)  # type: ignore

        return median, lower, upper

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance from median model."""
        if not self._fitted or self._model_median is None:
            return {}

        importance = self._model_median.feature_importance(importance_type="gain")
        feature_names = self._model_median.feature_name()

        total = importance.sum()
        if total > 0:
            importance = importance / total

        result = dict(zip(feature_names, importance))
        return dict(sorted(result.items(), key=lambda x: -x[1])[:20])

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "model_lower": self._model_lower,
                "model_median": self._model_median,
                "model_upper": self._model_upper,
                "params": self._base_params,
                "categorical_features": self._categorical_features,
                "feature_names": self._feature_names,
                "fitted": self._fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "QuantileLightGBMModel":
        """Load model from disk."""
        data = joblib.load(path)

        model = cls(
            params=data["params"],
            categorical_features=data["categorical_features"],
        )
        model._model_lower = data["model_lower"]
        model._model_median = data["model_median"]
        model._model_upper = data["model_upper"]
        model._feature_names = data["feature_names"]
        model._fitted = data["fitted"]

        return model
