"""LightGBM model for price prediction."""

from pathlib import Path
from typing import Any, cast

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from ticket_price_predictor.ml.models.base import PriceModel


class LightGBMModel(PriceModel):
    """LightGBM gradient boosting model.

    Advanced model with native handling of missing values,
    categorical features, and built-in feature importance.

    Supported objectives:
        - "regression" (default): L2 loss, optimizes RMSE
        - "regression_l1": L1 loss, optimizes MAE
        - "huber": Huber loss, robust to outliers. Controlled by the
          "alpha" parameter which sets the transition point between
          L2 (near zero) and L1 (large errors). Because we train on
          log1p-transformed targets, alpha operates in LOG-SPACE:
          alpha=1.0 means errors up to ~1 log-unit (~2.7x price
          difference) use L2, larger errors use L1. Typical useful
          range: 0.5–2.0.
        - "quantile": Quantile regression (use QuantileLightGBMModel)
    """

    # Default params: DART + MSE. DART's dropout provides implicit regularization
    # but does NOT support early stopping — all n_estimators trees are built.
    # Huber loss is incompatible with DART (causes catastrophic overfitting);
    # use GBDT_PARAMS for Huber experiments.
    DEFAULT_PARAMS = {
        "objective": "regression",
        "metric": ["mae", "rmse"],
        "boosting_type": "dart",
        "num_leaves": 63,
        "learning_rate": 0.03,
        # DART-specific dropout regularization
        "drop_rate": 0.1,
        "skip_drop": 0.5,
        # Feature/data sampling
        "feature_fraction": 0.6,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "reg_alpha": 0.3,
        "reg_lambda": 0.3,
        "verbose": -1,
        "n_estimators": 2000,
        "early_stopping_rounds": 200,
    }

    # A/B test variants for controlled DART vs GBDT comparison (Phase 1a).
    # Use whichever produces lower test MAE. Both use Huber loss.
    GBDT_PARAMS = {
        "objective": "huber",
        "alpha": 0.5,
        "metric": ["mae", "rmse"],
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.5,
        "verbose": -1,
        "n_estimators": 3000,
        "early_stopping_rounds": 100,
    }

    DART_PARAMS = {
        "objective": "huber",
        "alpha": 0.5,
        "metric": ["mae", "rmse"],
        "boosting_type": "dart",
        "num_leaves": 63,
        "learning_rate": 0.03,
        "drop_rate": 0.1,
        "skip_drop": 0.5,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "reg_alpha": 0.3,
        "reg_lambda": 0.3,
        "verbose": -1,
        "n_estimators": 2000,
        "early_stopping_rounds": 200,
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
        sample_weight: np.ndarray | None = None,
    ) -> "LightGBMModel":
        """Fit the model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (used for early stopping)
            y_val: Validation target (used for early stopping)
            sample_weight: Optional per-sample weights for training

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
            weight=sample_weight,
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

        # Extract training params (use copy to avoid mutating self._params)
        train_params = {
            k: v
            for k, v in self._params.items()
            if k not in ("n_estimators", "early_stopping_rounds")
        }
        n_estimators = self._params.get("n_estimators", 500)
        early_stopping_rounds = self._params.get("early_stopping_rounds", 50)

        # Train
        callbacks: list[Any] = []
        if X_val is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=int(early_stopping_rounds)))
        callbacks.append(lgb.log_evaluation(period=100))

        self._model = lgb.train(
            train_params,
            train_data,
            num_boost_round=int(n_estimators),
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

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

        return self._model.predict(X, num_iteration=self._best_iteration)  # type: ignore[return-value]

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

        result = dict(zip(feature_names, importance, strict=False))

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
        # Force quantile objective — overrides any "huber" from DEFAULT_PARAMS.
        # alpha is also overridden per quantile in fit(), so DEFAULT_PARAMS alpha is irrelevant.
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
        sample_weight: np.ndarray | None = None,
    ) -> "QuantileLightGBMModel":
        """Fit three quantile models.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            sample_weight: Optional per-sample weights for training

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

            n_estimators = cast(int, params.pop("n_estimators", 500))
            early_stopping = cast(int, params.pop("early_stopping_rounds", 50))

            cat_features = [c for c in self._categorical_features if c in X_train.columns]

            train_data = lgb.Dataset(
                X_train,
                label=y_train,
                weight=sample_weight,
                categorical_feature=cat_features if cat_features else "auto",
            )

            valid_sets = [train_data]
            if X_val is not None and y_val is not None:
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                valid_sets.append(val_data)

            callbacks: list[Any] = []
            if X_val is not None:
                callbacks.append(lgb.early_stopping(stopping_rounds=int(early_stopping)))

            model = lgb.train(
                params,
                train_data,
                num_boost_round=int(n_estimators),
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

        return self._model_median.predict(X)  # type: ignore[return-value]

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

        assert self._model_median is not None
        assert self._model_lower is not None
        assert self._model_upper is not None
        median: np.ndarray = self._model_median.predict(X)  # type: ignore[assignment,type-arg]
        lower: np.ndarray = self._model_lower.predict(X)  # type: ignore[assignment,type-arg]
        upper: np.ndarray = self._model_upper.predict(X)  # type: ignore[assignment,type-arg]

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

        result = dict(zip(feature_names, importance, strict=False))
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
