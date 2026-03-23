"""CatBoost model for price prediction."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd

try:
    from catboost import CatBoostRegressor  # type: ignore[import-not-found,unused-ignore]
except ImportError:
    CatBoostRegressor = None  # type: ignore[assignment,misc,unused-ignore]

from ticket_price_predictor.ml.models.base import PriceModel


class CatBoostModel(PriceModel):
    """CatBoost gradient boosting model.

    Provides model diversity through ordered boosting, which reduces
    target leakage in gradient estimation. Native categorical feature
    handling via ordered target statistics.
    """

    DEFAULT_PARAMS: dict[str, Any] = {
        "loss_function": "Huber:delta=0.5",
        "eval_metric": "MAE",
        "iterations": 3000,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "subsample": 0.8,
        "colsample_bylevel": 0.7,
        "min_data_in_leaf": 20,
        "early_stopping_rounds": 100,
        "verbose": 0,
        "random_seed": 42,
        "allow_writing_files": False,
    }

    def __init__(
        self,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Initialize CatBoost model.

        Args:
            params: CatBoost parameters (merged with defaults)
        """
        if CatBoostRegressor is None:
            raise ImportError("catboost is required: pip install catboost")
        self._params = {**self.DEFAULT_PARAMS, **(params or {})}
        self._model: CatBoostRegressor | None = None
        self._fitted = False
        self._feature_names: list[str] = []
        self._best_iteration: int | None = None

    @property
    def name(self) -> str:
        """Return model name."""
        return "catboost"

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
        sample_weight: npt.NDArray[Any] | None = None,
    ) -> "CatBoostModel":
        """Fit the model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (used for early stopping)
            y_val: Validation target (used for early stopping)
            sample_weight: Optional per-sample weights

        Returns:
            self
        """
        self._feature_names = list(X_train.columns)

        self._model = CatBoostRegressor(**self._params)

        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = (X_val, y_val)

        self._model.fit(X_train, y_train, **fit_kwargs)

        self._best_iteration = int(self._model.get_best_iteration()) if X_val is not None else None
        self._fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> npt.NDArray[Any]:
        """Make predictions.

        Args:
            X: Feature DataFrame

        Returns:
            Array of predictions
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("Model must be fitted before predicting")

        return np.asarray(self._model.predict(X))

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self._fitted or self._model is None:
            return {}

        importance = np.asarray(self._model.get_feature_importance())
        feature_names = self._feature_names

        # Normalize
        total = importance.sum()
        if total > 0:
            importance = importance / total

        result = dict(zip(feature_names, importance, strict=False))
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
                "feature_names": self._feature_names,
                "best_iteration": self._best_iteration,
                "fitted": self._fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "CatBoostModel":
        """Load model from disk.

        Args:
            path: Path to load model from

        Returns:
            Loaded model
        """
        data = joblib.load(path)

        model = cls(params=data["params"])
        model._model = data["model"]
        model._feature_names = data["feature_names"]
        model._best_iteration = data["best_iteration"]
        model._fitted = data["fitted"]

        return model

    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        return {"params": self._params}
