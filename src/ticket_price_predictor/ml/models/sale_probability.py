"""LightGBM binary classifier for sale probability prediction.

Predicts the probability that a ticket listing will sell within a time window.
Used as a CVR (conversion rate) analogue for commerce-style ranking:
listings with higher sale probability score higher in value-based ranking.

The model extends PriceModel ABC for consistent save/load/fit/predict contract,
with the following semantic differences from price regression:
  - predict() returns probabilities in [0, 1], not dollar values
  - predict_with_uncertainty() raises NotImplementedError (probability CIs
    are not calibrated confidence intervals in the price-bound sense)
  - fit() accepts optional sample_weight (following LightGBMModel precedent,
    which extends the ABC signature)
"""

from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy.typing as npt
import pandas as pd

from ticket_price_predictor.ml.models.base import PriceModel


class SaleProbabilityModel(PriceModel):
    """LightGBM binary classifier for listing sale probability.

    Predicts P(listing sells within window) from listing features.
    Structurally equivalent to a CVR prediction model in commerce:
    - Input: listing features (price, zone, artist, venue, time-to-event, etc.)
    - Output: probability score in [0, 1]

    Note: predict() returns probabilities, not dollar amounts. This deviates
    from the PriceModel ABC's regression-oriented docstrings. Use predict_class()
    for binary decisions.

    Class imbalance: sold listings are typically a minority class. is_unbalance=True
    in DEFAULT_PARAMS reweights by class frequency automatically.
    """

    DEFAULT_PARAMS: dict[str, Any] = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.01,
        "feature_fraction": 0.5,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "min_child_samples": 30,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "is_unbalance": True,
        "verbose": -1,
        "n_estimators": 8000,
        "early_stopping_rounds": 500,
        "max_bin": 127,
    }

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        categorical_features: list[str] | None = None,
    ) -> None:
        """Initialize sale probability model.

        Args:
            params: LightGBM parameters (merged with defaults)
            categorical_features: List of categorical feature names
        """
        self._params: dict[str, Any] = {**self.DEFAULT_PARAMS, **(params or {})}
        self._categorical_features: list[str] = categorical_features or []
        self._model: lgb.Booster | None = None
        self._fitted = False
        self._feature_names: list[str] = []
        self._best_iteration: int | None = None

    @property
    def name(self) -> str:
        """Return model name."""
        return "sale_probability"

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
    ) -> "SaleProbabilityModel":
        """Fit the binary classifier.

        Args:
            X_train: Training features
            y_train: Binary labels (1=sold, 0=not sold)
            X_val: Validation features (for early stopping)
            y_val: Validation labels
            sample_weight: Optional per-sample weights

        Returns:
            self
        """
        self._feature_names = list(X_train.columns)
        cat_features = [c for c in self._categorical_features if c in X_train.columns]

        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            weight=sample_weight,
            categorical_feature=cat_features if cat_features else "auto",
        )

        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("valid")

        train_params = {
            k: v
            for k, v in self._params.items()
            if k not in ("n_estimators", "early_stopping_rounds")
        }
        n_estimators = self._params.get("n_estimators", 500)
        early_stopping_rounds = self._params.get("early_stopping_rounds", 50)

        callbacks: list[Any] = []
        if X_val is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=int(early_stopping_rounds), first_metric_only=True))
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

    def predict(self, X: pd.DataFrame) -> npt.NDArray[Any]:
        """Return sale probability scores in [0, 1].

        Note: Returns probabilities, not dollar values. This method fulfills
        the PriceModel ABC contract but outputs classification probabilities.
        Use predict_class() for binary (sold/not-sold) decisions.

        Args:
            X: Feature DataFrame

        Returns:
            Array of probabilities in [0, 1]
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("Model must be fitted before predicting")
        return self._model.predict(X, num_iteration=self._best_iteration)  # type: ignore[return-value]

    def predict_with_uncertainty(
        self, X: pd.DataFrame
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
        """Not applicable for binary classification.

        Raises:
            NotImplementedError: Probability confidence intervals are not
                computed. Use predict() for sale probability scores.
        """
        raise NotImplementedError(
            "predict_with_uncertainty() is not defined for binary classification. "
            "Use predict() for sale probability scores in [0, 1]."
        )

    def predict_class(self, X: pd.DataFrame, threshold: float = 0.5) -> npt.NDArray[Any]:
        """Return binary predictions (sold=1, not-sold=0).

        Args:
            X: Feature DataFrame
            threshold: Probability threshold for positive class (default 0.5)

        Returns:
            Binary array (0 or 1)
        """
        probs = self.predict(X)
        return (probs >= threshold).astype(int)

    def get_feature_importance(self, top_k: int | None = None) -> dict[str, float]:
        """Get feature importance scores (by gain).

        Args:
            top_k: If set, return only top K features. None returns all.

        Returns:
            Dictionary mapping feature names to normalized importance scores
        """
        if not self._fitted or self._model is None:
            return {}

        importance = self._model.feature_importance(importance_type="gain")
        feature_names = self._model.feature_name()

        total = importance.sum()
        if total > 0:
            importance = importance / total

        result = dict(
            sorted(
                zip(feature_names, importance, strict=False),
                key=lambda x: -x[1],
            )
        )

        if top_k is not None:
            return dict(list(result.items())[:top_k])
        return result

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
    def load(cls, path: Path) -> "SaleProbabilityModel":
        """Load model from disk.

        Args:
            path: Path to load model from

        Returns:
            Loaded SaleProbabilityModel instance
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
