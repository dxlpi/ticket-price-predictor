"""Two-stage residual prediction model.

Stage 1 (Coarse Estimator): Predicts price from event-level pricing features.
Stage 2 (Residual Refiner): Predicts the residual using all other features.

Final prediction = coarse_prediction + residual_prediction.

This decomposition frees Stage 2 from being dominated by the top-3 event
pricing features (which account for 88.4% of importance in the single model).
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd

from ticket_price_predictor.ml.models.base import PriceModel
from ticket_price_predictor.ml.models.lightgbm_model import LightGBMModel

# Features used by the coarse estimator (event-level pricing signals)
COARSE_FEATURES = [
    "event_section_median_price",
    "event_zone_median_price",
    "event_median_price",
    "event_zone_price_ratio",
    "event_section_price_ratio",
]


class ResidualModel(PriceModel):
    """Two-stage residual prediction model.

    Stage 1 predicts a coarse price from event-level features.
    Stage 2 predicts the residual (y_true - coarse_pred) using
    the remaining features, allowing it to focus on listing-level
    and artist-level signals without event-pricing dominance.

    Both stages operate in log-space (targets are log1p-transformed).
    """

    def __init__(
        self,
        coarse_features: list[str] | None = None,
        coarse_params: dict[str, Any] | None = None,
        refiner_params: dict[str, Any] | None = None,
    ) -> None:
        """Initialize residual model.

        Args:
            coarse_features: Feature names for coarse estimator.
                Defaults to COARSE_FEATURES.
            coarse_params: LightGBM params for coarse model.
                Defaults to simpler configuration (fewer leaves).
            refiner_params: LightGBM params for residual refiner.
                Defaults to deeper model with more regularization.
        """
        self._coarse_feature_names = coarse_features or list(COARSE_FEATURES)

        self._coarse_params: dict[str, Any] = {
            **LightGBMModel.DEFAULT_PARAMS,
            "num_leaves": 15,  # Simpler model for coarse estimate
            "n_estimators": 1000,
            "min_child_samples": 30,
            **(coarse_params or {}),
        }

        self._refiner_params: dict[str, Any] = {
            **LightGBMModel.DEFAULT_PARAMS,
            "num_leaves": 63,
            "n_estimators": 2000,
            "learning_rate": 0.03,  # Slower learning for residuals
            "reg_alpha": 0.3,  # Stronger regularization
            "reg_lambda": 1.0,
            **(refiner_params or {}),
        }

        self._coarse_model: LightGBMModel | None = None
        self._refiner_model: LightGBMModel | None = None
        self._fitted = False
        self._feature_names: list[str] = []
        self._actual_coarse_features: list[str] = []
        self._refiner_features: list[str] = []

    @property
    def name(self) -> str:
        """Return model name."""
        return "residual"

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
        sample_weight: npt.NDArray[Any] | None = None,
    ) -> "ResidualModel":
        """Fit two-stage model.

        Stage 1: Train coarse model on event-level features.
        Stage 2: Compute residuals, train refiner on remaining features.

        Args:
            X_train: Training features
            y_train: Training target (log-space)
            X_val: Validation features
            y_val: Validation target (log-space)
            sample_weight: Optional per-sample weights

        Returns:
            self
        """
        self._feature_names = list(X_train.columns)

        # Partition features into coarse and refiner sets
        self._actual_coarse_features = [
            f for f in self._coarse_feature_names if f in X_train.columns
        ]
        self._refiner_features = [
            f for f in X_train.columns if f not in self._actual_coarse_features
        ]

        if not self._actual_coarse_features:
            raise ValueError(
                "No coarse features found in training data. "
                f"Expected some of: {self._coarse_feature_names}"
            )

        print(
            f"Residual model: {len(self._actual_coarse_features)} coarse features, "
            f"{len(self._refiner_features)} refiner features"
        )

        # Stage 1: Coarse estimator
        print("  Stage 1: Training coarse estimator...")
        X_coarse_train = X_train[self._actual_coarse_features]
        X_coarse_val = X_val[self._actual_coarse_features] if X_val is not None else None

        self._coarse_model = LightGBMModel(params=dict(self._coarse_params))
        try:
            self._coarse_model.fit(
                X_coarse_train,
                y_train,
                X_coarse_val,
                y_val,
                sample_weight=sample_weight,
            )
        except TypeError:
            self._coarse_model.fit(X_coarse_train, y_train, X_coarse_val, y_val)

        # Compute residuals on training set
        coarse_preds_train = self._coarse_model.predict(X_coarse_train)
        residuals_train = pd.Series(
            np.asarray(y_train) - coarse_preds_train,
            index=y_train.index,
        )

        # Compute residuals on validation set
        residuals_val = None
        X_refiner_val = None
        if X_val is not None and y_val is not None:
            coarse_preds_val = self._coarse_model.predict(X_coarse_val)
            residuals_val = pd.Series(
                np.asarray(y_val) - coarse_preds_val,
                index=y_val.index,
            )
            X_refiner_val = X_val[self._refiner_features]

        # Stage 2: Residual refiner (uses ALL features except coarse)
        print("  Stage 2: Training residual refiner...")
        X_refiner_train = X_train[self._refiner_features]

        self._refiner_model = LightGBMModel(params=dict(self._refiner_params))
        try:
            self._refiner_model.fit(
                X_refiner_train,
                residuals_train,
                X_refiner_val,
                residuals_val,
                sample_weight=sample_weight,
            )
        except TypeError:
            self._refiner_model.fit(
                X_refiner_train,
                residuals_train,
                X_refiner_val,
                residuals_val,
            )

        # Diagnostic: report residual statistics
        print(
            f"  Residual stats: mean={residuals_train.mean():.4f}, "
            f"std={residuals_train.std():.4f}, "
            f"abs_mean={residuals_train.abs().mean():.4f}"
        )

        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> npt.NDArray[Any]:
        """Make predictions: coarse + residual.

        Args:
            X: Feature DataFrame

        Returns:
            Array of predictions (log-space)
        """
        if not self._fitted or self._coarse_model is None or self._refiner_model is None:
            raise RuntimeError("Model must be fitted before predicting")

        coarse_preds = self._coarse_model.predict(X[self._actual_coarse_features])
        residual_preds = self._refiner_model.predict(X[self._refiner_features])

        return np.asarray(coarse_preds + residual_preds)

    def get_feature_importance(self) -> dict[str, float]:
        """Get combined feature importance from both stages.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self._fitted:
            return {}

        combined: dict[str, float] = {}

        # Coarse model importance (scaled by ~0.65, its typical contribution)
        if self._coarse_model is not None:
            coarse_imp = self._coarse_model.get_feature_importance()
            for feat, imp in coarse_imp.items():
                combined[feat] = imp * 0.65

        # Refiner model importance (scaled by ~0.35)
        if self._refiner_model is not None:
            refiner_imp = self._refiner_model.get_feature_importance()
            for feat, imp in refiner_imp.items():
                combined[feat] = combined.get(feat, 0.0) + imp * 0.35

        # Normalize
        total = sum(combined.values())
        if total > 0:
            combined = {k: v / total for k, v in combined.items()}

        return dict(sorted(combined.items(), key=lambda x: -x[1])[:20])

    def save(self, path: Path) -> None:
        """Save model to disk.

        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "coarse_model": self._coarse_model,
                "refiner_model": self._refiner_model,
                "coarse_feature_names": self._coarse_feature_names,
                "actual_coarse_features": self._actual_coarse_features,
                "refiner_features": self._refiner_features,
                "coarse_params": self._coarse_params,
                "refiner_params": self._refiner_params,
                "feature_names": self._feature_names,
                "fitted": self._fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "ResidualModel":
        """Load model from disk.

        Args:
            path: Path to load model from

        Returns:
            Loaded model
        """
        data = joblib.load(path)

        model = cls(
            coarse_features=data["coarse_feature_names"],
            coarse_params=data["coarse_params"],
            refiner_params=data["refiner_params"],
        )
        model._coarse_model = data["coarse_model"]
        model._refiner_model = data["refiner_model"]
        model._actual_coarse_features = data["actual_coarse_features"]
        model._refiner_features = data["refiner_features"]
        model._feature_names = data["feature_names"]
        model._fitted = data["fitted"]

        return model

    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        return {
            "coarse_features": self._coarse_feature_names,
            "coarse_params": self._coarse_params,
            "refiner_params": self._refiner_params,
        }
