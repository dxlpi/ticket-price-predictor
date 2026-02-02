"""Baseline Ridge regression model."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ticket_price_predictor.ml.models.base import PriceModel


class BaselineModel(PriceModel):
    """Ridge regression baseline model.

    Simple, interpretable baseline using Ridge regression with
    standard scaling for numeric features and one-hot encoding
    for categorical features.
    """

    # Default categorical and numeric features
    CATEGORICAL_FEATURES = [
        "seat_zone_encoded",
        "event_type_encoded",
        "city_tier",
        "day_of_week",
        "urgency_bucket",
        "popularity_tier",
        "venue_capacity_bucket",
    ]

    NUMERIC_FEATURES = [
        "days_to_event",
        "days_to_event_squared",
        "zone_price_ratio",
        "row_numeric",
        "is_weekend",
        "is_kpop",
        "is_major_artist",
        "is_country",
        "is_floor",
        "is_ga",
        "is_last_week",
    ]

    def __init__(
        self,
        alpha: float = 1.0,
        categorical_features: list[str] | None = None,
        numeric_features: list[str] | None = None,
    ) -> None:
        """Initialize baseline model.

        Args:
            alpha: Ridge regularization strength
            categorical_features: List of categorical feature names
            numeric_features: List of numeric feature names
        """
        self._alpha = alpha
        self._categorical_features = categorical_features or self.CATEGORICAL_FEATURES
        self._numeric_features = numeric_features or self.NUMERIC_FEATURES
        self._pipeline: Pipeline | None = None
        self._fitted = False
        self._feature_names: list[str] = []

    @property
    def name(self) -> str:
        """Return model name."""
        return "baseline_ridge"

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
    ) -> "BaselineModel":
        """Fit the model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (unused for Ridge)
            y_val: Validation target (unused for Ridge)

        Returns:
            self
        """
        # Filter to available features
        cat_cols = [c for c in self._categorical_features if c in X_train.columns]
        num_cols = [c for c in self._numeric_features if c in X_train.columns]

        self._feature_names = cat_cols + num_cols

        # Build preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    cat_cols,
                ),
            ],
            remainder="drop",
        )

        # Full pipeline
        self._pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("regressor", Ridge(alpha=self._alpha)),
            ]
        )

        # Fit
        self._pipeline.fit(X_train, y_train)
        self._fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.

        Args:
            X: Feature DataFrame

        Returns:
            Array of predictions
        """
        if not self._fitted or self._pipeline is None:
            raise RuntimeError("Model must be fitted before predicting")

        return self._pipeline.predict(X)

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance from Ridge coefficients.

        Returns:
            Dictionary mapping feature names to absolute coefficient values
        """
        if not self._fitted or self._pipeline is None:
            return {}

        ridge = self._pipeline.named_steps["regressor"]
        preprocessor = self._pipeline.named_steps["preprocessor"]

        # Get feature names after preprocessing
        try:
            feature_names = preprocessor.get_feature_names_out()
        except AttributeError:
            feature_names = [f"feature_{i}" for i in range(len(ridge.coef_))]

        # Map to importance (absolute coefficient values)
        importance = dict(zip(feature_names, np.abs(ridge.coef_)))

        # Sort by importance
        return dict(sorted(importance.items(), key=lambda x: -x[1])[:20])

    def save(self, path: Path) -> None:
        """Save model to disk.

        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "pipeline": self._pipeline,
                "alpha": self._alpha,
                "categorical_features": self._categorical_features,
                "numeric_features": self._numeric_features,
                "feature_names": self._feature_names,
                "fitted": self._fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "BaselineModel":
        """Load model from disk.

        Args:
            path: Path to load model from

        Returns:
            Loaded model
        """
        data = joblib.load(path)

        model = cls(
            alpha=data["alpha"],
            categorical_features=data["categorical_features"],
            numeric_features=data["numeric_features"],
        )
        model._pipeline = data["pipeline"]
        model._feature_names = data["feature_names"]
        model._fitted = data["fitted"]

        return model

    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        return {
            "alpha": self._alpha,
            "categorical_features": self._categorical_features,
            "numeric_features": self._numeric_features,
        }
