"""Two-stage residual prediction models.

Stage 1 (Coarse Estimator): Predicts price from event-level pricing features.
Stage 2 (Residual Refiner): Predicts the residual using all other features.

Final prediction = coarse_prediction + residual_prediction.

This decomposition frees Stage 2 from being dominated by the top-3 event
pricing features (which account for 88.4% of importance in the single model).

Also provides ``HierarchicalResidualModel`` — a leak-safe variant where
Stage 1 trains on *event-level* aggregates and Stage 2 sees only OOF-predicted
event bases during training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import KFold

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
    ) -> ResidualModel:
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
    def load(cls, path: Path) -> ResidualModel:
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


# ---------------------------------------------------------------------------
# HierarchicalResidualModel
# ---------------------------------------------------------------------------

# Features that encode the target — excluded from stage 1 to prevent trivial
# learning (stage 1 should generalise from context, not memorise the label).
STAGE1_EXCLUDE_FEATURES = [
    "event_median_price",
    "event_zone_median_price",
    "event_section_median_price",
    "event_zone_price_ratio",
    "event_section_price_ratio",
    # Relative-pricing features derived from event median
    "section_median_deviation",
    "zone_median_deviation",
]


class HierarchicalResidualModel(PriceModel):
    """Two-stage hierarchical model with OOF event-base predictions.

    Stage 1: Event-level model predicts median log-price per event from
    non-target-encoding features (one row per event, deduplicated).

    Stage 2: Listing-level model predicts the *deviation* of each listing
    from the Stage-1 base, using all features.

    OOF (out-of-fold) predictions for Stage 1 prevent train/inference
    distribution shift in Stage 2.

    ``event_id`` **must** be present as a column in ``X_train`` passed to
    :meth:`fit`.  It is extracted and dropped before any tree training.
    At inference time (:meth:`predict`) ``event_id`` is optional — if absent
    the model simply predicts Stage 1 from each row's features directly.
    """

    N_FOLDS = 5

    def __init__(
        self,
        stage1_params: dict[str, Any] | None = None,
        stage2_params: dict[str, Any] | None = None,
    ) -> None:
        self._stage1_params: dict[str, Any] = {
            **LightGBMModel.DEFAULT_PARAMS,
            "num_leaves": 15,
            "n_estimators": 1000,
            "min_child_samples": 30,
            **(stage1_params or {}),
        }

        self._stage2_params: dict[str, Any] = {
            **LightGBMModel.DEFAULT_PARAMS,
            "num_leaves": 63,
            "n_estimators": 2000,
            "learning_rate": 0.03,
            "reg_alpha": 0.3,
            "reg_lambda": 1.0,
            **(stage2_params or {}),
        }

        self._stage1_model: LightGBMModel | None = None
        self._stage2_model: LightGBMModel | None = None
        self._fitted = False
        self._stage1_cols: list[str] = []
        self._stage2_cols: list[str] = []

    # -- PriceModel interface -------------------------------------------------

    @property
    def name(self) -> str:
        return "hierarchical"

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # -- fit ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        sample_weight: npt.NDArray[Any] | None = None,
    ) -> HierarchicalResidualModel:
        """Fit the two-stage hierarchical model.

        Args:
            X_train: Training features **including an ``event_id`` column**.
            y_train: Training target (log-space).
            X_val: Validation features (may include ``event_id``).
            y_val: Validation target (log-space).
            sample_weight: Optional per-sample weights (applied to stage 2).

        Returns:
            self
        """
        if "event_id" not in X_train.columns:
            raise ValueError("HierarchicalResidualModel requires an 'event_id' column in X_train")

        # 1. Extract and drop event_id
        event_ids_train = X_train["event_id"].copy()
        X_features = X_train.drop(columns=["event_id"])

        # 2. Stage 1 feature set: exclude target-encoding features
        stage1_cols = [c for c in X_features.columns if c not in STAGE1_EXCLUDE_FEATURES]

        # 3. Build event-level dataset (one row per event, first occurrence
        #    acts as proxy for the event-level median in log-space).
        event_df = (
            X_features[stage1_cols]
            .assign(_eid=event_ids_train.values, _y=y_train.values)
            .groupby("_eid", as_index=False)
            .first()
        )
        X_event = event_df[stage1_cols]
        y_event = event_df["_y"]
        event_id_arr = event_df["_eid"].values

        print(
            f"HierarchicalResidual: {len(event_id_arr)} events, "
            f"{len(stage1_cols)} stage-1 features (excl {len(STAGE1_EXCLUDE_FEATURES)} target-enc), "
            f"{len(X_features.columns)} stage-2 features"
        )

        # 4. OOF stage-1 predictions (K-fold by event)
        print(f"  Stage 1: {self.N_FOLDS}-fold OOF on event-level rows...")
        oof_event_base: dict[Any, float] = {}
        kf = KFold(n_splits=self.N_FOLDS, shuffle=False)
        for fold_idx, (trn_idx, val_idx) in enumerate(kf.split(X_event)):
            fold_model = LightGBMModel(params=dict(self._stage1_params))
            fold_model.fit(
                X_event.iloc[trn_idx],
                y_event.iloc[trn_idx],
                X_event.iloc[val_idx],
                y_event.iloc[val_idx],
            )
            preds = fold_model.predict(X_event.iloc[val_idx])
            for eid, pred in zip(event_id_arr[val_idx], preds, strict=False):
                oof_event_base[eid] = float(pred)
            print(f"    Fold {fold_idx + 1}/{self.N_FOLDS}: {len(val_idx)} events")

        # 5. Map OOF base back to listing-level
        oof_base_series = event_ids_train.map(oof_event_base)
        # Safety: fill any unmapped events with global mean
        oof_base_series = oof_base_series.fillna(y_train.mean())

        # 6. Stage-2 target: listing deviation from OOF base
        y_deviation = y_train - oof_base_series

        print(
            f"  Deviation stats: mean={y_deviation.mean():.4f}, "
            f"std={y_deviation.std():.4f}, abs_mean={y_deviation.abs().mean():.4f}"
        )

        # 7. Retrain stage 1 on ALL training events for inference
        print("  Stage 1: Retraining on all events for inference...")
        self._stage1_model = LightGBMModel(params=dict(self._stage1_params))
        self._stage1_model.fit(X_event, y_event)

        # 8. Train stage 2 on full listing-level training set
        print("  Stage 2: Training listing-level deviation model...")
        X_stage2 = X_features.copy()

        # Prepare val set for stage 2 (compute val deviation if possible)
        X_stage2_val: pd.DataFrame | None = None
        y_deviation_val: pd.Series | None = None

        if X_val is not None and y_val is not None:
            X_val_clean = X_val.drop(columns=["event_id"], errors="ignore")
            # Predict val event bases from the final stage-1 model
            X_val_s1 = X_val_clean[[c for c in stage1_cols if c in X_val_clean.columns]]
            val_base = self._stage1_model.predict(X_val_s1)
            y_deviation_val = y_val - pd.Series(val_base, index=y_val.index)
            X_stage2_val = X_val_clean

        self._stage2_model = LightGBMModel(params=dict(self._stage2_params))
        try:
            self._stage2_model.fit(
                X_stage2,
                y_deviation,
                X_stage2_val,
                y_deviation_val,
                sample_weight=sample_weight,
            )
        except TypeError:
            # Model does not accept sample_weight
            self._stage2_model.fit(
                X_stage2,
                y_deviation,
                X_stage2_val,
                y_deviation_val,
            )

        self._stage1_cols = stage1_cols
        self._stage2_cols = list(X_features.columns)
        self._fitted = True
        return self

    # -- predict --------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> npt.NDArray[Any]:
        """Predict: event_base + listing_deviation.

        Args:
            X: Feature DataFrame (``event_id`` column optional at inference).

        Returns:
            Array of predictions (log-space).
        """
        if not self._fitted or self._stage1_model is None or self._stage2_model is None:
            raise RuntimeError("Model must be fitted before predicting")

        X_clean = X.drop(columns=["event_id"], errors="ignore")

        # Stage 1: event-level base
        X_s1 = X_clean[self._stage1_cols]
        event_base = self._stage1_model.predict(X_s1)

        # Stage 2: listing deviation
        X_s2 = X_clean[self._stage2_cols]
        deviation = self._stage2_model.predict(X_s2)

        return np.asarray(event_base + deviation)

    # -- feature importance ---------------------------------------------------

    def get_feature_importance(self) -> dict[str, float]:
        """Combined feature importance (60% stage-1, 40% stage-2)."""
        if not self._fitted:
            return {}

        combined: dict[str, float] = {}

        if self._stage1_model is not None:
            for feat, imp in self._stage1_model.get_feature_importance().items():
                combined[feat] = imp * 0.60

        if self._stage2_model is not None:
            for feat, imp in self._stage2_model.get_feature_importance().items():
                combined[feat] = combined.get(feat, 0.0) + imp * 0.40

        total = sum(combined.values())
        if total > 0:
            combined = {k: v / total for k, v in combined.items()}

        return dict(sorted(combined.items(), key=lambda x: -x[1])[:20])

    # -- save / load ----------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "stage1_model": self._stage1_model,
                "stage2_model": self._stage2_model,
                "stage1_params": self._stage1_params,
                "stage2_params": self._stage2_params,
                "stage1_cols": self._stage1_cols,
                "stage2_cols": self._stage2_cols,
                "fitted": self._fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> HierarchicalResidualModel:
        """Load model from disk."""
        data = joblib.load(path)
        model = cls(
            stage1_params=data["stage1_params"],
            stage2_params=data["stage2_params"],
        )
        model._stage1_model = data["stage1_model"]
        model._stage2_model = data["stage2_model"]
        model._stage1_cols = data["stage1_cols"]
        model._stage2_cols = data["stage2_cols"]
        model._fitted = data["fitted"]
        return model

    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        return {
            "stage1_params": self._stage1_params,
            "stage2_params": self._stage2_params,
        }
