"""Stacking ensemble V2 with LightGBM + ResidualModel + optional Neural base learners.

KNOWN LIMITATION: Feature pipeline is fit on the full training set before
OOF splitting. This means OOF predictions for target-encoded features are
slightly optimistic. Accepted trade-off: re-fitting the pipeline K times
is impractical (K x fit cost), and Bayesian smoothing (factor=20) limits
the bias. Documented here for transparency.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from ticket_price_predictor.ml.models.base import PriceModel
from ticket_price_predictor.ml.models.lightgbm_model import LightGBMModel
from ticket_price_predictor.ml.models.residual import ResidualModel

if TYPE_CHECKING:
    from ticket_price_predictor.ml.training.target_transforms import TargetTransform


def _default_v2_base_configs() -> list[dict[str, Any]]:
    """Return default base learner configurations for V2.

    Diversity comes from:
    - Two LightGBM variants with different depth/learning-rate/iterations
    - ResidualModel (two-stage coarse+refiner decomposition)
    """
    return [
        {
            "name": "lgb_huber",
            "cls": LightGBMModel,
            "params": {**LightGBMModel.DEFAULT_PARAMS, "n_estimators": 2000},
        },
        {
            "name": "lgb_deeper",
            "cls": LightGBMModel,
            "params": {
                **LightGBMModel.DEFAULT_PARAMS,
                "num_leaves": 63,
                "learning_rate": 0.05,
                "n_estimators": 1500,
            },
        },
        {
            "name": "residual",
            "cls": ResidualModel,
            "params": {},  # ResidualModel uses its own defaults
        },
    ]


class StackingEnsembleV2(PriceModel):
    """Stacking ensemble V2: LightGBM variants + ResidualModel + optional Neural.

    Uses out-of-fold predictions to prevent meta-learner leakage.

    KNOWN LIMITATION: Feature pipeline is fit on the full training set before
    OOF splitting. OOF predictions for target-encoded features are slightly
    optimistic due to Bayesian smoothing. Accepted trade-off -- re-fitting the
    pipeline K times is impractical.

    Architecture:
        Layer 0 (Base Learners): LightGBM GBDT+Huber, LightGBM deeper variant,
        ResidualModel (coarse+refiner), and optionally TabularNeuralModel.
        Trained with expanding-window temporal CV. Out-of-fold predictions
        form the meta-features.

        Layer 1 (Meta-Learner): Ridge regression trained on OOF predictions
        from Layer 0. Simple linear combination prevents overfitting at the
        meta level.

    Leakage prevention:
        - Base learners use temporal group CV (expanding window)
        - Meta-learner trains only on out-of-fold predictions
        - Feature pipeline is NOT re-fit per fold (already fit on full train set
          by ModelTrainer before reaching this model)
    """

    def __init__(
        self,
        base_configs: list[dict[str, Any]] | None = None,
        n_folds: int = 5,
        meta_alpha: float = 1.0,
        include_neural: bool = False,
        anchor_feature: str | None = None,
        target_transform: TargetTransform | None = None,
    ) -> None:
        """Initialize stacking ensemble V2.

        Args:
            base_configs: List of base learner configs. Each dict has:
                - "name": str identifier
                - "cls": PriceModel subclass
                - "params": dict of model parameters
            n_folds: Number of temporal CV folds for OOF predictions
            meta_alpha: Ridge regularization for meta-learner
            include_neural: Whether to include TabularNeuralModel (requires pytorch-tabular)
            anchor_feature: Optional feature name to include as meta-feature
                alongside base predictions (e.g., "event_section_median_price")
            target_transform: Optional transform used to convert OOF predictions
                back to dollar space for per-base-learner MAE reporting. When
                None (default), dollar-space metrics are not computed.
        """
        self._base_configs = base_configs or _default_v2_base_configs()
        self._n_folds = n_folds
        self._meta_alpha = meta_alpha
        self._include_neural = include_neural
        self._anchor_feature = anchor_feature
        self._target_transform = target_transform

        # Add neural model if requested and available
        if include_neural:
            try:
                from ticket_price_predictor.ml.models.neural import (  # noqa: F401
                    TabularNeuralModel,
                )

                self._base_configs = list(self._base_configs) + [
                    {
                        "name": "neural",
                        "cls": TabularNeuralModel,
                        "params": {},
                    }
                ]
            except ImportError:
                pass  # Neural model unavailable, proceed without it

        # Fitted state
        self._base_models: list[PriceModel] = []
        self._meta_model: Ridge | None = None
        self._meta_scaler: StandardScaler | None = None
        self._fitted = False
        self._feature_names: list[str] = []
        self._base_importances: list[dict[str, float]] = []
        self._oof_metrics: dict[str, float] = {}

    @property
    def name(self) -> str:
        """Return model name."""
        return "stacking_v2"

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
        timestamps: pd.Series | npt.NDArray[Any] | None = None,
    ) -> StackingEnsembleV2:
        """Fit stacking ensemble V2.

        Step 1: Generate OOF predictions via temporal K-fold CV.
        Step 2: Train final base models on full training set.
        Step 3: Train meta-learner on OOF predictions.

        Args:
            X_train: Training features (already transformed by feature pipeline)
            y_train: Training target (log-space)
            X_val: Validation features (used for early stopping of final base models)
            y_val: Validation target
            sample_weight: Optional per-sample weights
            timestamps: Optional per-row timestamps used to globally sort X_train
                by time. Required when upstream splitter produces non-temporally-
                ordered rows (e.g. artist-stratified splits); without it the
                positional k-fold indices do not correspond to true temporal order.

        Returns:
            self
        """
        self._feature_names = list(X_train.columns)

        # Upstream splitter (artist-stratified concat) produces rows ordered by
        # (artist_group, time_within_artist), not globally by time. Sort here so
        # the positional k-fold indices below correspond to true temporal order.
        if timestamps is not None:
            ts = pd.Series(timestamps).reset_index(drop=True)
            if len(ts) != len(X_train):
                raise ValueError(
                    f"timestamps length ({len(ts)}) does not match X_train ({len(X_train)})"
                )
            sort_idx = ts.argsort(kind="stable").to_numpy()
            X_train = X_train.iloc[sort_idx].reset_index(drop=True)
            y_train = pd.Series(y_train).iloc[sort_idx].reset_index(drop=True)
            if sample_weight is not None:
                sample_weight = np.asarray(sample_weight)[sort_idx]

        n_samples = len(X_train)
        n_base = len(self._base_configs)

        print(f"StackingV2: {n_base} base learners, {self._n_folds}-fold temporal CV")

        # Step 1: Generate OOF predictions
        # With expanding-window temporal CV, the first fold's training samples
        # are never covered as val → track covered indices, use only those for
        # meta-learner training to avoid poisoning it with zero-filled rows.
        oof_preds = np.zeros((n_samples, n_base))
        oof_covered = np.zeros(n_samples, dtype=bool)
        fold_indices = self._temporal_kfold_indices(n_samples)

        for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
            print(
                f"  Fold {fold_idx + 1}/{len(fold_indices)}: "
                f"train={len(train_idx)}, val={len(val_idx)}"
            )

            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]

            fold_weight = sample_weight[train_idx] if sample_weight is not None else None

            for model_idx, config in enumerate(self._base_configs):
                model = self._instantiate_base_model(config)
                try:
                    model.fit(  # type: ignore[call-arg]
                        X_fold_train,
                        y_fold_train,
                        X_fold_val,
                        y_fold_val,
                        sample_weight=fold_weight,
                    )
                except TypeError:
                    model.fit(X_fold_train, y_fold_train, X_fold_val, y_fold_val)

                oof_preds[val_idx, model_idx] = model.predict(X_fold_val)

            oof_covered[val_idx] = True

        print(
            f"  OOF coverage: {oof_covered.sum()}/{n_samples} samples ({100 * oof_covered.mean():.1f}%)"
        )

        # Compute per-base-learner dollar-space OOF MAE (requires target_transform)
        if self._target_transform is not None:
            y_covered = np.asarray(y_train)[oof_covered]
            y_dollars = self._target_transform.inverse_transform(y_covered)
            base_names = [c["name"] for c in self._base_configs]
            self._oof_metrics = {}
            for i, bname in enumerate(base_names):
                preds_dollars = self._target_transform.inverse_transform(oof_preds[oof_covered, i])
                self._oof_metrics[bname] = float(mean_absolute_error(y_dollars, preds_dollars))
            print("  OOF dollar-space MAE per base learner:")
            for bname, mae in self._oof_metrics.items():
                print(f"    {bname}: ${mae:.2f}")

        # Step 2: Train final base models on full training set
        print("  Training final base models on full training set...")
        self._base_models = []
        self._base_importances = []

        for config in self._base_configs:
            model = self._instantiate_base_model(config)
            try:
                model.fit(X_train, y_train, X_val, y_val, sample_weight=sample_weight)  # type: ignore[call-arg]
            except TypeError:
                model.fit(X_train, y_train, X_val, y_val)
            self._base_models.append(model)
            self._base_importances.append(model.get_feature_importance())
            print(f"    {config['name']}: fitted")

        # Step 3: Train meta-learner on OOF predictions (covered samples only)
        oof_X = X_train.iloc[oof_covered]
        oof_y = y_train.iloc[oof_covered]
        meta_features = self._build_meta_features(oof_preds[oof_covered], oof_X)

        self._meta_scaler = StandardScaler()
        meta_scaled = self._meta_scaler.fit_transform(meta_features)

        self._meta_model = Ridge(alpha=self._meta_alpha)
        self._meta_model.fit(meta_scaled, oof_y)
        # Store Ridge OOF predictions for dollar-space meta-learner MAE
        self._meta_oof_preds: npt.NDArray[Any] = np.asarray(self._meta_model.predict(meta_scaled))

        # Add Ridge meta-learner dollar-space OOF MAE if transform is available
        if self._target_transform is not None:
            y_covered = np.asarray(y_train)[oof_covered]
            y_dollars = self._target_transform.inverse_transform(y_covered)
            meta_preds_dollars = self._target_transform.inverse_transform(self._meta_oof_preds)
            ridge_mae = float(mean_absolute_error(y_dollars, meta_preds_dollars))
            self._oof_metrics["ridge"] = ridge_mae
            print(f"    ridge (meta): ${ridge_mae:.2f}")

        # Report meta-learner weights
        meta_names = [c["name"] for c in self._base_configs]
        if self._anchor_feature:
            meta_names.append(self._anchor_feature)
        weights = self._meta_model.coef_
        print("  Meta-learner weights:")
        for mname, w in zip(meta_names, weights, strict=False):
            print(f"    {mname}: {w:.4f}")

        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> npt.NDArray[Any]:
        """Make predictions using stacking ensemble V2.

        Args:
            X: Feature DataFrame

        Returns:
            Array of predictions
        """
        if not self._fitted or self._meta_model is None:
            raise RuntimeError("Model must be fitted before predicting")

        # Get base predictions
        base_preds = np.column_stack([model.predict(X) for model in self._base_models])

        # Build meta-features and predict
        meta_features = self._build_meta_features(base_preds, X)

        assert self._meta_scaler is not None
        meta_scaled = self._meta_scaler.transform(meta_features)

        return np.asarray(self._meta_model.predict(meta_scaled))

    def get_feature_importance(self) -> dict[str, float]:
        """Get aggregated feature importance across base models.

        Weights each base model's importance by its meta-learner coefficient.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self._fitted or self._meta_model is None:
            return {}

        # Weight base model importances by meta-learner coefficients
        meta_weights = np.abs(self._meta_model.coef_[: len(self._base_models)])
        total_weight = meta_weights.sum()
        if total_weight == 0:
            return {}

        meta_weights = meta_weights / total_weight

        aggregated: dict[str, float] = {}
        for importance, weight in zip(self._base_importances, meta_weights, strict=False):
            for feat, imp in importance.items():
                aggregated[feat] = aggregated.get(feat, 0.0) + imp * weight

        # Normalize
        total = sum(aggregated.values())
        if total > 0:
            aggregated = {k: v / total for k, v in aggregated.items()}

        return dict(sorted(aggregated.items(), key=lambda x: -x[1])[:20])

    def save(self, path: Path) -> None:
        """Save model to disk.

        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "base_models": self._base_models,
                "base_configs": [
                    {"name": c["name"], "cls": c["cls"], "params": c["params"]}
                    for c in self._base_configs
                ],
                "meta_model": self._meta_model,
                "meta_scaler": self._meta_scaler,
                "meta_alpha": self._meta_alpha,
                "n_folds": self._n_folds,
                "include_neural": self._include_neural,
                "anchor_feature": self._anchor_feature,
                "feature_names": self._feature_names,
                "base_importances": self._base_importances,
                "fitted": self._fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> StackingEnsembleV2:
        """Load model from disk.

        Args:
            path: Path to load model from

        Returns:
            Loaded model
        """
        data = joblib.load(path)

        model = cls(
            base_configs=data["base_configs"],
            n_folds=data["n_folds"],
            meta_alpha=data["meta_alpha"],
            include_neural=False,  # Configs already loaded, don't re-add neural
            anchor_feature=data.get("anchor_feature"),
        )
        model._base_models = data["base_models"]
        model._meta_model = data["meta_model"]
        model._meta_scaler = data["meta_scaler"]
        model._feature_names = data["feature_names"]
        model._base_importances = data["base_importances"]
        model._fitted = data["fitted"]

        return model

    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        return {
            "n_folds": self._n_folds,
            "meta_alpha": self._meta_alpha,
            "include_neural": self._include_neural,
            "anchor_feature": self._anchor_feature,
            "n_base_models": len(self._base_configs),
            "base_model_names": [c["name"] for c in self._base_configs],
        }

    def _temporal_kfold_indices(
        self,
        n_samples: int,
    ) -> list[tuple[npt.NDArray[Any], npt.NDArray[Any]]]:
        """Generate expanding-window temporal CV fold indices.

        Each fold uses an expanding training window with a fixed-size
        validation window. This respects temporal ordering (no future data
        leaks into training).

        Args:
            n_samples: Total number of samples

        Returns:
            List of (train_indices, val_indices) tuples
        """
        fold_size = n_samples // (self._n_folds + 1)
        folds = []

        for i in range(self._n_folds):
            train_end = fold_size * (i + 1)
            val_end = min(train_end + fold_size, n_samples)

            if val_end <= train_end:
                break

            train_idx = np.arange(train_end)
            val_idx = np.arange(train_end, val_end)
            folds.append((train_idx, val_idx))

        return folds

    def _build_meta_features(
        self,
        base_preds: npt.NDArray[Any],
        X: pd.DataFrame,
    ) -> npt.NDArray[Any]:
        """Build meta-feature matrix from base predictions and optional anchor.

        Args:
            base_preds: Array of shape (n_samples, n_base_models)
            X: Original feature DataFrame (for anchor feature extraction)

        Returns:
            Meta-feature array
        """
        if self._anchor_feature and self._anchor_feature in X.columns:
            anchor = X[self._anchor_feature].values.reshape(-1, 1)
            return np.hstack([base_preds, anchor])
        return base_preds

    @staticmethod
    def _instantiate_base_model(config: dict[str, Any]) -> PriceModel:
        """Instantiate a base model from its config.

        Handles the difference between models that accept ``params=``
        keyword (LightGBMModel) and models that use their own defaults
        (ResidualModel).

        Args:
            config: Base learner config dict with "cls" and "params" keys.

        Returns:
            Instantiated (unfitted) model.
        """
        if config["params"]:
            try:
                return config["cls"](params=dict(config["params"]))  # type: ignore[no-any-return]
            except TypeError:
                return config["cls"]()  # type: ignore[no-any-return]
        return config["cls"]()  # type: ignore[no-any-return]
