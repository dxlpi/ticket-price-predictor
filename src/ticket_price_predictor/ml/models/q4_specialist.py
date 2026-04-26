"""Q4 specialist ensemble — conditional Phase 7 escalation for v38.

Architecture: a binary-routed two-model design wrapped in a `PriceModel` subclass.

1. **Router**: lightweight LightGBM binary classifier predicting `P(price >= $310)`
   from the v38 feature set. Trained on all training rows.

2. **Q4 specialist**: `StackingEnsembleV2` trained only on training rows with
   `price >= $310` (~25% of training data). Same architecture as v38.

3. **Inference**: at predict time, route each row through the router. The router
   probability blends the two predictions:
   `final = (1 - p) * v38_pred + p * q4spec_pred`.

The v38 prediction is loaded from the existing v38 artifact at fit time. The
operator MUST run q4_specialist with identical pipeline flags as v38 (no
`--no-listing-structural` etc.) so the loaded v38 pipeline's feature_names
match the q4spec pipeline's output.

Activation:
- Triggered manually after `stacking_v2_v38` training when AC1 (seen MAE ratio)
  or AC2 (Q4 MAE ratio) is unmet.
- Operator command:
  `python scripts/train_model.py --model q4_specialist --version v38_q4spec`

This is documented as "expected, not exceptional" in the v38 plan because the
historical evidence (v28→v36 changes) suggests B+C+D alone may deliver only
$5–15 in seen-MAE delta vs. the ~$20–25 AC1 demands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd

from ticket_price_predictor.ml.models.base import PriceModel
from ticket_price_predictor.ml.models.stacking_v2 import StackingEnsembleV2

# Q4 threshold for both router label and routing decision (in raw dollars,
# matching plan math spec § C). Convert to log-target space when comparing
# against base predictions.
_Q4_PRICE_THRESHOLD: float = 310.0


class Q4SpecialistEnsemble(PriceModel):
    """Binary-routed ensemble: v38 (general) + Q4 specialist."""

    # Default paths for the standard v38 artifact layout. Overridable via
    # constructor args (e.g. for tests or non-standard versions).
    _DEFAULT_V38_ARTIFACT = Path("data/models/stacking_v2_v38.joblib")
    _DEFAULT_V38_PIPELINE = Path("data/models/stacking_v2_v38_pipeline.joblib")

    def __init__(
        self,
        v38_artifact_path: Path | str | None = None,
        v38_pipeline_path: Path | str | None = None,
        q4_threshold: float = _Q4_PRICE_THRESHOLD,
    ) -> None:
        """Initialize the Q4 specialist ensemble.

        Args:
            v38_artifact_path: Path to the trained v38 stacking_v2 model joblib.
                Defaults to data/models/stacking_v2_v38.joblib.
            v38_pipeline_path: Path to the trained v38 feature pipeline joblib.
                Defaults to data/models/stacking_v2_v38_pipeline.joblib.
            q4_threshold: Price threshold (raw dollars) for the binary router
                and for the Q4 training subset filter.
        """
        self._v38_artifact_path = (
            Path(v38_artifact_path) if v38_artifact_path is not None else self._DEFAULT_V38_ARTIFACT
        )
        self._v38_pipeline_path = (
            Path(v38_pipeline_path) if v38_pipeline_path is not None else self._DEFAULT_V38_PIPELINE
        )
        self._q4_threshold = q4_threshold

        # Loaded lazily in fit() so unpickling works
        self._v38_model: StackingEnsembleV2 | None = None
        self._v38_pipeline: Any | None = None

        # Trained components
        self._router: lgb.Booster | None = None
        self._q4_specialist: StackingEnsembleV2 | None = None
        self._fitted = False
        self._feature_names: list[str] = []

    @property
    def name(self) -> str:
        return "q4_specialist"

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _load_v38(self) -> None:
        """Load v38 model + pipeline from disk."""
        if self._v38_artifact_path is None:
            raise RuntimeError(
                "v38_artifact_path is required for Q4SpecialistEnsemble. "
                "Pass via constructor or model_kwargs."
            )
        if not self._v38_artifact_path.exists():
            raise FileNotFoundError(
                f"v38 artifact not found at {self._v38_artifact_path}. "
                "Train v38 first via: "
                "python scripts/train_model.py --model stacking_v2 --version v38"
            )
        self._v38_model = StackingEnsembleV2.load(self._v38_artifact_path)
        if self._v38_pipeline_path is not None and self._v38_pipeline_path.exists():
            self._v38_pipeline = joblib.load(self._v38_pipeline_path)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        sample_weight: npt.NDArray[Any] | None = None,
        timestamps: pd.Series | npt.NDArray[Any] | None = None,
    ) -> Q4SpecialistEnsemble:
        """Fit the router + Q4 specialist.

        The y_train values are in log-target space (log1p of dollar prices),
        so we convert the threshold to log-space for filtering.
        """
        self._load_v38()
        self._feature_names = list(X_train.columns)

        # Column-set assertion: q4spec X_train (post-filter) must match
        # v38 model's expected columns (also post-filter, stored on the model).
        # Use v38_model._feature_names — NOT v38_pipeline.feature_names which
        # is the pre-filter raw extractor output.
        assert self._v38_model is not None
        v38_columns = set(self._v38_model._feature_names)
        q4spec_columns = set(X_train.columns)
        extra = q4spec_columns - v38_columns
        missing = v38_columns - q4spec_columns
        if extra or missing:
            print(
                f"  WARNING: column drift between q4spec ({len(q4spec_columns)}) "
                f"and v38 ({len(v38_columns)}). Extra: {extra}; missing: {missing}. "
                "Will align via reorder + zero-fill at predict time."
            )

        # Router: LightGBM binary classifier on raw price >= $310
        log_threshold = float(np.log1p(self._q4_threshold))
        y_router = (y_train.to_numpy() >= log_threshold).astype(int)

        print(
            f"  Q4SpecialistEnsemble: training router on {len(X_train)} rows, "
            f"{y_router.sum()} positive ({100 * y_router.mean():.1f}%)"
        )

        router_data = lgb.Dataset(X_train, label=y_router, weight=sample_weight)
        router_params: dict[str, Any] = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "verbose": -1,
        }
        valid_sets = [router_data]
        if X_val is not None and y_val is not None:
            y_val_router = (y_val.to_numpy() >= log_threshold).astype(int)
            valid_sets.append(lgb.Dataset(X_val, label=y_val_router, reference=router_data))
        self._router = lgb.train(
            router_params,
            router_data,
            num_boost_round=500,
            valid_sets=valid_sets,
            callbacks=[lgb.early_stopping(stopping_rounds=50)] if len(valid_sets) > 1 else [],
        )

        # Q4 specialist: train stacking_v2 on the >= threshold subset
        q4_mask = y_train.to_numpy() >= log_threshold
        if q4_mask.sum() < 100:
            raise RuntimeError(
                f"Q4 training subset too small ({q4_mask.sum()} rows). Cannot train Q4 specialist."
            )
        X_train_q4 = X_train.iloc[q4_mask]
        y_train_q4 = y_train.iloc[q4_mask]

        if X_val is not None and y_val is not None:
            q4_val_mask = y_val.to_numpy() >= log_threshold
            X_val_q4 = X_val.iloc[q4_val_mask] if q4_val_mask.sum() > 0 else None
            y_val_q4 = y_val.iloc[q4_val_mask] if q4_val_mask.sum() > 0 else None
        else:
            X_val_q4 = None
            y_val_q4 = None

        print(f"  Q4SpecialistEnsemble: training Q4 specialist on {len(X_train_q4)} rows")
        sample_weight_q4 = np.asarray(sample_weight)[q4_mask] if sample_weight is not None else None
        timestamps_q4 = np.asarray(timestamps)[q4_mask] if timestamps is not None else None
        self._q4_specialist = StackingEnsembleV2(
            n_folds=5, include_quantile_bases=True, include_neural=False
        )
        self._q4_specialist.fit(
            X_train_q4,
            y_train_q4,
            X_val_q4,
            y_val_q4,
            sample_weight=sample_weight_q4,
            timestamps=timestamps_q4,
        )

        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> npt.NDArray[Any]:
        """Predict by blending v38 and Q4 specialist via router probability.

        Aligns X to v38's expected column order before forwarding; missing
        columns are zero-filled (Bayesian smoothing in the missing extractors
        means zero is a reasonable neutral value).
        """
        if not self._fitted or self._router is None or self._q4_specialist is None:
            raise RuntimeError("Q4SpecialistEnsemble must be fitted before predicting")

        # Lazy load v38 if not yet loaded (e.g. after pickle.loads)
        if self._v38_model is None:
            self._load_v38()
        assert self._v38_model is not None

        # Router probability and Q4 specialist prediction use q4spec's column set
        p_q4 = np.asarray(self._router.predict(X), dtype=np.float64)
        q4_pred = np.asarray(self._q4_specialist.predict(X), dtype=np.float64)

        # v38 prediction needs X in v38's column order; pad missing with 0
        v38_columns = self._v38_model._feature_names
        X_v38 = X.reindex(columns=v38_columns, fill_value=0.0)
        v38_pred = np.asarray(self._v38_model.predict(X_v38), dtype=np.float64)

        # Linear blend by router probability
        blended = (1.0 - p_q4) * v38_pred + p_q4 * q4_pred
        return blended

    def get_feature_importance(self) -> dict[str, float]:
        """Return aggregated importance from the Q4 specialist."""
        if not self._fitted or self._q4_specialist is None:
            return {}
        return self._q4_specialist.get_feature_importance()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "v38_artifact_path": str(self._v38_artifact_path)
                if self._v38_artifact_path
                else None,
                "v38_pipeline_path": str(self._v38_pipeline_path)
                if self._v38_pipeline_path
                else None,
                "q4_threshold": self._q4_threshold,
                "router": self._router,
                "q4_specialist": self._q4_specialist,
                "feature_names": self._feature_names,
                "fitted": self._fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> Q4SpecialistEnsemble:
        data = joblib.load(path)
        model = cls(
            v38_artifact_path=data["v38_artifact_path"],
            v38_pipeline_path=data.get("v38_pipeline_path"),
            q4_threshold=data["q4_threshold"],
        )
        model._router = data["router"]
        model._q4_specialist = data["q4_specialist"]
        model._feature_names = data["feature_names"]
        model._fitted = data["fitted"]
        return model

    def get_params(self) -> dict[str, Any]:
        return {
            "v38_artifact_path": str(self._v38_artifact_path) if self._v38_artifact_path else None,
            "q4_threshold": self._q4_threshold,
        }
