"""FT-Transformer neural network model with entity embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

try:
    from pytorch_tabular import TabularModel
    from pytorch_tabular.config import (
        DataConfig,
        OptimizerConfig,
        TrainerConfig,
    )
    from pytorch_tabular.models import (
        FTTransformerConfig,
    )

    PYTORCH_TABULAR_AVAILABLE = True
except ImportError:
    TabularModel = None
    PYTORCH_TABULAR_AVAILABLE = False

import joblib

from ticket_price_predictor.ml.models.base import PriceModel

# Categorical columns to embed — must be present as columns in X_train/X_val/X_predict
# These are raw categorical strings, concatenated alongside numeric features
CATEGORICAL_COLS = ["artist_or_team", "venue_name", "city", "day_of_week_str"]

# Embedding dimensions per categorical (n_unique -> embed_dim)
# These are maximum dims; pytorch_tabular auto-calculates actual dims
EMBEDDING_DIMS: dict[str, int] = {
    "artist_or_team": 32,
    "venue_name": 16,
    "city": 12,
    "day_of_week_str": 4,
}


class TabularNeuralModel(PriceModel):
    """FT-Transformer with entity embeddings for price prediction.

    Uses pytorch-tabular's FT-Transformer implementation. Learns dense
    embeddings for categorical entities (artist, venue, city) to capture
    latent relationships that tree-based models cannot.

    Requires pytorch-tabular to be installed:
        pip install pytorch-tabular

    If not available, fit() raises ImportError with a clear message.

    Architecture (CPU-feasible):
    - 2 attention blocks, d_model=64, 4 heads
    - Batch size 512, max 50 epochs, early stopping patience=10
    - Projected training time: <15 min on 197K rows on Apple Silicon
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._params = params or {}
        self._model: Any | None = None
        self._fitted = False
        self._feature_names: list[str] = []
        self._continuous_cols: list[str] = []
        self._cat_cols_present: list[str] = []

    @property
    def name(self) -> str:
        return "neural"

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        sample_weight: npt.NDArray[Any] | None = None,
    ) -> TabularNeuralModel:
        if not PYTORCH_TABULAR_AVAILABLE:
            raise ImportError(
                "pytorch-tabular is required for TabularNeuralModel. "
                "Install with: pip install pytorch-tabular"
            )

        _ = sample_weight  # Not supported by pytorch_tabular FT-Transformer

        self._feature_names = list(X_train.columns)

        # Split continuous vs categorical columns
        self._cat_cols_present = [c for c in CATEGORICAL_COLS if c in X_train.columns]
        self._continuous_cols = [c for c in X_train.columns if c not in self._cat_cols_present]

        # Prepare target column
        target_col = "_target"
        train_df = X_train.copy()
        train_df[target_col] = y_train.values

        val_df: pd.DataFrame | None = None
        if X_val is not None and y_val is not None:
            val_df = X_val.copy()
            val_df[target_col] = y_val.values

        data_config = DataConfig(
            target=[target_col],
            continuous_cols=self._continuous_cols,
            categorical_cols=self._cat_cols_present,
        )

        model_config = FTTransformerConfig(
            task="regression",
            num_attn_blocks=self._params.get("num_attn_blocks", 2),
            num_heads=self._params.get("num_heads", 4),
            d_model=self._params.get("d_model", 64),
            attn_dropout=self._params.get("attn_dropout", 0.1),
            add_norm_dropout=self._params.get("add_norm_dropout", 0.1),
            ff_dropout=self._params.get("ff_dropout", 0.1),
        )

        trainer_config = TrainerConfig(
            auto_lr_find=False,
            batch_size=self._params.get("batch_size", 512),
            max_epochs=self._params.get("max_epochs", 50),
            early_stopping="valid_loss",
            early_stopping_patience=self._params.get("patience", 10),
            accelerator="cpu",
            checkpoints_path="data/models/neural_checkpoints",
        )

        optimizer_config = OptimizerConfig()

        self._model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )

        self._model.fit(
            train=train_df,
            validation=val_df,
        )

        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> npt.NDArray[Any]:
        if not self._fitted or self._model is None:
            raise RuntimeError("Model must be fitted before predicting")

        result = self._model.predict(X)
        # pytorch_tabular returns a DataFrame with prediction column
        pred_col = [
            c for c in result.columns if "prediction" in c.lower() or c == "_target_prediction"
        ]
        if pred_col:
            return np.asarray(result[pred_col[0]].to_numpy())
        return np.asarray(result.iloc[:, -1].to_numpy())

    def get_feature_importance(self) -> dict[str, float]:
        # FT-Transformer doesn't provide traditional feature importance
        return {}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self._model,
                "params": self._params,
                "feature_names": self._feature_names,
                "continuous_cols": self._continuous_cols,
                "cat_cols_present": self._cat_cols_present,
                "fitted": self._fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> TabularNeuralModel:
        data = joblib.load(path)
        model = cls(params=data["params"])
        model._model = data["model"]
        model._feature_names = data["feature_names"]
        model._continuous_cols = data["continuous_cols"]
        model._cat_cols_present = data["cat_cols_present"]
        model._fitted = data["fitted"]
        return model

    def get_params(self) -> dict[str, Any]:
        return {"params": self._params}
