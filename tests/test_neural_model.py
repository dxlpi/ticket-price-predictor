"""Tests for TabularNeuralModel (FT-Transformer with entity embeddings)."""

from __future__ import annotations

import pytest

# Skip entire module if pytorch_tabular is not installed
pytest.importorskip("pytorch_tabular", reason="pytorch-tabular not installed")


def test_neural_model_imports():
    from ticket_price_predictor.ml.models.neural import (
        PYTORCH_TABULAR_AVAILABLE,
        TabularNeuralModel,
    )

    assert PYTORCH_TABULAR_AVAILABLE
    assert TabularNeuralModel is not None


def test_neural_model_skipif_unavailable():
    """This module's tests auto-skip if pytorch_tabular unavailable (via importorskip above)."""


def test_neural_model_structure():
    from ticket_price_predictor.ml.models.neural import TabularNeuralModel

    model = TabularNeuralModel()
    assert model.name == "neural"
    assert not model.is_fitted
    assert model.get_feature_importance() == {}


def test_neural_model_custom_params():
    from ticket_price_predictor.ml.models.neural import TabularNeuralModel

    params = {"num_attn_blocks": 3, "d_model": 128, "batch_size": 256}
    model = TabularNeuralModel(params=params)
    assert model.get_params() == {"params": params}
    assert not model.is_fitted


def test_neural_model_predict_before_fit_raises():
    import pandas as pd

    from ticket_price_predictor.ml.models.neural import TabularNeuralModel

    model = TabularNeuralModel()
    with pytest.raises(RuntimeError, match="Model must be fitted before predicting"):
        model.predict(pd.DataFrame({"a": [1, 2]}))


def test_categorical_cols_defined():
    from ticket_price_predictor.ml.models.neural import CATEGORICAL_COLS, EMBEDDING_DIMS

    assert "artist_or_team" in CATEGORICAL_COLS
    assert "venue_name" in CATEGORICAL_COLS
    assert "city" in CATEGORICAL_COLS
    assert "day_of_week_str" in CATEGORICAL_COLS
    # Every categorical has an embedding dim
    for col in CATEGORICAL_COLS:
        assert col in EMBEDDING_DIMS
