"""Tests for the inference-time event gate in PricePredictor."""

from __future__ import annotations

import json
import warnings
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from ticket_price_predictor.ml.features.pipeline import FeaturePipeline
from ticket_price_predictor.ml.inference import PricePredictor, UnknownEventError
from ticket_price_predictor.ml.models.base import PriceModel
from ticket_price_predictor.ml.training.trainer import ModelTrainer


def _make_stub_pipeline() -> MagicMock:
    """Return a stub fitted FeaturePipeline whose transform() yields a 1-row float frame."""
    stub = MagicMock(spec=FeaturePipeline)
    stub.transform.return_value = pd.DataFrame([[0.0]], columns=["dummy"])
    return stub


def _make_stub_model(prediction: float = 100.0) -> MagicMock:
    """Return a stub PriceModel.predict() that returns a fixed numpy array."""
    stub = MagicMock(spec=PriceModel)
    stub.predict.return_value = np.array([prediction])
    return stub


def _predict_kwargs(event_id: str | int | np.str_) -> dict[str, object]:
    """Return the standard kwargs for a single predict() call."""
    return {
        "event_id": event_id,
        "artist_or_team": "Test Artist",
        "venue_name": "Test Venue",
        "city": "New York",
        "event_datetime": datetime(2024, 6, 1, tzinfo=UTC),
        "section": "Lower Level 100",
        "row": "10",
        "days_to_event": 14,
        "event_type": "CONCERT",
        "quantity": 2,
    }


def test_predict_raises_for_unknown_event_id() -> None:
    """Gate runs before feature extraction — unknown event_id raises UnknownEventError."""
    predictor = PricePredictor(
        model=_make_stub_model(),
        known_events={"e1"},
    )

    with pytest.raises(UnknownEventError):
        predictor.predict(**_predict_kwargs("e_new"))


def test_predict_passes_for_known_event_id() -> None:
    """Known event_id passes the gate and reaches model.predict()."""
    model = _make_stub_model(prediction=100.0)
    predictor = PricePredictor(
        model=model,
        known_events={"e1"},
        fitted_pipeline=_make_stub_pipeline(),
    )

    result = predictor.predict(**_predict_kwargs("e1"))

    assert result.predicted_price == pytest.approx(100.0)
    model.predict.assert_called_once()


def test_predict_coerces_non_str_event_id() -> None:
    """numpy scalar / int event_ids must be coerced via str() before gate check."""
    predictor = PricePredictor(
        model=_make_stub_model(),
        known_events={"e1", "1"},
        fitted_pipeline=_make_stub_pipeline(),
    )

    # numpy.str_ should pass through str() to match the set member "e1"
    predictor.predict(**_predict_kwargs(np.str_("e1")))

    # Plain int should be coerced to "1" and match
    predictor.predict(**_predict_kwargs(1))


def _write_artifacts_for_meta_test(tmp_path: Path, meta: dict[str, object] | None) -> Path:
    """Train a tiny baseline model in tmp_path, then optionally rewrite/remove _meta.json.

    Returns the model path (e.g. tmp_path / "baseline_v1.joblib").
    """
    np.random.seed(0)
    n = 200
    df = pd.DataFrame(
        {
            "artist_or_team": np.random.choice(["Artist A", "Artist B"], n),
            "event_type": ["CONCERT"] * n,
            "city": np.random.choice(["New York", "Los Angeles"], n),
            "event_datetime": pd.date_range("2024-01-01", periods=n, freq="6h"),
            "section": np.random.choice(["Floor", "Lower Level", "Upper Level"], n),
            "row": np.random.choice(["1", "5", "10"], n),
            "days_to_event": np.random.randint(1, 60, n),
            "listing_price": 100 + np.random.randn(n) * 30,
            "event_id": [f"e{i % 20}" for i in range(n)],
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="6h"),
        }
    )

    trainer = ModelTrainer(model_type="baseline", model_version="v1")
    trainer.train(df)
    model_path = trainer.save(tmp_path)

    meta_path = tmp_path / "baseline_v1_meta.json"
    if meta is None:
        if meta_path.exists():
            meta_path.unlink()
    else:
        meta_path.write_text(json.dumps(meta))

    return Path(model_path)


def test_from_path_legacy_meta_warns_and_disables_gate(tmp_path: Path) -> None:
    """Meta without 'known_events' triggers UserWarning and leaves gate disabled."""
    model_path = _write_artifacts_for_meta_test(
        tmp_path,
        meta={"log_transformed_cols": []},
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        predictor = PricePredictor.from_path(model_path, model_type="baseline")

    legacy_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(legacy_warnings) == 1
    assert predictor._known_events is None


def test_from_path_no_meta_file_warns_and_disables_gate(tmp_path: Path) -> None:
    """Absent _meta.json triggers UserWarning and leaves gate disabled."""
    model_path = _write_artifacts_for_meta_test(tmp_path, meta=None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        predictor = PricePredictor.from_path(model_path, model_type="baseline")

    legacy_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(legacy_warnings) == 1
    assert predictor._known_events is None


def test_from_path_loads_known_events(tmp_path: Path) -> None:
    """Meta with 'known_events' populates _known_events as a set."""
    model_path = _write_artifacts_for_meta_test(
        tmp_path,
        meta={"known_events": ["e1", "e2"]},
    )

    predictor = PricePredictor.from_path(model_path, model_type="baseline")

    assert predictor._known_events == {"e1", "e2"}
