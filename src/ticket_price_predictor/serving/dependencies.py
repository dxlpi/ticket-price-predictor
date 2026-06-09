"""FastAPI dependency providers wrapping process-wide singletons.

Singletons are populated once by `init_dependencies` (called from the entry-point
script before uvicorn starts) and read by route handlers via FastAPI `Depends`.
Tests bypass this module by setting `app.dependency_overrides` directly.
"""

from __future__ import annotations

from pathlib import Path

from ticket_price_predictor.ml.inference.predictor import PricePredictor
from ticket_price_predictor.storage.repository import EventRepository

_PREDICTOR: PricePredictor | None = None
_EVENT_REPO: EventRepository | None = None
_KNOWN_EVENTS: set[str] | None = None


def init_dependencies(model_path: Path, data_dir: Path) -> None:
    """Load the predictor and repository, then cache them as module singletons.

    Performs a cheap path-existence check before loading the model so a missing
    artifact fails fast without paying the load cost. Singletons are assigned
    only after all loads succeed, so a partial failure leaves the module in its
    pre-init state and the next call can retry cleanly.

    Args:
        model_path: Path to the saved predictor (e.g. lightgbm_v36.joblib).
        data_dir: Base data directory for `EventRepository`.

    Raises:
        FileNotFoundError: When `model_path` does not exist.
        RuntimeError: When the loaded model has no `known_events` set
            (legacy artifact — scope cannot be guaranteed).
    """
    global _PREDICTOR, _EVENT_REPO, _KNOWN_EVENTS

    if not model_path.exists():
        raise FileNotFoundError(f"model path does not exist: {model_path}")

    predictor = PricePredictor.from_path(model_path)
    event_repo = EventRepository(data_dir)
    known_events = predictor.known_events
    if known_events is None:
        raise RuntimeError("model has no known_events; cannot guarantee scope")

    _PREDICTOR = predictor
    _EVENT_REPO = event_repo
    _KNOWN_EVENTS = known_events


def get_predictor() -> PricePredictor:
    """Return the loaded predictor singleton.

    Raises:
        RuntimeError: When `init_dependencies` has not been called.
    """
    if _PREDICTOR is None:
        raise RuntimeError("dependencies not initialized")
    return _PREDICTOR


def get_event_repo() -> EventRepository:
    """Return the event repository singleton.

    Raises:
        RuntimeError: When `init_dependencies` has not been called.
    """
    if _EVENT_REPO is None:
        raise RuntimeError("dependencies not initialized")
    return _EVENT_REPO


def get_known_events() -> set[str]:
    """Return the set of known event_ids cached at init time.

    Raises:
        RuntimeError: When `init_dependencies` has not been called.
    """
    if _KNOWN_EVENTS is None:
        raise RuntimeError("dependencies not initialized")
    return _KNOWN_EVENTS
