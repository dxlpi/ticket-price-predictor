"""Startup-wiring tests for the serving layer (AC15).

Covers `create_app` route + static-mount wiring, fail-fast behaviour of
`init_dependencies` when the model artifact is missing, and the uninitialised
`get_predictor` guard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

from ticket_price_predictor.ml.inference import predictor as predictor_mod
from ticket_price_predictor.serving import dependencies as deps_mod
from ticket_price_predictor.serving.app import create_app


def _api_route_paths(app: FastAPI) -> set[str]:
    # Assert against the OpenAPI schema (stable public API) rather than walking
    # `app.routes`: newer FastAPI (>=0.138) represents an included router as a
    # single internal `_IncludedRouter` node instead of flattening child
    # `APIRoute`s into `app.routes`, so isinstance-based introspection is
    # version-fragile and silently returned an empty set under the CI resolve.
    return set(app.openapi()["paths"].keys())


def _has_static_root_mount(app: FastAPI) -> bool:
    # Starlette normalises `app.mount("/", ...)` to an empty `path`; accept both
    # so the test does not break if a future Starlette version changes that.
    for r in app.routes:
        if isinstance(r, Mount) and r.path in {"", "/"} and isinstance(r.app, StaticFiles):
            return True
    return False


def test_create_app_returns_fastapi_instance(tmp_path: Path) -> None:
    app = create_app(static_dir=tmp_path)
    assert isinstance(app, FastAPI)


def test_create_app_mounts_three_api_routes(tmp_path: Path) -> None:
    app = create_app(static_dir=tmp_path)
    paths = _api_route_paths(app)
    assert "/api/events/search" in paths
    assert "/api/events/{event_id}" in paths
    assert "/api/predict" in paths


def test_create_app_mounts_static_files_at_root(tmp_path: Path) -> None:
    app = create_app(static_dir=tmp_path)
    assert _has_static_root_mount(app)


def test_init_dependencies_fails_fast_without_loading_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def _spy(*args: Any, **kwargs: Any) -> None:
        called.append((args, kwargs))

    monkeypatch.setattr(predictor_mod.PricePredictor, "from_path", _spy)

    with pytest.raises(FileNotFoundError):
        deps_mod.init_dependencies(model_path=Path("/does/not/exist.joblib"), data_dir=tmp_path)

    assert called == []


def test_get_predictor_uninitialized_raises_runtimeerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(deps_mod, "_PREDICTOR", None)
    monkeypatch.setattr(deps_mod, "_EVENT_REPO", None)
    monkeypatch.setattr(deps_mod, "_KNOWN_EVENTS", None)

    with pytest.raises(RuntimeError, match="dependencies not initialized"):
        deps_mod.get_predictor()
