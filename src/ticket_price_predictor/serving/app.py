"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ticket_price_predictor.serving.routes import router


def create_app(static_dir: Path) -> FastAPI:
    """Build the FastAPI app, mount the JSON API, and serve static assets at root.

    `docs_url`/`openapi_url` are namespaced under `/api` so the static `/` mount
    can serve `index.html` without colliding with the auto OpenAPI UI.
    """
    app = FastAPI(
        title="Ticket Price Predictor",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.include_router(router, prefix="/api")
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app
