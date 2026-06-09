"""FastAPI serving layer for the price predictor web app."""

from ticket_price_predictor.serving.app import create_app

__all__ = ["create_app"]
