"""Listing value ranker for commerce recommendation.

Ranks ticket listings by predicted value score (predicted fair price / actual price),
surfacing underpriced listings first. Analogous to product ranking by conversion value
in commerce recommendation systems.
"""

import warnings
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from ticket_price_predictor.ml.inference.predictor import PricePredictor, UnknownEventError
from ticket_price_predictor.ml.schemas import RankedListing


class ListingRanker:
    """Ranks ticket listings by predicted value relative to asking price.

    Wraps PricePredictor to compute a value_score for each listing:
        value_score = predicted_fair_price / listing_price

    A score > 1.0 indicates the listing is priced below the model's fair value
    estimate — analogous to a high-conversion-probability product in commerce
    recommendation (underpriced items convert faster).

    Known limitation: predict() is called per-listing (not batch-vectorized).
    For large listing sets (50+ items), this is slow due to full feature pipeline
    per call. TODO: batch feature extraction for production use.
    """

    def __init__(self, predictor: PricePredictor) -> None:
        """Initialize ranker.

        Args:
            predictor: Fitted PricePredictor instance
        """
        self._predictor = predictor

    def rank_listings(
        self,
        listings: list[dict[str, Any]] | pd.DataFrame,
        event_id: str | None = None,
    ) -> list[RankedListing]:
        """Rank listings by value score (predicted fair price / actual price).

        Args:
            listings: List of listing dicts or DataFrame. Each must contain:
                - listing_price (float): actual asking price
                - section (str): seat section
                - row (str): seat row
                - artist_or_team (str): artist/team name
                - venue_name (str): venue
                - city (str): city
                - event_datetime (datetime): event date/time
                - days_to_event (int): days until event
                Optional:
                - listing_id (str): marketplace listing ID
                - event_id (str): event identifier
                - event_type (str): event type
            event_id: Default event_id if not in listing dicts

        Returns:
            Listings ranked by value_score descending (rank 1 = best value)
        """
        rows = listings.to_dict("records") if isinstance(listings, pd.DataFrame) else listings

        # Collect raw scored data before building RankedListing (rank unknown until sorted)
        pending: list[tuple[float, dict[str, Any]]] = []

        for row in rows:
            listing_price = float(row.get("listing_price", 0))
            if listing_price <= 0:
                continue

            try:
                pred = self._predictor.predict(
                    event_id=str(row.get("event_id", event_id or "unknown")),
                    artist_or_team=str(row.get("artist_or_team", "Unknown")),
                    venue_name=str(row.get("venue_name", "Unknown")),
                    city=str(row.get("city", "Unknown")),
                    event_datetime=row.get("event_datetime", datetime.now(UTC)),
                    section=str(row.get("section", "Upper Level")),
                    row=str(row.get("row", "10")),
                    days_to_event=int(row.get("days_to_event", 14)),
                    event_type=str(row.get("event_type", "CONCERT")),
                    quantity=int(row.get("quantity", 2)),
                )

                value_score = pred.predicted_price / listing_price
                pending.append(
                    (
                        value_score,
                        {
                            "event_id": pred.event_id,
                            "listing_id": str(row["listing_id"]) if row.get("listing_id") else None,
                            "section": str(row.get("section", "")),
                            "row": str(row.get("row", "")),
                            "listing_price": listing_price,
                            "predicted_fair_price": pred.predicted_price,
                            "value_score": value_score,
                            "savings_estimate": pred.predicted_price - listing_price,
                            "confidence_score": pred.confidence_score,
                        },
                    )
                )
            except UnknownEventError:
                warnings.warn(
                    f"Listing skipped — unknown event_id: {row.get('event_id')}",
                    UserWarning,
                    stacklevel=2,
                )
                continue

        # Sort descending by value_score, then build RankedListing with correct rank
        pending.sort(key=lambda x: x[0], reverse=True)
        return [RankedListing(**data, rank=rank) for rank, (_, data) in enumerate(pending, start=1)]
