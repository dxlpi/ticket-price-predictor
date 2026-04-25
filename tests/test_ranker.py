"""Tests for ListingRanker."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from ticket_price_predictor.ml.inference.ranker import ListingRanker
from ticket_price_predictor.ml.schemas import PricePrediction, RankedListing


def _make_prediction(predicted_price: float, confidence: float = 0.8) -> PricePrediction:
    return PricePrediction(
        event_id="evt_001",
        seat_zone="lower_tier",
        target_days_to_event=14,
        predicted_price=predicted_price,
        price_lower_bound=predicted_price * 0.8,
        price_upper_bound=predicted_price * 1.2,
        confidence_score=confidence,
        predicted_direction="STABLE",
        direction_probability=0.5,
        model_version="test",
    )


def _make_listing(listing_price: float, listing_id: str = "L001") -> dict:
    return {
        "listing_id": listing_id,
        "event_id": "evt_001",
        "listing_price": listing_price,
        "section": "Lower Level 100",
        "row": "5",
        "artist_or_team": "Test Artist",
        "venue_name": "Test Venue",
        "city": "Los Angeles",
        "event_datetime": datetime(2025, 6, 1, 20, 0, tzinfo=UTC),
        "days_to_event": 14,
        "event_type": "CONCERT",
    }


class TestListingRanker:
    def test_rank_listings_orders_by_value_score(self):
        """Listings with higher predicted/actual ratio rank higher."""
        predictor = MagicMock()
        # Listing A: predicted=$100, actual=$50 → value_score=2.0 (great deal)
        # Listing B: predicted=$100, actual=$200 → value_score=0.5 (overpriced)
        # Listing C: predicted=$100, actual=$100 → value_score=1.0
        predictor.predict.return_value = _make_prediction(100.0)

        ranker = ListingRanker(predictor)
        listings = [
            _make_listing(200.0, "overpriced"),  # value_score=0.5
            _make_listing(100.0, "fair"),  # value_score=1.0
            _make_listing(50.0, "underpriced"),  # value_score=2.0
        ]

        results = ranker.rank_listings(listings)

        assert len(results) == 3
        assert results[0].listing_id == "underpriced"
        assert results[0].rank == 1
        assert results[1].listing_id == "fair"
        assert results[1].rank == 2
        assert results[2].listing_id == "overpriced"
        assert results[2].rank == 3

    def test_value_score_calculation(self):
        """value_score = predicted_fair_price / listing_price."""
        predictor = MagicMock()
        predictor.predict.return_value = _make_prediction(120.0)

        ranker = ListingRanker(predictor)
        results = ranker.rank_listings([_make_listing(80.0)])

        assert len(results) == 1
        assert results[0].value_score == pytest.approx(120.0 / 80.0)

    def test_savings_estimate(self):
        """savings_estimate = predicted - actual (positive = savings)."""
        predictor = MagicMock()
        predictor.predict.return_value = _make_prediction(150.0)

        ranker = ListingRanker(predictor)
        results = ranker.rank_listings([_make_listing(100.0)])

        assert results[0].savings_estimate == pytest.approx(50.0)

    def test_negative_savings_when_overpriced(self):
        """savings_estimate is negative when listing price exceeds fair value."""
        predictor = MagicMock()
        predictor.predict.return_value = _make_prediction(80.0)

        ranker = ListingRanker(predictor)
        results = ranker.rank_listings([_make_listing(120.0)])

        assert results[0].savings_estimate == pytest.approx(-40.0)

    def test_zero_price_excluded(self):
        """Listings with zero or negative price are excluded."""
        predictor = MagicMock()
        predictor.predict.return_value = _make_prediction(100.0)

        ranker = ListingRanker(predictor)
        results = ranker.rank_listings([_make_listing(0.0)])

        assert len(results) == 0

    def test_negative_price_excluded(self):
        """Listings with negative price are excluded."""
        predictor = MagicMock()
        predictor.predict.return_value = _make_prediction(100.0)

        ranker = ListingRanker(predictor)
        listing = _make_listing(100.0)
        listing["listing_price"] = -50.0
        results = ranker.rank_listings([listing])

        assert len(results) == 0

    def test_empty_listings(self):
        """Empty input returns empty list."""
        predictor = MagicMock()
        ranker = ListingRanker(predictor)
        results = ranker.rank_listings([])
        assert results == []

    def test_ranked_listing_fields(self):
        """RankedListing has all required fields populated correctly."""
        predictor = MagicMock()
        predictor.predict.return_value = _make_prediction(100.0, confidence=0.9)

        ranker = ListingRanker(predictor)
        results = ranker.rank_listings([_make_listing(80.0, "L999")])

        r = results[0]
        assert isinstance(r, RankedListing)
        assert r.listing_id == "L999"
        assert r.listing_price == 80.0
        assert r.predicted_fair_price == 100.0
        assert r.confidence_score == pytest.approx(0.9)
        assert r.rank == 1

    def test_single_listing_rank_is_one(self):
        """A single listing always gets rank 1."""
        predictor = MagicMock()
        predictor.predict.return_value = _make_prediction(200.0)

        ranker = ListingRanker(predictor)
        results = ranker.rank_listings([_make_listing(500.0)])

        assert results[0].rank == 1

    def test_dataframe_input(self):
        """DataFrame input is handled the same as a list of dicts."""
        import pandas as pd

        predictor = MagicMock()
        predictor.predict.return_value = _make_prediction(100.0)

        ranker = ListingRanker(predictor)
        df = pd.DataFrame(
            [
                _make_listing(50.0, "L1"),
                _make_listing(200.0, "L2"),
            ]
        )

        results = ranker.rank_listings(df)

        assert len(results) == 2
        # L1 (value_score=2.0) should rank before L2 (value_score=0.5)
        assert results[0].listing_id == "L1"
        assert results[1].listing_id == "L2"

    def test_predictor_called_per_listing(self):
        """predict() is called once per valid listing."""
        predictor = MagicMock()
        predictor.predict.return_value = _make_prediction(100.0)

        ranker = ListingRanker(predictor)
        listings = [
            _make_listing(50.0, "L1"),
            _make_listing(100.0, "L2"),
            _make_listing(0.0, "L3"),  # excluded — no predict call
        ]
        ranker.rank_listings(listings)

        assert predictor.predict.call_count == 2

    def test_equal_value_scores_all_ranked(self):
        """Listings with identical value scores all receive a rank."""
        predictor = MagicMock()
        predictor.predict.return_value = _make_prediction(100.0)

        ranker = ListingRanker(predictor)
        listings = [_make_listing(100.0, f"L{i}") for i in range(4)]
        results = ranker.rank_listings(listings)

        assert len(results) == 4
        ranks = [r.rank for r in results]
        assert sorted(ranks) == [1, 2, 3, 4]
