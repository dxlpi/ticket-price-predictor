"""Tests for listing context feature extraction."""

import pandas as pd
import pytest

from ticket_price_predictor.ml.features.listing import ListingContextFeatureExtractor


@pytest.fixture
def extractor():
    return ListingContextFeatureExtractor()


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "source": ["VividSeats", "StubHub", "VividSeats"],
            "quantity": [1, 2, 4],
        }
    )


class TestListingContextFeatureExtractor:
    def test_feature_names(self, extractor):
        assert len(extractor.feature_names) == 4
        assert "source_encoded" in extractor.feature_names
        assert "quantity" in extractor.feature_names
        assert "is_single_ticket" in extractor.feature_names
        assert "is_pair" in extractor.feature_names

    def test_no_timestamp_features(self, extractor):
        assert "listing_hour" not in extractor.feature_names
        assert "listing_day_of_week" not in extractor.feature_names

    def test_extract_with_all_columns(self, extractor, sample_df):
        result = extractor.extract(sample_df)
        assert len(result) == 3
        assert list(result.columns) == extractor.feature_names

    def test_source_encoding(self, extractor, sample_df):
        result = extractor.extract(sample_df)
        assert result["source_encoded"].iloc[0] == 0  # vividseats
        assert result["source_encoded"].iloc[1] == 1  # stubhub

    def test_quantity_features(self, extractor, sample_df):
        result = extractor.extract(sample_df)
        assert result["is_single_ticket"].iloc[0] == 1
        assert result["is_pair"].iloc[1] == 1
        assert result["is_pair"].iloc[2] == 0

    def test_missing_columns_defaults(self, extractor):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        result = extractor.extract(df)
        assert len(result) == 3
        assert (result["source_encoded"] == 0).all()
        assert (result["quantity"] == 1).all()

    def test_missing_columns_single_ticket_default(self, extractor):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        result = extractor.extract(df)
        # Default quantity=1 means all are single tickets
        assert (result["is_single_ticket"] == 1).all()
        assert (result["is_pair"] == 0).all()

    def test_unknown_source_encodes_to_zero(self, extractor):
        df = pd.DataFrame({"source": ["Ticketmaster", "SeatGeek"]})
        result = extractor.extract(df)
        assert (result["source_encoded"] == 0).all()

    def test_source_case_insensitive(self, extractor):
        df = pd.DataFrame({"source": ["STUBHUB", "stubhub", "StubHub"]})
        result = extractor.extract(df)
        assert (result["source_encoded"] == 1).all()
