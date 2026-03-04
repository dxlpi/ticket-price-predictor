"""Tests for interaction feature extraction."""

import numpy as np
import pandas as pd
import pytest

from ticket_price_predictor.ml.features.interactions import InteractionFeatureExtractor


@pytest.fixture
def extractor():
    return InteractionFeatureExtractor()


@pytest.fixture
def base_features_df():
    """Simulate concatenated base feature output."""
    return pd.DataFrame(
        {
            "artist_avg_price": [100.0, 200.0, 150.0],
            "zone_price_ratio": [0.8, 1.2, 0.5],
            "popularity_score": [0.7, 0.3, 0.9],
            "urgency_bucket": [1, 3, 2],
            "is_known_artist": [1, 1, 0],
            "city_tier": [1, 2, 3],
            "days_to_event": [5, 30, 60],
        }
    )


class TestInteractionFeatureExtractor:
    def test_feature_names(self, extractor):
        assert len(extractor.feature_names) == 7
        assert "artist_zone_price" in extractor.feature_names
        assert "days_to_event_log" in extractor.feature_names
        assert "popularity_zone" in extractor.feature_names
        assert "urgency_zone" in extractor.feature_names
        assert "artist_city_tier" in extractor.feature_names
        assert "price_per_urgency" in extractor.feature_names
        assert "artist_venue_price" in extractor.feature_names

    def test_extract_returns_all_columns(self, extractor, base_features_df):
        result = extractor.extract(base_features_df)
        assert set(result.columns) == set(extractor.feature_names)
        assert len(result) == 3

    def test_artist_zone_price(self, extractor, base_features_df):
        result = extractor.extract(base_features_df)
        # artist_zone_price = artist_avg_price * zone_price_ratio
        assert result["artist_zone_price"].iloc[0] == pytest.approx(80.0)  # 100.0 * 0.8
        assert result["artist_zone_price"].iloc[1] == pytest.approx(240.0)  # 200.0 * 1.2
        assert result["artist_zone_price"].iloc[2] == pytest.approx(75.0)  # 150.0 * 0.5

    def test_popularity_zone(self, extractor, base_features_df):
        result = extractor.extract(base_features_df)
        # popularity_zone = popularity_score * zone_price_ratio
        assert result["popularity_zone"].iloc[0] == pytest.approx(0.7 * 0.8)
        assert result["popularity_zone"].iloc[1] == pytest.approx(0.3 * 1.2)

    def test_urgency_zone(self, extractor, base_features_df):
        result = extractor.extract(base_features_df)
        # urgency_zone = urgency_bucket * zone_price_ratio
        assert result["urgency_zone"].iloc[0] == pytest.approx(1 * 0.8)
        assert result["urgency_zone"].iloc[1] == pytest.approx(3 * 1.2)

    def test_artist_city_tier(self, extractor, base_features_df):
        result = extractor.extract(base_features_df)
        # artist_city_tier = artist_avg_price * city_tier (preserves city signal for all artists)
        assert result["artist_city_tier"].iloc[0] == pytest.approx(100.0 * 1)
        assert result["artist_city_tier"].iloc[1] == pytest.approx(200.0 * 2)
        assert result["artist_city_tier"].iloc[2] == pytest.approx(
            150.0 * 3
        )  # unknown artist still gets city tier

    def test_days_to_event_log(self, extractor, base_features_df):
        result = extractor.extract(base_features_df)
        assert result["days_to_event_log"].iloc[0] == pytest.approx(np.log1p(5))
        assert result["days_to_event_log"].iloc[1] == pytest.approx(np.log1p(30))
        assert result["days_to_event_log"].iloc[2] == pytest.approx(np.log1p(60))

    def test_price_per_urgency(self, extractor, base_features_df):
        result = extractor.extract(base_features_df)
        # price_per_urgency = artist_avg_price / (days_to_event + 1)
        assert result["price_per_urgency"].iloc[0] == pytest.approx(100.0 / 6)
        assert result["price_per_urgency"].iloc[1] == pytest.approx(200.0 / 31)
        assert result["price_per_urgency"].iloc[2] == pytest.approx(150.0 / 61)

    def test_missing_features_use_defaults(self, extractor):
        df = pd.DataFrame({"unrelated": [1, 2, 3]})
        result = extractor.extract(df)
        assert len(result) == 3
        assert set(result.columns) == set(extractor.feature_names)
        # With defaults: artist_avg_price=0, zone_price_ratio=0.5
        assert (result["artist_zone_price"] == 0.0).all()

    def test_missing_days_uses_default_30(self, extractor):
        df = pd.DataFrame({"unrelated": [1, 2]})
        result = extractor.extract(df)
        # Default days_to_event=30
        assert result["days_to_event_log"].iloc[0] == pytest.approx(np.log1p(30))

    def test_get_params(self, extractor):
        params = extractor.get_params()
        assert isinstance(params, dict)

    def test_days_to_event_log_clips_negative(self, extractor):
        df = pd.DataFrame(
            {
                "days_to_event": [-5, 0, 10],
            }
        )
        result = extractor.extract(df)
        # Negative days clipped to 0 before log1p
        assert result["days_to_event_log"].iloc[0] == pytest.approx(np.log1p(0))
        assert result["days_to_event_log"].iloc[1] == pytest.approx(np.log1p(0))
        assert result["days_to_event_log"].iloc[2] == pytest.approx(np.log1p(10))

    def test_artist_city_tier_nonzero_for_unknown_artist(self, extractor, base_features_df):
        """City tier signal is preserved even for unknown artists."""
        result = extractor.extract(base_features_df)
        # Row 2 has is_known_artist=0 but should still have nonzero artist_city_tier
        assert result["artist_city_tier"].iloc[2] != 0.0

    def test_artist_venue_price_with_venue_median(self):
        """artist_venue_price = artist_avg * venue_median / global_median."""
        extractor = InteractionFeatureExtractor()
        train_df = pd.DataFrame(
            {
                "artist_avg_price": [100.0, 200.0],
                "venue_median_price": [150.0, 300.0],
            }
        )
        extractor.fit(train_df)
        assert extractor._global_median == pytest.approx(225.0)  # median of [150, 300]

        result = extractor.extract(train_df)
        expected_0 = 100.0 * 150.0 / 225.0
        expected_1 = 200.0 * 300.0 / 225.0
        assert result["artist_venue_price"].iloc[0] == pytest.approx(expected_0)
        assert result["artist_venue_price"].iloc[1] == pytest.approx(expected_1)

    def test_artist_venue_price_defaults_to_zero_when_venue_missing(self, extractor):
        """When venue_median_price is absent, artist_venue_price = 0."""
        df = pd.DataFrame({"artist_avg_price": [100.0, 200.0]})
        result = extractor.extract(df)
        # venue_median_price missing → defaults to 0.0 → product is 0
        assert (result["artist_venue_price"] == 0.0).all()

    def test_fit_uses_defensive_default_when_venue_absent(self):
        """fit() with no venue_median_price keeps 150.0 defensive default."""
        extractor = InteractionFeatureExtractor()
        extractor.fit(pd.DataFrame({"other_col": [1, 2, 3]}))
        assert extractor._global_median == pytest.approx(150.0)
