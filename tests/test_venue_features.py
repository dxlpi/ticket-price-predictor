"""Tests for venue feature extraction."""

import pandas as pd
import pytest

from ticket_price_predictor.ml.features.venue import VenueFeatureExtractor, VenueStatsCache


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "venue_name": ["MSG", "MSG", "MSG", "Staples", "Staples"],
            "listing_price": [200.0, 250.0, 300.0, 100.0, 150.0],
        }
    )


class TestVenueStatsCache:
    def test_fit_computes_stats(self, sample_df):
        cache = VenueStatsCache()
        cache.fit(sample_df)
        assert cache.is_fitted
        assert cache.is_known_venue("MSG")
        assert cache.is_known_venue("Staples")

    def test_unknown_venue_returns_global(self, sample_df):
        cache = VenueStatsCache()
        cache.fit(sample_df)
        stats = cache.get_stats("Unknown Venue")
        assert stats.listing_count == 0  # unknown
        assert stats.avg_price > 0  # global fallback

    def test_unfitted_cache_returns_defaults(self):
        cache = VenueStatsCache()
        stats = cache.get_stats("MSG")
        assert stats.listing_count == 0
        assert stats.avg_price == 150.0

    def test_bayesian_smoothing_pulls_toward_global(self, sample_df):
        cache = VenueStatsCache()
        cache.fit(sample_df)
        msg_stats = cache.get_stats("MSG")
        # Raw MSG avg = (200+250+300)/3 = 250.0
        # Global avg = (200+250+300+100+150)/5 = 200.0
        # With SMOOTHING_FACTOR=200 and n=3:
        # smoothed = (3*250 + 200*200) / (3+200) = 40750/203 ≈ 200.74
        # Smoothed value must be less than raw group average (pulled toward global)
        raw_msg_avg = 250.0
        assert msg_stats.avg_price < raw_msg_avg

    def test_bayesian_smoothing_between_group_and_global(self, sample_df):
        cache = VenueStatsCache()
        cache.fit(sample_df)
        msg_stats = cache.get_stats("MSG")
        global_avg = sample_df["listing_price"].mean()  # 200.0
        raw_msg_avg = 250.0
        # Smoothed average must be between global and raw group average
        assert global_avg <= msg_stats.avg_price <= raw_msg_avg

    def test_is_known_venue_case_insensitive(self, sample_df):
        cache = VenueStatsCache()
        cache.fit(sample_df)
        assert cache.is_known_venue("msg")
        assert cache.is_known_venue("MSG")
        assert cache.is_known_venue("  MSG  ")

    def test_fit_without_required_columns(self):
        cache = VenueStatsCache()
        df = pd.DataFrame({"other": [1, 2, 3]})
        cache.fit(df)
        assert cache.is_fitted
        # No venues should be known
        assert not cache.is_known_venue("anything")

    def test_venue_listing_count(self, sample_df):
        cache = VenueStatsCache()
        cache.fit(sample_df)
        msg_stats = cache.get_stats("MSG")
        assert msg_stats.listing_count == 3
        staples_stats = cache.get_stats("Staples")
        assert staples_stats.listing_count == 2

    def test_smoothing_factor_is_high(self):
        # Ensure smoothing factor is aggressive enough to prevent overfitting
        assert VenueStatsCache.SMOOTHING_FACTOR >= 50


class TestVenueFeatureExtractor:
    def test_feature_names(self):
        extractor = VenueFeatureExtractor()
        assert len(extractor.feature_names) == 4
        assert "venue_avg_price" in extractor.feature_names
        assert "venue_median_price" in extractor.feature_names
        assert "venue_price_std" in extractor.feature_names
        assert "is_known_venue" in extractor.feature_names

    def test_no_noisy_features(self):
        extractor = VenueFeatureExtractor()
        # venue_listing_count is noisy (raw count) and excluded
        assert "venue_listing_count" not in extractor.feature_names
        # venue_price_std is Bayesian-smoothed and intentionally included
        assert "venue_price_std" in extractor.feature_names

    def test_extract_returns_all_feature_columns(self, sample_df):
        extractor = VenueFeatureExtractor()
        extractor.fit(sample_df)
        result = extractor.extract(sample_df)
        assert set(result.columns) == set(extractor.feature_names)
        assert len(result) == len(sample_df)

    def test_extract_known_venues_flagged(self, sample_df):
        extractor = VenueFeatureExtractor()
        extractor.fit(sample_df)
        result = extractor.extract(sample_df)
        assert (result["is_known_venue"] == 1.0).all()

    def test_extract_unknown_venue_flagged_zero(self, sample_df):
        extractor = VenueFeatureExtractor()
        extractor.fit(sample_df)
        unknown_df = pd.DataFrame(
            {
                "venue_name": ["Madison Square Garden"],
                "listing_price": [300.0],
            }
        )
        result = extractor.extract(unknown_df)
        assert result["is_known_venue"].iloc[0] == 0.0

    def test_missing_venue_column_returns_zeros(self):
        extractor = VenueFeatureExtractor()
        df = pd.DataFrame({"other": [1, 2]})
        result = extractor.extract(df)
        assert (result["venue_avg_price"] == 0.0).all()
        assert (result["is_known_venue"] == 0.0).all()

    def test_get_params_returns_fitted_status(self, sample_df):
        extractor = VenueFeatureExtractor()
        assert extractor.get_params()["fitted"] is False
        extractor.fit(sample_df)
        assert extractor.get_params()["fitted"] is True

    def test_venue_avg_price_is_smoothed(self, sample_df):
        extractor = VenueFeatureExtractor()
        extractor.fit(sample_df)
        result = extractor.extract(sample_df)
        # MSG rows should all have the same smoothed avg price
        msg_rows = result.iloc[:3]
        assert msg_rows["venue_avg_price"].nunique() == 1
        # Smoothed avg should be less than raw MSG avg of 250
        assert msg_rows["venue_avg_price"].iloc[0] < 250.0
