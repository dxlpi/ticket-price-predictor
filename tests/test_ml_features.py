"""Tests for ML feature extractors."""

import numpy as np
import pandas as pd
import pytest

from ticket_price_predictor.ml.features.artist_stats import ArtistStatsCache
from ticket_price_predictor.ml.features.event import EventFeatureExtractor
from ticket_price_predictor.ml.features.pipeline import FeaturePipeline
from ticket_price_predictor.ml.features.seating import SeatingFeatureExtractor
from ticket_price_predictor.ml.features.timeseries import (
    MomentumFeatureExtractor,
    TimeSeriesFeatureExtractor,
)

# ============================================================================
# ArtistStatsCache Tests
# ============================================================================


class TestArtistStatsCache:
    """Tests for ArtistStatsCache."""

    def test_fit_creates_stats(self):
        """Test fitting creates artist statistics."""
        df = pd.DataFrame(
            {
                "artist_or_team": ["Artist A", "Artist A", "Artist B"],
                "listing_price": [100.0, 150.0, 200.0],
                "event_id": ["e1", "e1", "e2"],
            }
        )

        cache = ArtistStatsCache()
        cache.fit(df)

        assert cache.is_fitted
        assert cache.artist_count == 2

    def test_get_stats_known_artist(self):
        """Test getting stats for known artist."""
        df = pd.DataFrame(
            {
                "artist_or_team": ["Taylor Swift", "Taylor Swift"],
                "listing_price": [200.0, 300.0],
                "event_id": ["e1", "e1"],
            }
        )

        cache = ArtistStatsCache()
        cache.fit(df)

        stats = cache.get_stats("Taylor Swift")
        assert stats.artist_name == "Taylor Swift"
        assert stats.avg_price == 250.0
        assert stats.median_price == 250.0
        assert stats.listing_count == 2

    def test_get_stats_unknown_artist(self):
        """Test getting stats for unknown artist returns global defaults."""
        df = pd.DataFrame(
            {
                "artist_or_team": ["Known Artist"],
                "listing_price": [100.0],
                "event_id": ["e1"],
            }
        )

        cache = ArtistStatsCache()
        cache.fit(df)

        stats = cache.get_stats("Unknown Artist")
        assert stats.artist_name == "Unknown Artist"
        assert stats.avg_price == 100.0  # Global average

    def test_get_stats_unfitted_returns_defaults(self):
        """Test unfitted cache returns sensible defaults."""
        cache = ArtistStatsCache()
        stats = cache.get_stats("Any Artist")

        assert stats.avg_price == 150.0
        assert stats.median_price == 150.0
        assert stats.price_std == 50.0

    def test_is_known_artist(self):
        """Test is_known_artist method."""
        df = pd.DataFrame(
            {
                "artist_or_team": ["Known"],
                "listing_price": [100.0],
                "event_id": ["e1"],
            }
        )

        cache = ArtistStatsCache()
        cache.fit(df)

        assert cache.is_known_artist("Known")
        assert cache.is_known_artist("known")  # Case insensitive
        assert not cache.is_known_artist("Unknown")

    def test_premium_ratio(self):
        """Test premium ratio calculation."""
        df = pd.DataFrame(
            {
                "artist_or_team": ["Artist"] * 4,
                "listing_price": [100.0, 150.0, 250.0, 300.0],  # 2 above $200
                "event_id": ["e1"] * 4,
            }
        )

        cache = ArtistStatsCache()
        cache.fit(df)

        stats = cache.get_stats("Artist")
        assert stats.premium_ratio == 0.5  # 2/4 above threshold

    def test_save_and_load(self, tmp_path):
        """Test saving and loading cache."""
        df = pd.DataFrame(
            {
                "artist_or_team": ["Artist A"],
                "listing_price": [100.0],
                "event_id": ["e1"],
            }
        )

        cache = ArtistStatsCache()
        cache.fit(df)

        save_path = tmp_path / "cache.joblib"
        cache.save(save_path)

        loaded = ArtistStatsCache.load(save_path)
        assert loaded.is_fitted
        assert loaded.artist_count == 1
        assert loaded.is_known_artist("Artist A")


# ============================================================================
# EventFeatureExtractor Tests
# ============================================================================


class TestEventFeatureExtractor:
    """Tests for EventFeatureExtractor."""

    def test_feature_names(self):
        """Test feature names are defined."""
        extractor = EventFeatureExtractor()
        assert "event_type_encoded" in extractor.feature_names
        assert "city_tier" in extractor.feature_names
        assert "day_of_week" in extractor.feature_names
        assert "is_weekend" in extractor.feature_names

    def test_extract_event_type(self):
        """Test event type encoding — map keys are lowercase."""
        extractor = EventFeatureExtractor()
        df = pd.DataFrame(
            {
                "event_type": ["concert", "sports", "theater"],
                "city": ["New York", "Chicago", "Miami"],
                "event_datetime": pd.to_datetime(["2024-06-15", "2024-06-16", "2024-12-25"]),
            }
        )

        result = extractor.extract(df)
        assert result["event_type_encoded"].tolist() == [0, 1, 2]

    def test_city_tier(self):
        """Test city tier assignment."""
        extractor = EventFeatureExtractor()
        df = pd.DataFrame(
            {
                "city": ["New York", "Seattle", "Small Town"],
                "event_type": ["CONCERT"] * 3,
                "event_datetime": pd.to_datetime(["2024-06-15"] * 3),
            }
        )

        result = extractor.extract(df)
        assert result["city_tier"].tolist() == [1, 2, 3]

    def test_datetime_features(self):
        """Test datetime feature extraction."""
        extractor = EventFeatureExtractor()
        df = pd.DataFrame(
            {
                "event_datetime": pd.to_datetime(
                    [
                        "2024-07-13",  # Saturday in July (summer)
                        "2024-12-25",  # Wednesday in December (holiday)
                        "2024-03-15",  # Friday in March
                    ]
                ),
                "city": ["NYC"] * 3,
                "event_type": ["CONCERT"] * 3,
            }
        )

        result = extractor.extract(df)

        # First event: Saturday, summer
        assert result["is_weekend"].iloc[0] == 1
        assert result["is_summer"].iloc[0] == 1
        assert result["is_holiday_season"].iloc[0] == 0

        # Second event: December (holiday season)
        assert result["is_holiday_season"].iloc[1] == 1

    def test_venue_capacity_bucket(self):
        """Test venue capacity bucketing."""
        extractor = EventFeatureExtractor()
        df = pd.DataFrame(
            {
                "venue_capacity": [3000, 10000, 25000, 60000, None],
                "city": ["NYC"] * 5,
                "event_type": ["CONCERT"] * 5,
                "event_datetime": pd.to_datetime(["2024-06-15"] * 5),
            }
        )

        result = extractor.extract(df)
        assert result["venue_capacity_bucket"].tolist() == [0, 1, 2, 3, 2]

    def test_missing_columns_use_defaults(self):
        """Test handling of missing columns."""
        extractor = EventFeatureExtractor()
        df = pd.DataFrame({"some_other_column": [1, 2, 3]})

        result = extractor.extract(df)
        assert len(result) == 3
        assert "event_type_encoded" in result.columns
        assert "market_saturation" in result.columns

    def test_market_saturation_feature(self):
        """Test market saturation counts concurrent city events."""
        extractor = EventFeatureExtractor()
        # Two events in NYC same week, one in Chicago
        train_df = pd.DataFrame(
            {
                "event_id": ["e1", "e2", "e3"],
                "city": ["New York", "New York", "Chicago"],
                "event_datetime": pd.to_datetime(["2024-07-10", "2024-07-12", "2024-07-11"]),
            }
        )
        extractor.fit(train_df)

        # NYC same week should have 2 events
        nyc_row = pd.DataFrame(
            {
                "event_id": ["e1"],
                "city": ["New York"],
                "event_datetime": pd.to_datetime(["2024-07-10"]),
                "event_type": ["CONCERT"],
            }
        )
        result = extractor.extract(nyc_row)
        assert result["market_saturation"].iloc[0] == pytest.approx(np.log1p(2))

        # Chicago same week has 1 event
        chi_row = pd.DataFrame(
            {
                "event_id": ["e3"],
                "city": ["Chicago"],
                "event_datetime": pd.to_datetime(["2024-07-11"]),
                "event_type": ["CONCERT"],
            }
        )
        result_chi = extractor.extract(chi_row)
        assert result_chi["market_saturation"].iloc[0] == pytest.approx(np.log1p(1))

    def test_market_saturation_different_years_same_week(self):
        """Year must be part of key to prevent cross-year collisions."""
        extractor = EventFeatureExtractor()
        train_df = pd.DataFrame(
            {
                "event_id": ["e1"],
                "city": ["New York"],
                "event_datetime": pd.to_datetime(["2023-07-10"]),
            }
        )
        extractor.fit(train_df)

        # 2024 same ISO week as 2023 event — should NOT inherit 2023 count
        test_df = pd.DataFrame(
            {
                "event_id": ["e2"],
                "city": ["New York"],
                "event_datetime": pd.to_datetime(["2024-07-10"]),
                "event_type": ["CONCERT"],
            }
        )
        result = extractor.extract(test_df)
        # 2024 week has no events in training (training only has 2023)
        assert result["market_saturation"].iloc[0] == pytest.approx(np.log1p(0))


# ============================================================================
# SeatingFeatureExtractor Tests
# ============================================================================


class TestSeatingFeatureExtractor:
    """Tests for SeatingFeatureExtractor."""

    def test_feature_names(self):
        """Test feature names are defined."""
        extractor = SeatingFeatureExtractor()
        assert "seat_zone_encoded" in extractor.feature_names
        assert "row_numeric" in extractor.feature_names
        assert "is_floor" in extractor.feature_names
        assert "is_ga" in extractor.feature_names

    def test_seat_zone_encoding(self):
        """Test seat zone encoding from section names."""
        extractor = SeatingFeatureExtractor()
        df = pd.DataFrame(
            {
                "section": ["Floor VIP", "Lower Level 100", "Upper Deck", "GA"],
            }
        )

        result = extractor.extract(df)
        assert "seat_zone_encoded" in result.columns

    def test_row_numeric(self):
        """Test row number extraction."""
        extractor = SeatingFeatureExtractor()
        df = pd.DataFrame(
            {
                "section": ["Section 100"] * 5,
                "row": ["1", "10", "AA", "GA", ""],
            }
        )

        result = extractor.extract(df)
        assert result["row_numeric"].iloc[0] == 1
        assert result["row_numeric"].iloc[1] == 10

    def test_floor_and_ga_detection(self):
        """Test floor and GA detection."""
        extractor = SeatingFeatureExtractor()
        df = pd.DataFrame(
            {
                "section": ["Floor A", "GA Pit", "Section 200", "VIP Floor"],
            }
        )

        result = extractor.extract(df)
        assert result["is_floor"].iloc[0] == 1
        assert result["is_ga"].iloc[1] == 1
        assert result["is_floor"].iloc[2] == 0

    def test_is_premium_detection(self):
        """Test premium section keyword detection."""
        extractor = SeatingFeatureExtractor()
        df = pd.DataFrame(
            {
                "section": [
                    "VIP BOX 1",  # → 1 (vip)
                    "PLATINUM SEATS",  # → 1 (platinum)
                    "Floor GA",  # → 0 (no premium keyword)
                    "General Admission",  # → 0 (excluded)
                    "PIT STANDING",  # → 1 (pit)
                    "Section 415 Row 22",  # → 0 (no keyword)
                    "Club Level 205",  # → 0 (no premium keyword)
                    "SUITE 12",  # → 1 (suite)
                    "FIELD A",  # → 1 (field)
                    "Courtside Row 1",  # → 1 (courtside)
                ],
            }
        )

        result = extractor.extract(df)
        expected = [1, 1, 0, 0, 1, 0, 0, 1, 1, 1]
        assert result["is_premium"].tolist() == expected

    def test_is_premium_in_feature_names(self):
        """Test that is_premium is in feature names."""
        extractor = SeatingFeatureExtractor()
        assert "is_premium" in extractor.feature_names
        assert len(extractor.feature_names) == 6


# ============================================================================
# TimeSeriesFeatureExtractor Tests
# ============================================================================


class TestTimeSeriesFeatureExtractor:
    """Tests for TimeSeriesFeatureExtractor."""

    def test_feature_names(self):
        """Test feature names are defined."""
        extractor = TimeSeriesFeatureExtractor()
        assert "days_to_event" in extractor.feature_names
        assert "days_to_event_squared" in extractor.feature_names
        assert "urgency_bucket" in extractor.feature_names
        assert "is_last_week" in extractor.feature_names

    def test_days_to_event_features(self):
        """Test days to event feature extraction."""
        extractor = TimeSeriesFeatureExtractor()
        df = pd.DataFrame({"days_to_event": [1, 7, 14, 30, 60]})

        result = extractor.extract(df)

        assert result["days_to_event"].tolist() == [1, 7, 14, 30, 60]
        assert result["days_to_event_squared"].iloc[0] == 1
        assert result["days_to_event_squared"].iloc[1] == 49

    def test_urgency_bucket(self):
        """Test urgency bucket assignment."""
        extractor = TimeSeriesFeatureExtractor()
        df = pd.DataFrame({"days_to_event": [0, 3, 10, 20, 45, 90]})

        result = extractor.extract(df)

        # 0 days = extreme urgency (5)
        assert result["urgency_bucket"].iloc[0] == 5
        # 3 days = high urgency (4)
        assert result["urgency_bucket"].iloc[1] == 4
        # 10 days = moderate (3)
        assert result["urgency_bucket"].iloc[2] == 3
        # 20 days = low (2)
        assert result["urgency_bucket"].iloc[3] == 2
        # 45 days = planning (1)
        assert result["urgency_bucket"].iloc[4] == 1
        # 90 days = early bird (0)
        assert result["urgency_bucket"].iloc[5] == 0

    def test_last_week_flag(self):
        """Test is_last_week flag."""
        extractor = TimeSeriesFeatureExtractor()
        df = pd.DataFrame({"days_to_event": [1, 7, 8, 14]})

        result = extractor.extract(df)
        assert result["is_last_week"].tolist() == [1, 1, 0, 0]


# ============================================================================
# MomentumFeatureExtractor Tests
# ============================================================================


class TestMomentumFeatureExtractor:
    """Tests for MomentumFeatureExtractor."""

    def test_feature_names(self):
        """Test feature names are defined."""
        extractor = MomentumFeatureExtractor()
        assert "price_momentum_7d" in extractor.feature_names
        assert "price_momentum_30d" in extractor.feature_names
        assert "price_vs_initial" in extractor.feature_names
        assert "price_volatility" in extractor.feature_names

    def test_extract_with_precomputed(self):
        """Test extraction with pre-computed columns."""
        extractor = MomentumFeatureExtractor()
        df = pd.DataFrame(
            {
                "price_momentum_7d": [0.05, -0.02, 0.10],
                "price_momentum_30d": [0.10, 0.05, 0.15],
                "price_vs_initial": [1.1, 0.95, 1.2],
                "price_volatility": [0.02, 0.05, 0.03],
            }
        )

        result = extractor.extract(df)

        assert result["price_momentum_7d"].tolist() == [0.05, -0.02, 0.10]
        assert result["price_vs_initial"].tolist() == [1.1, 0.95, 1.2]

    def test_extract_without_precomputed(self):
        """Test extraction fills defaults when columns missing."""
        extractor = MomentumFeatureExtractor()
        df = pd.DataFrame({"other_column": [1, 2, 3]})

        result = extractor.extract(df)

        # Should fill with default values
        assert len(result) == 3
        assert "price_momentum_7d" in result.columns


# ============================================================================
# FeaturePipeline Tests
# ============================================================================


class TestFeaturePipeline:
    """Tests for FeaturePipeline."""

    def test_initialization_with_momentum(self):
        """Test pipeline includes momentum features when enabled."""
        pipeline = FeaturePipeline(include_momentum=True)
        assert "price_momentum_7d" in pipeline.feature_names
        assert "price_momentum_30d" in pipeline.feature_names

    def test_initialization_without_momentum(self):
        """Test pipeline excludes momentum features when disabled."""
        pipeline = FeaturePipeline(include_momentum=False)
        assert "price_momentum_7d" not in pipeline.feature_names

    def test_fit_transform(self):
        """Test fit_transform produces all expected features."""
        pipeline = FeaturePipeline(include_momentum=False)

        df = pd.DataFrame(
            {
                "artist_or_team": ["Taylor Swift"],
                "event_type": ["CONCERT"],
                "city": ["New York"],
                "event_datetime": pd.to_datetime(["2024-07-15"]),
                "section": ["Floor VIP"],
                "row": ["1"],
                "days_to_event": [14],
                "listing_price": [500.0],
                "event_id": ["e1"],
            }
        )

        result = pipeline.fit_transform(df)

        # Check key features are present
        assert "artist_avg_price" in result.columns
        assert "city_tier" in result.columns
        assert "seat_zone_encoded" in result.columns
        assert "days_to_event" in result.columns

    def test_transform_after_fit(self):
        """Test transform works after fitting."""
        pipeline = FeaturePipeline(include_momentum=False)

        train_df = pd.DataFrame(
            {
                "artist_or_team": ["Artist A", "Artist B"],
                "event_type": ["CONCERT", "CONCERT"],
                "city": ["NYC", "LA"],
                "event_datetime": pd.to_datetime(["2024-07-15", "2024-07-16"]),
                "section": ["Floor", "Upper"],
                "row": ["1", "10"],
                "days_to_event": [14, 30],
                "listing_price": [100.0, 200.0],
                "event_id": ["e1", "e2"],
            }
        )

        test_df = pd.DataFrame(
            {
                "artist_or_team": ["Artist A"],
                "event_type": ["CONCERT"],
                "city": ["NYC"],
                "event_datetime": pd.to_datetime(["2024-08-01"]),
                "section": ["Floor"],
                "row": ["5"],
                "days_to_event": [7],
                "listing_price": [150.0],
                "event_id": ["e3"],
            }
        )

        pipeline.fit(train_df)
        result = pipeline.transform(test_df)

        assert len(result) == 1
        assert len(result.columns) == len(pipeline.feature_names)

    def test_feature_count(self):
        """Test expected feature count."""
        # No snapshot, no momentum
        pipeline_no_extras = FeaturePipeline(include_momentum=False, include_snapshot=False)
        # Default: snapshot=True, momentum=False
        pipeline_with_snapshot = FeaturePipeline(include_momentum=False, include_snapshot=True)
        # Momentum (no snapshot)
        pipeline_with_momentum = FeaturePipeline(include_momentum=True, include_snapshot=False)

        # No extras baseline (no snapshot, no momentum); +8 from ListingStructuralFeatureExtractor (v38)
        assert len(pipeline_no_extras.feature_names) == 104

        # With snapshot (default): baseline + 4 snapshot features
        assert len(pipeline_with_snapshot.feature_names) == 108

        # With momentum (no snapshot): baseline + 4 momentum features
        assert len(pipeline_with_momentum.feature_names) == 108

    def test_feature_count_without_new_extractors(self):
        """Test feature count when optional extractors are disabled."""
        pipeline = FeaturePipeline(
            include_momentum=False,
            include_snapshot=False,
            include_popularity=False,
            include_regional=False,
        )
        assert len(pipeline.feature_names) == 90


class TestEventPricingLOO:
    """Tests for LOO (Leave-One-Out) encoding in EventPricingFeatureExtractor."""

    def _make_extractor(self, train_df: pd.DataFrame):
        from ticket_price_predictor.ml.features.event_pricing import (
            EventPricingFeatureExtractor,
        )

        ext = EventPricingFeatureExtractor()
        ext.fit(train_df)
        return ext

    def _make_train_df(self) -> pd.DataFrame:
        """Three listings in event e1, one listing in event e2."""
        return pd.DataFrame(
            {
                "event_id": ["e1", "e1", "e1", "e2"],
                "artist_or_team": ["Artist A"] * 4,
                "listing_price": [100.0, 200.0, 300.0, 150.0],
                "section": ["Floor", "Floor", "Upper", "Floor"],
            }
        )

    def test_loo_passthrough_for_unseen_events(self):
        """Val/test rows (event_id not in training set) are unaffected by LOO.

        For rows whose event_id was never seen during fit(), the LOO branch
        cannot fire (the event_id is absent from _event_price_sums). The
        returned event_median_price must therefore equal the non-LOO path,
        which falls back to artist-level or global stats.
        """
        train_df = self._make_train_df()
        ext = self._make_extractor(train_df)

        # Unseen event — LOO cannot fire regardless of listing_price presence
        unseen_row = pd.DataFrame(
            {
                "event_id": ["e99"],
                "artist_or_team": ["Artist A"],
                "listing_price": [999.0],
                "section": ["Floor"],
            }
        )

        result_with_price = ext.extract(unseen_row)

        # Extract again without listing_price to get the "pure" non-LOO value
        unseen_no_price = unseen_row.drop(columns=["listing_price"])
        result_no_price = ext.extract(unseen_no_price)

        # Both paths must produce identical event_median_price
        assert result_with_price["event_median_price"].iloc[0] == pytest.approx(
            result_no_price["event_median_price"].iloc[0]
        )

    def test_loo_adjusts_training_rows(self):
        """LOO encoding produces different event_median_price for training rows with n > 1.

        When a row's event_id IS in the training set and n_group > 1, the LOO
        branch fires and removes the row's own price from the event mean.
        The result must differ from the naively-smoothed event median (non-LOO path),
        confirming the adjustment is non-trivial.
        """
        train_df = self._make_train_df()
        ext = self._make_extractor(train_df)

        # Extract with listing_price — LOO fires for e1 rows (n=3 > 1)
        result_loo = ext.extract(train_df)

        # Extract without listing_price — LOO branch skipped (pd.isna check fails)
        train_no_price = train_df.drop(columns=["listing_price"])
        result_no_loo = ext.extract(train_no_price)

        # e1 rows (index 0-2): LOO must differ from non-LOO
        for i in range(3):
            assert result_loo["event_median_price"].iloc[i] != pytest.approx(
                result_no_loo["event_median_price"].iloc[i]
            ), f"Row {i} in event e1 (n=3): LOO should differ from non-LOO"

        # e2 row (index 3, n=1): LOO falls back to global mean — may equal non-LOO
        # We don't assert equality here, just confirm it doesn't crash
        assert not pd.isna(result_loo["event_median_price"].iloc[3])


# ============================================================================
# EventFeatureExtractor EVENT_TYPE_MAP Tests
# ============================================================================


class TestEventTypeMapEncoding:
    """Tests for EventFeatureExtractor EVENT_TYPE_MAP encoding."""

    def _extractor(self):
        return EventFeatureExtractor()

    def _df(self, event_types: list[str]) -> "pd.DataFrame":
        return pd.DataFrame(
            {
                "event_type": event_types,
                "city": ["New York"] * len(event_types),
                "event_datetime": pd.to_datetime(["2024-07-15"] * len(event_types)),
            }
        )

    def test_concert_encodes_to_0(self):
        """'concert' maps to 0."""
        result = self._extractor().extract(self._df(["concert"]))
        assert result["event_type_encoded"].iloc[0] == 0

    def test_sports_encodes_to_1(self):
        """'sports' maps to 1."""
        result = self._extractor().extract(self._df(["sports"]))
        assert result["event_type_encoded"].iloc[0] == 1

    def test_theater_encodes_to_2(self):
        """'theater' maps to 2."""
        result = self._extractor().extract(self._df(["theater"]))
        assert result["event_type_encoded"].iloc[0] == 2

    def test_comedy_encodes_to_3(self):
        """'comedy' maps to 3."""
        result = self._extractor().extract(self._df(["comedy"]))
        assert result["event_type_encoded"].iloc[0] == 3

    def test_all_four_types_together(self):
        """All four event types encode to the correct integers."""
        result = self._extractor().extract(self._df(["concert", "sports", "theater", "comedy"]))
        assert result["event_type_encoded"].tolist() == [0, 1, 2, 3]

    def test_lowercase_input_matches(self):
        """Lowercase input values resolve correctly (map keys are lowercase)."""
        result = self._extractor().extract(self._df(["comedy", "concert"]))
        assert result["event_type_encoded"].tolist() == [3, 0]

    def test_unknown_event_type_defaults_to_0(self):
        """Unrecognised event_type falls back to 0 (fillna default)."""
        result = self._extractor().extract(self._df(["festival"]))
        assert result["event_type_encoded"].iloc[0] == 0

    def test_missing_event_type_column_defaults_to_0(self):
        """Missing event_type column produces event_type_encoded=0 for all rows."""
        df = pd.DataFrame(
            {
                "city": ["Chicago", "LA"],
                "event_datetime": pd.to_datetime(["2024-07-15", "2024-08-01"]),
            }
        )
        result = self._extractor().extract(df)
        assert (result["event_type_encoded"] == 0).all()
