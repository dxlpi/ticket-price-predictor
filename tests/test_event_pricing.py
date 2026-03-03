"""Tests for EventPricingFeatureExtractor."""

import numpy as np
import pandas as pd

from ticket_price_predictor.ml.features.event_pricing import EventPricingFeatureExtractor


def _make_df(
    event_ids: list[str],
    prices: list[float],
    sections: list[str] | None = None,
    artists: list[str] | None = None,
) -> pd.DataFrame:
    """Helper to construct a test DataFrame."""
    data: dict[str, list] = {
        "event_id": event_ids,
        "listing_price": prices,
    }
    if sections is not None:
        data["section"] = sections
    if artists is not None:
        data["artist_or_team"] = artists
    return pd.DataFrame(data)


class TestFeatureNames:
    """Test that feature names are correct and stable."""

    def test_feature_names_correct(self) -> None:
        extractor = EventPricingFeatureExtractor()
        assert extractor.feature_names == [
            "event_median_price",
            "event_zone_median_price",
            "event_listing_count",
            "event_price_cv",
            "event_zone_price_ratio",
        ]

    def test_feature_names_count(self) -> None:
        extractor = EventPricingFeatureExtractor()
        assert len(extractor.feature_names) == 5


class TestFitStats:
    """Test that fit() computes correct statistics."""

    def test_fit_computes_event_median(self) -> None:
        df = _make_df(
            event_ids=["e1", "e1", "e1"],
            prices=[100.0, 200.0, 300.0],
        )
        extractor = EventPricingFeatureExtractor()
        extractor.fit(df)
        # Smoothing: n=3, global_median=200, m=20 → (3*200 + 20*200) / 23 = 200
        assert "e1" in extractor._event_stats
        stats = extractor._event_stats["e1"]
        assert abs(stats["median"] - 200.0) < 1.0  # median=200, global=200 → smoothed≈200

    def test_fit_smoothes_small_groups_toward_global(self) -> None:
        # e1 has 1 listing at $500, global median is closer to $100 (from e2)
        df = _make_df(
            event_ids=["e1", "e2", "e2", "e2", "e2", "e2"],
            prices=[500.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        )
        extractor = EventPricingFeatureExtractor()
        extractor.fit(df)
        # e1: n=1, group_median=500, global_median≈100, m=20
        # smoothed = (1*500 + 20*100) / 21 = 2500/21 ≈ 119.0
        stats = extractor._event_stats["e1"]
        assert stats["median"] < 200.0  # Should be pulled toward global

    def test_fit_sets_fitted_flag(self) -> None:
        df = _make_df(["e1"], [100.0])
        extractor = EventPricingFeatureExtractor()
        assert not extractor._fitted
        extractor.fit(df)
        assert extractor._fitted

    def test_fit_computes_zone_stats_when_section_present(self) -> None:
        df = _make_df(
            event_ids=["e1", "e1", "e1"],
            prices=[300.0, 150.0, 80.0],
            sections=["Floor A", "Section 101", "Section 401"],
        )
        extractor = EventPricingFeatureExtractor()
        extractor.fit(df)
        # Should have event-zone stats
        assert len(extractor._event_zone_stats) > 0

    def test_fit_computes_artist_stats_when_present(self) -> None:
        df = _make_df(
            event_ids=["e1", "e2"],
            prices=[100.0, 200.0],
            artists=["Taylor Swift", "Taylor Swift"],
        )
        extractor = EventPricingFeatureExtractor()
        extractor.fit(df)
        assert "Taylor Swift" in extractor._artist_stats

    def test_fit_missing_required_columns_uses_defaults(self) -> None:
        # No event_id column — should gracefully fall back
        df = pd.DataFrame({"listing_price": [100.0, 200.0]})
        extractor = EventPricingFeatureExtractor()
        extractor.fit(df)
        assert extractor._fitted
        assert extractor._global_stats["median"] == 150.0  # default


class TestExtract:
    """Test that extract() produces correct shapes and values."""

    def test_extract_produces_correct_shape(self) -> None:
        df = _make_df(
            event_ids=["e1", "e1", "e2"],
            prices=[100.0, 200.0, 300.0],
        )
        extractor = EventPricingFeatureExtractor()
        extractor.fit(df)
        result = extractor.extract(df)
        assert result.shape == (3, 5)

    def test_extract_column_names(self) -> None:
        df = _make_df(["e1", "e1"], [100.0, 200.0])
        extractor = EventPricingFeatureExtractor()
        extractor.fit(df)
        result = extractor.extract(df)
        assert list(result.columns) == extractor.feature_names

    def test_extract_listing_count_is_log1p(self) -> None:
        df = _make_df(
            event_ids=["e1", "e1", "e1"],
            prices=[100.0, 200.0, 300.0],
        )
        extractor = EventPricingFeatureExtractor()
        extractor.fit(df)
        result = extractor.extract(df)
        # event e1 has 3 listings → log1p(3) ≈ 1.386
        expected = np.log1p(3)
        assert abs(result["event_listing_count"].iloc[0] - expected) < 0.01

    def test_extract_price_cv_is_clamped(self) -> None:
        # Very high std/mean ratio
        df = _make_df(["e1", "e1"], [1.0, 10000.0])
        extractor = EventPricingFeatureExtractor()
        extractor.fit(df)
        result = extractor.extract(df)
        assert result["event_price_cv"].iloc[0] <= 3.0
        assert result["event_price_cv"].iloc[0] >= 0.0

    def test_extract_zone_ratio_clamped(self) -> None:
        df = _make_df(
            event_ids=["e1", "e1", "e1"],
            prices=[100.0, 200.0, 10000.0],
            sections=["Section 101", "Section 101", "VIP Floor"],
        )
        extractor = EventPricingFeatureExtractor()
        extractor.fit(df)
        result = extractor.extract(df)
        assert (result["event_zone_price_ratio"] >= 0.1).all()
        assert (result["event_zone_price_ratio"] <= 10.0).all()

    def test_extract_index_preserved(self) -> None:
        df = _make_df(["e1", "e2", "e3"], [100.0, 200.0, 300.0])
        df.index = [10, 20, 30]
        extractor = EventPricingFeatureExtractor()
        extractor.fit(df)
        result = extractor.extract(df)
        assert list(result.index) == [10, 20, 30]

    def test_extract_no_nan_values(self) -> None:
        df = _make_df(
            event_ids=["e1", "e1", "e2"],
            prices=[100.0, 200.0, 300.0],
            sections=["Floor", "Section 101", "Section 401"],
            artists=["Artist A", "Artist A", "Artist B"],
        )
        extractor = EventPricingFeatureExtractor()
        extractor.fit(df)
        result = extractor.extract(df)
        assert not result.isnull().any().any()


class TestFallbackChain:
    """Test that the fallback chain works for unknown events."""

    def test_unknown_event_falls_back_to_artist(self) -> None:
        train_df = _make_df(
            event_ids=["e1", "e1"],
            prices=[200.0, 300.0],
            artists=["Taylor Swift", "Taylor Swift"],
        )
        extractor = EventPricingFeatureExtractor()
        extractor.fit(train_df)

        # Test with unseen event but known artist
        test_df = pd.DataFrame(
            {
                "event_id": ["unseen_event"],
                "listing_price": [250.0],
                "artist_or_team": ["Taylor Swift"],
            }
        )
        result = extractor.extract(test_df)
        # Should use artist stats, not global defaults
        artist_median = extractor._artist_stats["Taylor Swift"]["median"]
        assert abs(result["event_median_price"].iloc[0] - artist_median) < 0.01

    def test_unknown_event_unknown_artist_falls_back_to_global(self) -> None:
        train_df = _make_df(
            event_ids=["e1", "e1"],
            prices=[200.0, 300.0],
        )
        extractor = EventPricingFeatureExtractor()
        extractor.fit(train_df)

        test_df = pd.DataFrame(
            {
                "event_id": ["totally_unseen"],
                "listing_price": [250.0],
            }
        )
        result = extractor.extract(test_df)
        global_median = extractor._global_stats["median"]
        assert abs(result["event_median_price"].iloc[0] - global_median) < 0.01

    def test_unknown_zone_for_known_event_falls_back_to_event_median(self) -> None:
        train_df = _make_df(
            event_ids=["e1", "e1"],
            prices=[200.0, 300.0],
            sections=["Floor A", "Floor B"],  # All floor zones
        )
        extractor = EventPricingFeatureExtractor()
        extractor.fit(train_df)

        # Section 401 = balcony zone — not seen for e1
        test_df = pd.DataFrame(
            {
                "event_id": ["e1"],
                "listing_price": [150.0],
                "section": ["Section 401"],
            }
        )
        result = extractor.extract(test_df)
        # Zone falls back to event median, so ratio should be ~1.0
        assert abs(result["event_zone_price_ratio"].iloc[0] - 1.0) < 0.01

    def test_known_event_zone_uses_zone_stats(self) -> None:
        train_df = _make_df(
            event_ids=["e1", "e1", "e1", "e1"],
            prices=[500.0, 500.0, 100.0, 100.0],
            sections=["VIP Floor", "VIP Floor", "Section 401", "Section 401"],
        )
        extractor = EventPricingFeatureExtractor()
        extractor.fit(train_df)

        # Extract for VIP zone — should have higher zone median than event median
        test_df = pd.DataFrame(
            {
                "event_id": ["e1"],
                "listing_price": [500.0],
                "section": ["VIP Floor"],
            }
        )
        result = extractor.extract(test_df)
        # VIP zone median > event median → ratio > 1
        assert result["event_zone_price_ratio"].iloc[0] > 1.0


class TestMissingColumns:
    """Test graceful handling of missing or NaN columns."""

    def test_missing_section_column(self) -> None:
        df = _make_df(["e1", "e1"], [100.0, 200.0])
        extractor = EventPricingFeatureExtractor()
        extractor.fit(df)
        result = extractor.extract(df)
        assert result.shape == (2, 5)
        assert not result.isnull().any().any()

    def test_missing_artist_column(self) -> None:
        df = _make_df(["e1", "e1"], [100.0, 200.0])
        extractor = EventPricingFeatureExtractor()
        extractor.fit(df)
        # Extract with no artist_or_team column
        test_df = pd.DataFrame(
            {
                "event_id": ["e1", "unseen"],
                "listing_price": [150.0, 150.0],
            }
        )
        result = extractor.extract(test_df)
        assert result.shape == (2, 5)
        assert not result.isnull().any().any()

    def test_nan_section_values(self) -> None:
        df = pd.DataFrame(
            {
                "event_id": ["e1", "e1"],
                "listing_price": [100.0, 200.0],
                "section": [None, "Floor A"],
            }
        )
        extractor = EventPricingFeatureExtractor()
        extractor.fit(df)
        result = extractor.extract(df)
        assert result.shape == (2, 5)
        assert not result.isnull().any().any()

    def test_empty_dataframe_extract(self) -> None:
        train_df = _make_df(["e1", "e1"], [100.0, 200.0])
        extractor = EventPricingFeatureExtractor()
        extractor.fit(train_df)
        empty_df = pd.DataFrame({"event_id": [], "listing_price": []})
        result = extractor.extract(empty_df)
        assert result.shape == (0, 5)
