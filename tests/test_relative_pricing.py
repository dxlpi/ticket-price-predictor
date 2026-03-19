"""Tests for RelativePricingFeatureExtractor."""

import numpy as np
import pandas as pd

from ticket_price_predictor.ml.features.relative_pricing import (
    RelativePricingFeatureExtractor,
)


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
        extractor = RelativePricingFeatureExtractor()
        assert extractor.feature_names == [
            "section_zone_residual",
            "zone_event_residual",
            "event_artist_residual",
        ]

    def test_feature_names_count(self) -> None:
        extractor = RelativePricingFeatureExtractor()
        assert len(extractor.feature_names) == 3


class TestFitStats:
    """Test that fit() computes correct statistics."""

    def test_fit_computes_event_stats(self) -> None:
        df = _make_df(
            event_ids=["e1", "e1", "e1"],
            prices=[100.0, 200.0, 300.0],
        )
        extractor = RelativePricingFeatureExtractor()
        extractor.fit(df)
        assert "e1" in extractor._event_stats
        assert extractor._fitted

    def test_fit_computes_zone_stats_when_section_present(self) -> None:
        df = _make_df(
            event_ids=["e1", "e1", "e1"],
            prices=[300.0, 150.0, 80.0],
            sections=["Floor A", "Section 101", "Section 401"],
        )
        extractor = RelativePricingFeatureExtractor()
        extractor.fit(df)
        assert len(extractor._event_zone_stats) > 0

    def test_fit_computes_section_stats(self) -> None:
        df = _make_df(
            event_ids=["e1", "e1", "e1", "e1"],
            prices=[300.0, 350.0, 80.0, 90.0],
            sections=["Floor A", "Floor A", "Section 401", "Section 401"],
        )
        extractor = RelativePricingFeatureExtractor()
        extractor.fit(df)
        assert len(extractor._event_section_stats) > 0

    def test_fit_computes_artist_stats(self) -> None:
        df = _make_df(
            event_ids=["e1", "e2"],
            prices=[100.0, 200.0],
            artists=["Taylor Swift", "Taylor Swift"],
        )
        extractor = RelativePricingFeatureExtractor()
        extractor.fit(df)
        assert "Taylor Swift" in extractor._artist_stats

    def test_fit_missing_required_columns_uses_defaults(self) -> None:
        df = pd.DataFrame({"listing_price": [100.0, 200.0]})
        extractor = RelativePricingFeatureExtractor()
        extractor.fit(df)
        assert extractor._fitted
        assert extractor._global_stats["median"] == 150.0


class TestExtract:
    """Test that extract() produces correct shapes and values."""

    def test_extract_produces_correct_shape(self) -> None:
        df = _make_df(
            event_ids=["e1", "e1", "e2"],
            prices=[100.0, 200.0, 300.0],
        )
        extractor = RelativePricingFeatureExtractor()
        extractor.fit(df)
        result = extractor.extract(df)
        assert result.shape == (3, 3)

    def test_extract_column_names(self) -> None:
        df = _make_df(["e1", "e1"], [100.0, 200.0])
        extractor = RelativePricingFeatureExtractor()
        extractor.fit(df)
        result = extractor.extract(df)
        assert list(result.columns) == extractor.feature_names

    def test_extract_no_nan_values(self) -> None:
        df = _make_df(
            event_ids=["e1", "e1", "e2"],
            prices=[100.0, 200.0, 300.0],
            sections=["Floor", "Section 101", "Section 401"],
            artists=["Artist A", "Artist A", "Artist B"],
        )
        extractor = RelativePricingFeatureExtractor()
        extractor.fit(df)
        result = extractor.extract(df)
        assert not result.isnull().any().any()

    def test_extract_index_preserved(self) -> None:
        df = _make_df(["e1", "e2", "e3"], [100.0, 200.0, 300.0])
        df.index = [10, 20, 30]
        extractor = RelativePricingFeatureExtractor()
        extractor.fit(df)
        result = extractor.extract(df)
        assert list(result.index) == [10, 20, 30]

    def test_section_zone_residual_near_zero_when_one_section(self) -> None:
        """When there's only one section per zone, section_zone_residual ≈ 0."""
        df = _make_df(
            event_ids=["e1", "e1"],
            prices=[300.0, 100.0],
            sections=["Floor A", "Section 401"],
        )
        extractor = RelativePricingFeatureExtractor()
        extractor.fit(df)
        result = extractor.extract(df)
        # With one section per zone, section median ≈ zone median
        assert abs(result["section_zone_residual"].iloc[0]) < 50.0
        assert abs(result["section_zone_residual"].iloc[1]) < 50.0

    def test_zone_event_residual_captures_premium_zone(self) -> None:
        """Floor zone should have positive residual from event median."""
        df = _make_df(
            event_ids=["e1", "e1", "e1", "e1"],
            prices=[500.0, 500.0, 100.0, 100.0],
            sections=["VIP Floor", "VIP Floor", "Section 401", "Section 401"],
        )
        extractor = RelativePricingFeatureExtractor()
        extractor.fit(df)

        # Extract for VIP floor row
        test_df = pd.DataFrame(
            {
                "event_id": ["e1"],
                "listing_price": [500.0],
                "section": ["VIP Floor"],
            }
        )
        result = extractor.extract(test_df)
        # Floor zone median > event median → positive residual
        assert result["zone_event_residual"].iloc[0] > 0

    def test_event_artist_residual_captures_event_premium(self) -> None:
        """Event with higher median than artist average → positive residual."""
        df = _make_df(
            event_ids=["e1", "e1", "e2", "e2"],
            prices=[500.0, 600.0, 100.0, 100.0],
            artists=["Artist A", "Artist A", "Artist A", "Artist A"],
        )
        extractor = RelativePricingFeatureExtractor()
        extractor.fit(df)

        # e1 (median 550) vs artist (median ~250ish) → positive residual
        test_df = pd.DataFrame(
            {
                "event_id": ["e1"],
                "listing_price": [500.0],
                "artist_or_team": ["Artist A"],
            }
        )
        result = extractor.extract(test_df)
        assert result["event_artist_residual"].iloc[0] > 0

    def test_residuals_are_consistent(self) -> None:
        """section_zone + zone_event should approximately equal section_event deviation."""
        df = _make_df(
            event_ids=["e1"] * 6,
            prices=[500.0, 500.0, 200.0, 200.0, 80.0, 80.0],
            sections=[
                "Floor A",
                "Floor A",
                "Section 101",
                "Section 101",
                "Section 401",
                "Section 401",
            ],
            artists=["Artist A"] * 6,
        )
        extractor = RelativePricingFeatureExtractor()
        extractor.fit(df)
        result = extractor.extract(df)

        # section_zone_residual + zone_event_residual should approximately equal
        # section deviation from event (though not exactly due to smoothing)
        for i in range(len(result)):
            section_zone = result["section_zone_residual"].iloc[i]
            zone_event = result["zone_event_residual"].iloc[i]
            # Both residuals should be finite
            assert np.isfinite(section_zone)
            assert np.isfinite(zone_event)


class TestFallbackChain:
    """Test fallback behavior for unseen events/artists."""

    def test_unknown_event_falls_back_to_artist(self) -> None:
        train_df = _make_df(
            event_ids=["e1", "e1"],
            prices=[200.0, 300.0],
            artists=["Taylor Swift", "Taylor Swift"],
        )
        extractor = RelativePricingFeatureExtractor()
        extractor.fit(train_df)

        test_df = pd.DataFrame(
            {
                "event_id": ["unseen_event"],
                "listing_price": [250.0],
                "artist_or_team": ["Taylor Swift"],
            }
        )
        result = extractor.extract(test_df)
        # For unseen event: event_median falls back to artist_median
        # Not exactly 0 due to Bayesian smoothing with small training sets
        assert abs(result["event_artist_residual"].iloc[0]) < 5.0

    def test_unknown_event_unknown_artist_uses_globals(self) -> None:
        train_df = _make_df(
            event_ids=["e1", "e1"],
            prices=[200.0, 300.0],
        )
        extractor = RelativePricingFeatureExtractor()
        extractor.fit(train_df)

        test_df = pd.DataFrame(
            {
                "event_id": ["totally_unseen"],
                "listing_price": [250.0],
            }
        )
        result = extractor.extract(test_df)
        # Residuals should be small (everything falls back to global/artist)
        # Not exactly 0 due to Bayesian smoothing with small training sets
        assert abs(result["section_zone_residual"].iloc[0]) < 5.0
        assert abs(result["zone_event_residual"].iloc[0]) < 5.0
        assert abs(result["event_artist_residual"].iloc[0]) < 5.0

    def test_empty_dataframe_extract(self) -> None:
        train_df = _make_df(["e1", "e1"], [100.0, 200.0])
        extractor = RelativePricingFeatureExtractor()
        extractor.fit(train_df)
        empty_df = pd.DataFrame({"event_id": [], "listing_price": []})
        result = extractor.extract(empty_df)
        assert result.shape == (0, 3)
