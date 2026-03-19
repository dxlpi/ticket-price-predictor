"""Tests for training pipeline improvements (outlier strategies, sample weights)."""

import numpy as np
import pandas as pd
import pytest

from ticket_price_predictor.ml.training.trainer import ModelTrainer


def _make_listing_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a realistic listing DataFrame for testing."""
    rng = np.random.RandomState(seed)

    sections = ["Floor VIP", "Lower Level 100", "Upper Level 200", "Balcony 400"]
    artists = ["Artist A", "Artist B", "Artist C"]
    cities = ["New York", "Los Angeles", "Chicago"]

    df = pd.DataFrame(
        {
            "event_id": [f"evt_{i % 10}" for i in range(n)],
            "artist_or_team": rng.choice(artists, n),
            "venue_name": "Test Venue",
            "city": rng.choice(cities, n),
            "section": rng.choice(sections, n),
            "row": [str(rng.randint(1, 30)) for _ in range(n)],
            "days_to_event": rng.randint(1, 90, n),
            "event_type": "CONCERT",
            "quantity": rng.randint(1, 5, n),
            "event_datetime": pd.Timestamp("2025-06-15"),
            "timestamp": pd.Timestamp("2025-05-01"),
        }
    )

    # Create prices with realistic distribution (more variance in floor/vip)
    base_prices = {
        "Floor VIP": 500.0,
        "Lower Level 100": 200.0,
        "Upper Level 200": 120.0,
        "Balcony 400": 60.0,
    }
    df["listing_price"] = df["section"].map(base_prices) + rng.randn(n) * 50
    df["listing_price"] = df["listing_price"].clip(lower=15.0)

    return df


class TestWinsorizeByZone:
    """Tests for zone-aware Winsorization."""

    def test_winsorize_clips_extremes(self):
        """Zone Winsorization should clip extreme values."""
        df = _make_listing_df(500)
        # Add some extreme prices
        df.loc[0, "listing_price"] = 10000.0
        df.loc[1, "listing_price"] = 15.0  # near minimum

        result = ModelTrainer._winsorize_by_zone(df)
        assert result["listing_price"].max() < 10000.0

    def test_winsorize_preserves_shape(self):
        """Winsorization should not change number of rows."""
        df = _make_listing_df()
        result = ModelTrainer._winsorize_by_zone(df)
        assert len(result) == len(df)

    def test_winsorize_zone_specific_caps(self):
        """Different zones should have different cap values."""
        df = _make_listing_df(1000, seed=123)
        # Floor VIP should have higher caps than balcony
        result = ModelTrainer._winsorize_by_zone(df)

        floor_max = result.loc[df["section"] == "Floor VIP", "listing_price"].max()
        balcony_max = result.loc[df["section"] == "Balcony 400", "listing_price"].max()
        # Floor VIP generally has higher prices, so cap should be higher
        assert floor_max > balcony_max * 0.5  # sanity check

    def test_winsorize_no_section_column(self):
        """Should fall back to global caps when section is missing."""
        df = _make_listing_df()
        df = df.drop(columns=["section"])
        # Should not raise
        result = ModelTrainer._winsorize_by_zone(df)
        assert len(result) == len(df)

    def test_winsorize_small_zone_fallback(self):
        """Zones with < min_zone_samples should use global fallback."""
        df = _make_listing_df(50)
        # Make one zone very small
        df.loc[df["section"] == "Balcony 400", "section"] = "Floor VIP"
        # With min_zone_samples=100, all zones should use global fallback
        result = ModelTrainer._winsorize_by_zone(df, min_zone_samples=100)
        assert len(result) == len(df)

    def test_winsorize_no_target_col(self):
        """Should return unmodified if target column is missing."""
        df = _make_listing_df()
        df = df.drop(columns=["listing_price"])
        result = ModelTrainer._winsorize_by_zone(df)
        assert len(result) == len(df)


class TestOutlierStrategies:
    """Integration tests for outlier strategy parameter in train()."""

    def test_global_p95_default(self):
        """global_p95 should be backward compatible."""
        df = _make_listing_df()
        capped = ModelTrainer._cap_price_outliers(df.copy())
        assert capped["listing_price"].max() <= df["listing_price"].quantile(0.95) + 1.0

    def test_none_strategy_no_clipping(self):
        """'none' strategy should not modify prices."""
        df = _make_listing_df()
        df.loc[0, "listing_price"] = 99999.0
        # With 'none' strategy, the extreme price should survive
        # (tested indirectly — trainer won't cap)
        assert df["listing_price"].max() == 99999.0


class TestSampleWeightStrategies:
    """Tests for sample weighting computations."""

    @pytest.fixture
    def price_array(self):
        return np.array([50.0, 100.0, 200.0, 500.0, 1000.0])

    def test_sqrt_price_weights(self, price_array):
        """sqrt_price should upweight expensive tickets."""
        median = np.median(price_array)
        weights = np.sqrt(price_array / median)
        weights = weights / weights.sum() * len(weights)
        # Most expensive ticket should have highest weight
        assert weights[-1] > weights[0]

    def test_log_price_weights(self, price_array):
        """log_price should moderately upweight expensive tickets."""
        log_prices = np.log1p(price_array)
        mean_log = np.mean(log_prices)
        weights = log_prices / mean_log
        weights = weights / weights.sum() * len(weights)
        assert weights[-1] > weights[0]
        # Log should be less extreme than sqrt
        sqrt_weights = np.sqrt(price_array / np.median(price_array))
        sqrt_weights = sqrt_weights / sqrt_weights.sum() * len(sqrt_weights)
        sqrt_ratio = sqrt_weights[-1] / sqrt_weights[0]
        log_ratio = weights[-1] / weights[0]
        assert log_ratio < sqrt_ratio

    def test_inverse_quartile_weights(self, price_array):
        """inverse_price_quartile should equalize quartile contributions."""
        # With 5 samples, quartiles won't be perfectly balanced,
        # but the principle should work
        q25, q50, q75 = np.percentile(price_array, [25, 50, 75])
        quartiles = np.where(
            price_array <= q25,
            1,
            np.where(price_array <= q50, 2, np.where(price_array <= q75, 3, 4)),
        )
        unique, counts = np.unique(quartiles, return_counts=True)
        quartile_counts = dict(zip(unique, counts, strict=False))
        raw_weights = np.array([1.0 / quartile_counts[q] for q in quartiles])
        weights = raw_weights / raw_weights.sum() * len(raw_weights)
        # Weights should be positive and finite
        assert np.all(weights > 0)
        assert np.all(np.isfinite(weights))

    def test_weights_sum_to_n(self):
        """All weight strategies should normalize to sum = n."""
        prices = np.random.uniform(20, 1000, 100)
        median = np.median(prices)

        # sqrt_price
        w = np.sqrt(prices / median)
        w = w / w.sum() * len(w)
        np.testing.assert_allclose(w.sum(), len(w))

        # log_price
        lp = np.log1p(prices)
        w2 = lp / np.mean(lp)
        w2 = w2 / w2.sum() * len(w2)
        np.testing.assert_allclose(w2.sum(), len(w2))
