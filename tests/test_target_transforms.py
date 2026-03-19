"""Tests for target transform abstractions."""

import numpy as np
import pytest

from ticket_price_predictor.ml.training.target_transforms import (
    BoxCoxTransform,
    LogTransform,
    SqrtTransform,
    TargetTransform,
    create_target_transform,
)


class TestLogTransform:
    """Tests for LogTransform."""

    def test_is_target_transform(self):
        t = LogTransform()
        assert isinstance(t, TargetTransform)

    def test_name(self):
        assert LogTransform().name == "log"

    def test_round_trip(self):
        y = np.array([10.0, 50.0, 100.0, 500.0, 1000.0])
        t = LogTransform()
        t.fit(y)
        result = t.inverse_transform(t.transform(y))
        np.testing.assert_allclose(result, y, rtol=1e-10)

    def test_transform_values(self):
        y = np.array([100.0])
        t = LogTransform()
        t.fit(y)
        np.testing.assert_allclose(t.transform(y), np.log1p(y))

    def test_inverse_clips_negative(self):
        t = LogTransform()
        t.fit(np.array([1.0]))
        result = t.inverse_transform(np.array([-100.0]))
        assert result[0] == 0.0

    def test_no_fitting_needed(self):
        """LogTransform works without fitting."""
        t = LogTransform()
        result = t.transform(np.array([100.0]))
        assert np.isfinite(result[0])


class TestBoxCoxTransform:
    """Tests for BoxCoxTransform."""

    def test_is_target_transform(self):
        t = BoxCoxTransform()
        assert isinstance(t, TargetTransform)

    def test_name_contains_lambda(self):
        t = BoxCoxTransform()
        t.fit(np.array([10.0, 50.0, 100.0, 500.0, 1000.0]))
        assert "boxcox" in t.name

    def test_round_trip(self):
        np.random.seed(42)
        y = np.random.uniform(10, 2000, size=200)
        t = BoxCoxTransform()
        t.fit(y)
        result = t.inverse_transform(t.transform(y))
        np.testing.assert_allclose(result, y, rtol=1e-4)

    def test_transform_before_fit_raises(self):
        t = BoxCoxTransform()
        with pytest.raises(RuntimeError, match="not fitted"):
            t.transform(np.array([100.0]))

    def test_inverse_before_fit_raises(self):
        t = BoxCoxTransform()
        with pytest.raises(RuntimeError, match="not fitted"):
            t.inverse_transform(np.array([1.0]))

    def test_produces_finite_values(self):
        np.random.seed(42)
        y = np.random.uniform(10, 2000, size=200)
        t = BoxCoxTransform()
        t.fit(y)
        transformed = t.transform(y)
        assert np.all(np.isfinite(transformed))

    def test_inverse_produces_non_negative(self):
        np.random.seed(42)
        y = np.random.uniform(10, 2000, size=200)
        t = BoxCoxTransform()
        t.fit(y)
        transformed = t.transform(y)
        # Perturb some values to simulate model predictions
        perturbed = transformed + np.random.randn(len(transformed)) * 0.1
        result = t.inverse_transform(perturbed)
        assert np.all(result >= 0)

    def test_fit_on_realistic_prices(self):
        """Test with price distribution similar to real ticket data."""
        np.random.seed(42)
        # Mix of cheap and expensive tickets
        prices = np.concatenate(
            [
                np.random.uniform(20, 100, 150),
                np.random.uniform(100, 400, 100),
                np.random.uniform(400, 1500, 50),
            ]
        )
        t = BoxCoxTransform()
        t.fit(prices)
        result = t.inverse_transform(t.transform(prices))
        np.testing.assert_allclose(result, prices, rtol=1e-4)


class TestSqrtTransform:
    """Tests for SqrtTransform."""

    def test_is_target_transform(self):
        t = SqrtTransform()
        assert isinstance(t, TargetTransform)

    def test_name(self):
        assert SqrtTransform().name == "sqrt"

    def test_round_trip(self):
        y = np.array([10.0, 50.0, 100.0, 500.0, 1000.0])
        t = SqrtTransform()
        t.fit(y)
        result = t.inverse_transform(t.transform(y))
        np.testing.assert_allclose(result, y, rtol=1e-10)

    def test_better_separation_than_log(self):
        """sqrt provides more spread for high-value tickets than log."""
        y = np.array([100.0, 1000.0])
        log_t = LogTransform()
        sqrt_t = SqrtTransform()

        log_spread = np.diff(log_t.transform(y))[0]
        sqrt_spread = np.diff(sqrt_t.transform(y))[0]

        # sqrt should have larger spread between $100 and $1000
        assert sqrt_spread > log_spread

    def test_inverse_of_negative_input(self):
        t = SqrtTransform()
        t.fit(np.array([1.0]))
        # inverse is square: (-5)^2 = 25 — mathematically valid
        result = t.inverse_transform(np.array([-5.0]))
        assert result[0] == 25.0


class TestCreateTargetTransform:
    """Tests for factory function."""

    def test_create_log(self):
        t = create_target_transform("log")
        assert isinstance(t, LogTransform)

    def test_create_boxcox(self):
        t = create_target_transform("boxcox")
        assert isinstance(t, BoxCoxTransform)

    def test_create_sqrt(self):
        t = create_target_transform("sqrt")
        assert isinstance(t, SqrtTransform)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown target transform"):
            create_target_transform("invalid")


class TestTransformComparison:
    """Compare transforms on realistic price data."""

    @pytest.fixture
    def price_data(self):
        np.random.seed(42)
        return np.concatenate(
            [
                np.random.uniform(20, 100, 150),
                np.random.uniform(100, 400, 100),
                np.random.uniform(400, 1500, 50),
            ]
        )

    def test_all_transforms_round_trip(self, price_data):
        """All transforms should round-trip within tolerance."""
        for name in ["log", "boxcox", "sqrt"]:
            t = create_target_transform(name)
            t.fit(price_data)
            result = t.inverse_transform(t.transform(price_data))
            np.testing.assert_allclose(result, price_data, rtol=1e-3, err_msg=f"{name} failed")

    def test_all_transforms_produce_finite(self, price_data):
        """All transforms should produce finite values."""
        for name in ["log", "boxcox", "sqrt"]:
            t = create_target_transform(name)
            t.fit(price_data)
            transformed = t.transform(price_data)
            assert np.all(np.isfinite(transformed)), f"{name} produced non-finite"
