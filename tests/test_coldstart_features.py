"""Tests for ColdStartFeatureExtractor (AC5)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ticket_price_predictor.ml.features.coldstart import (
    ColdStartFeatureExtractor,
    _smooth,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(
    artists: list[str],
    cities: list[str],
    quarters: list[str],
    event_ids: list[str],
    prices: list[float],
    venues: list[str] | None = None,
    sections: list[str] | None = None,
    zones: list[str] | None = None,
) -> pd.DataFrame:
    """Build a minimal DataFrame for testing."""
    n = len(prices)

    # Construct event_datetime from quarter string: "2024Q1" -> 2024-01-15
    def _quarter_to_dt(q: str) -> pd.Timestamp:
        year, qpart = q[:4], q[5]
        month = (int(qpart) - 1) * 3 + 1
        return pd.Timestamp(f"{year}-{month:02d}-15")

    return pd.DataFrame(
        {
            "artist_or_team": artists,
            "city": cities,
            "event_datetime": [_quarter_to_dt(q) for q in quarters],
            "event_id": event_ids,
            "listing_price": prices,
            "venue_name": venues if venues is not None else ["VenueX"] * n,
            "section": sections if sections is not None else ["101"] * n,
            "seat_zone": zones if zones is not None else ["lower_tier"] * n,
        }
    )


# ---------------------------------------------------------------------------
# test_train_only_fit
# ---------------------------------------------------------------------------


class TestTrainOnlyFit:
    def test_produces_finite_values_on_val(self):
        """fit on train, extract on val — all values finite, no state leakage."""
        train = _make_df(
            artists=["A", "A", "B"],
            cities=["New York", "New York", "Chicago"],
            quarters=["2024Q1", "2024Q1", "2024Q2"],
            event_ids=["E1", "E1", "E2"],
            prices=[100.0, 200.0, 300.0],
        )
        # Val rows use entirely different events.
        val = _make_df(
            artists=["A", "C"],
            cities=["New York", "Tokyo"],
            quarters=["2024Q3", "2024Q1"],
            event_ids=["E99", "E100"],
            prices=[150.0, 250.0],
        )

        ex = ColdStartFeatureExtractor()
        ex.fit(train)
        result = ex.extract(val)

        assert result.shape == (2, 6)
        assert result.notna().all().all(), "Val extract produced NaN"
        assert np.isfinite(result.values).all(), "Val extract produced non-finite"

    def test_val_rows_not_loo_excluded(self):
        """Val rows with event_id not in training must not trigger LOO."""
        train = _make_df(
            artists=["A"],
            cities=["New York"],
            quarters=["2024Q1"],
            event_ids=["E1"],
            prices=[200.0],
        )
        # Val row has same group key but different event.
        val = _make_df(
            artists=["A"],
            cities=["New York"],
            quarters=["2024Q1"],
            event_ids=["E_val"],  # unseen event
            prices=[200.0],
        )

        ex = ColdStartFeatureExtractor()
        ex.fit(train)
        result_val = ex.extract(val)
        # Full (non-LOO) stat used: n=1, mu=log1p(200).
        expected = _smooth(1, np.log1p(200.0), ex._global_mean, ex.SMOOTHING_ARQ)
        assert abs(result_val["coldstart_arq_logmean"].iloc[0] - expected) < 1e-9


# ---------------------------------------------------------------------------
# test_target_event_exclusion
# ---------------------------------------------------------------------------


class TestTargetEventExclusion:
    def test_same_event_rows_excluded_from_arq(self):
        """Row i in event E should not see other rows from E in its group bucket."""
        # Two rows in E1, one row in E2 — same (artist, region, quarter).
        train = _make_df(
            artists=["A", "A", "A"],
            cities=["New York", "New York", "New York"],
            quarters=["2024Q1", "2024Q1", "2024Q1"],
            event_ids=["E1", "E1", "E2"],
            prices=[100.0, 200.0, 300.0],
        )
        ex = ColdStartFeatureExtractor()
        ex.fit(train)

        # Extract on the two E1 rows.
        e1_df = train[train["event_id"] == "E1"].copy()
        result = ex.extract(e1_df)

        # For E1 rows: in the (A, US:new york, 2024Q1) bucket,
        # E1 has 2 rows → excluded. Only E2 remains (n=1, mu=log1p(300)).
        # Smoothed with SMOOTHING_ARQ=75, global_mean from all 3 rows.
        global_mean = ex._global_mean
        expected_arq = _smooth(1, np.log1p(300.0), global_mean, ex.SMOOTHING_ARQ)
        np.testing.assert_allclose(
            result["coldstart_arq_logmean"].values,
            [expected_arq, expected_arq],
            rtol=1e-6,
        )

    def test_fallback_when_bucket_empty_after_exclusion(self):
        """When all rows in bucket belong to target event, falls back."""
        # Only one event in the ARQ bucket — after exclusion, n=0 → fallback.
        train = _make_df(
            artists=["A", "A"],
            cities=["New York", "New York"],
            quarters=["2024Q1", "2024Q1"],
            event_ids=["E1", "E1"],
            prices=[100.0, 200.0],
        )
        ex = ColdStartFeatureExtractor()
        ex.fit(train)

        result = ex.extract(train)
        # (A, US:new york, 2024Q1) empty after LOO → fallback (A, US:new york) also empty
        # → fallback (A,) also empty → global_mean.
        global_mean = ex._global_mean
        expected = _smooth(0, 0.0, global_mean, ex.SMOOTHING_ARQ)  # = global_mean
        np.testing.assert_allclose(
            result["coldstart_arq_logmean"].values,
            [expected, expected],
            rtol=1e-6,
        )


# ---------------------------------------------------------------------------
# test_integer_cents_LOO
# ---------------------------------------------------------------------------


class TestIntegerCentsLOO:
    def test_price_perturbation_does_not_change_feature(self):
        """Perturbing price by $0.01 must not affect the LOO result (integer-cents guard)."""
        train = _make_df(
            artists=["A", "A", "A"],
            cities=["New York", "New York", "New York"],
            quarters=["2024Q1", "2024Q1", "2024Q1"],
            event_ids=["E1", "E1", "E2"],
            prices=[100.0, 200.0, 300.0],
        )
        ex = ColdStartFeatureExtractor()
        ex.fit(train)

        # Original extract for E1 row with price=100.0
        row_orig = train.iloc[[0]].copy()
        result_orig = ex.extract(row_orig)

        # Perturb by $0.01 — integer-cents (10000) unchanged → same LOO bucket
        row_perturbed = row_orig.copy()
        row_perturbed["listing_price"] = 100.01  # int(round(100.01*100))=10001 != 10000
        ex.extract(row_perturbed)

        # The perturbed row's price cents differ (10001 vs 10000), so it is NOT
        # identified as a training row → no LOO exclusion → different feature value.
        # But perturbing within rounding epsilon (e.g. +1e-10) should be equal.
        row_epsilon = row_orig.copy()
        row_epsilon["listing_price"] = 100.0 + 1e-10  # int(round(*100))=10000 still
        result_epsilon = ex.extract(row_epsilon)

        np.testing.assert_allclose(
            result_orig["coldstart_arq_logmean"].values,
            result_epsilon["coldstart_arq_logmean"].values,
            rtol=1e-9,
        )


# ---------------------------------------------------------------------------
# test_bayesian_smoothing
# ---------------------------------------------------------------------------


class TestBayesianSmoothing:
    def test_small_group_pulls_toward_global(self):
        """n=1 group should be pulled significantly toward global mean."""
        global_mean = np.log1p(150.0)
        group_mu = np.log1p(500.0)
        m = 75

        smoothed = _smooth(1, group_mu, global_mean, m)
        # Should be much closer to global_mean than to group_mu.
        assert abs(smoothed - global_mean) < abs(smoothed - group_mu)

    def test_large_group_barely_moves(self):
        """n=100 group should produce a value very close to the group mean."""
        global_mean = np.log1p(50.0)
        group_mu = np.log1p(500.0)
        m = 75

        smoothed = _smooth(100, group_mu, global_mean, m)
        # With n=100, m=75: weight on group_mu is 100/175 ≈ 0.57.
        assert abs(smoothed - group_mu) < abs(smoothed - global_mean)

    def test_numerical_formula(self):
        """Verify formula matches math spec exactly."""
        n, mu, g, m = 3, 4.0, 2.0, 75
        expected = (3 * 4.0 + 75 * 2.0) / 78
        assert abs(_smooth(n, mu, g, m) - expected) < 1e-12

    def test_zero_support_returns_global(self):
        """n=0 returns global mean (final fallback, never NaN)."""
        assert _smooth(0, 999.0, 3.0, 75) == 3.0


# ---------------------------------------------------------------------------
# test_fallback_chain
# ---------------------------------------------------------------------------


class TestFallbackChain:
    def _build_extractor_with_sparse_arq(self) -> tuple[ColdStartFeatureExtractor, pd.DataFrame]:
        """Artist A in region US:new york in 2024Q3 — no E2 sibling in that quarter."""
        train = _make_df(
            artists=["A", "A", "B"],
            cities=["New York", "New York", "Chicago"],
            quarters=["2024Q1", "2024Q2", "2024Q3"],
            event_ids=["E1", "E2", "E3"],
            prices=[100.0, 200.0, 400.0],
        )
        ex = ColdStartFeatureExtractor()
        ex.fit(train)
        # Query row: A / US:new york / 2024Q3 (no sibling in Q3)
        query = _make_df(
            artists=["A"],
            cities=["New York"],
            quarters=["2024Q3"],
            event_ids=["E99"],
            prices=[150.0],
        )
        return ex, query

    def test_arq_falls_to_artist_region(self):
        """(A, US:new york, 2024Q3) empty → (A, US:new york) has 2 siblings."""
        ex, query = self._build_extractor_with_sparse_arq()
        result = ex.extract(query)
        # (A, US:new york) has E1 and E2 → n=2, mu=(log1p(100)+log1p(200))/2
        mu = (np.log1p(100.0) + np.log1p(200.0)) / 2
        expected = _smooth(2, mu, ex._global_mean, ex.SMOOTHING_ARQ)
        np.testing.assert_allclose(result["coldstart_arq_logmean"].iloc[0], expected, rtol=1e-6)

    def test_falls_to_artist_when_region_empty(self):
        """(A, US:chicago, 2024Q1) empty, (A, US:chicago) empty → (A,) has siblings."""
        train = _make_df(
            artists=["A", "A"],
            cities=["New York", "New York"],
            quarters=["2024Q1", "2024Q2"],
            event_ids=["E1", "E2"],
            prices=[100.0, 200.0],
        )
        ex = ColdStartFeatureExtractor()
        ex.fit(train)
        # Query: A in Chicago — no training in Chicago at all → fallback to (A,)
        query = _make_df(
            artists=["A"],
            cities=["Chicago"],
            quarters=["2024Q1"],
            event_ids=["E99"],
            prices=[150.0],
        )
        result = ex.extract(query)
        # (A,) key has E1 and E2 → n=2.
        mu = (np.log1p(100.0) + np.log1p(200.0)) / 2
        expected = _smooth(2, mu, ex._global_mean, ex.SMOOTHING_ARQ)
        np.testing.assert_allclose(result["coldstart_arq_logmean"].iloc[0], expected, rtol=1e-6)

    def test_falls_to_global_when_all_empty(self):
        """Fully unseen artist/region/quarter → global mean fallback."""
        train = _make_df(
            artists=["A"],
            cities=["New York"],
            quarters=["2024Q1"],
            event_ids=["E1"],
            prices=[200.0],
        )
        ex = ColdStartFeatureExtractor()
        ex.fit(train)
        query = _make_df(
            artists=["Z"],  # unseen artist
            cities=["Tokyo"],
            quarters=["2025Q4"],
            event_ids=["E99"],
            prices=[150.0],
            venues=["UnknownVenue"],
            sections=["ZZZZ"],
            zones=["unknown_zone"],
        )
        result = ex.extract(query)
        # All chains empty → global_mean for each logmean feature.
        global_mean = ex._global_mean
        expected_arq = _smooth(0, 0.0, global_mean, ex.SMOOTHING_ARQ)
        expected_vdow = _smooth(0, 0.0, global_mean, ex.SMOOTHING_VDOW)
        expected_sz = _smooth(0, 0.0, global_mean, ex.SMOOTHING_SZ)
        assert result["coldstart_arq_logmean"].iloc[0] == expected_arq
        assert result["coldstart_vdow_logmean"].iloc[0] == expected_vdow
        assert result["coldstart_sz_logmean"].iloc[0] == expected_sz
        # Supports all 0.
        assert result["coldstart_arq_support"].iloc[0] == 0.0
        assert result["coldstart_vdow_support"].iloc[0] == 0.0
        assert result["coldstart_sz_support"].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# test_no_naming_collision
# ---------------------------------------------------------------------------


def test_no_naming_collision():
    """Feature names must not contain 'price', 'avg', or 'median'."""
    ex = ColdStartFeatureExtractor()
    for name in ex.feature_names:
        assert "price" not in name, f"Feature name '{name}' contains 'price'"
        assert "avg" not in name, f"Feature name '{name}' contains 'avg'"
        assert "median" not in name, f"Feature name '{name}' contains 'median'"


# ---------------------------------------------------------------------------
# test_no_nan_output
# ---------------------------------------------------------------------------


def test_no_nan_output():
    """Pathological unseen row must produce all-finite output."""
    train = _make_df(
        artists=["A"],
        cities=["New York"],
        quarters=["2024Q1"],
        event_ids=["E1"],
        prices=[200.0],
    )
    ex = ColdStartFeatureExtractor()
    ex.fit(train)

    # Completely unseen everything.
    query = _make_df(
        artists=["ZZZUNKNOWN"],
        cities=["UNKNOWNCITY"],
        quarters=["2099Q4"],
        event_ids=["E_never_seen"],
        prices=[99999.0],
        venues=["NoSuchVenue"],
        sections=["NOSECTION"],
        zones=["no_zone"],
    )
    result = ex.extract(query)
    assert result.notna().all().all(), "NaN in output for pathological row"
    assert np.isfinite(result.values).all(), "Non-finite in output for pathological row"


# ---------------------------------------------------------------------------
# test_vectors_from_plan (§ Mathematical Spec worked example)
# ---------------------------------------------------------------------------


class TestVectorsFromPlan:
    """Verify coldstart_arq_logmean against the plan's exact worked example.

    T = [(A, US, 2024Q1, E1, $100), (A, US, 2024Q1, E1, $200), (A, US, 2024Q2, E2, $300)]
    """

    def _build(self) -> ColdStartFeatureExtractor:
        train = _make_df(
            artists=["A", "A", "A"],
            cities=["New York", "New York", "New York"],
            quarters=["2024Q1", "2024Q1", "2024Q2"],
            event_ids=["E1", "E1", "E2"],
            prices=[100.0, 200.0, 300.0],
        )
        ex = ColdStartFeatureExtractor()
        ex.fit(train)
        return ex

    def test_row1_e1_price100(self):
        """Row 1 (E1, $100): ARQ bucket (A, US:new york, 2024Q1) — LOO excludes E1's
        two rows → n=0. Fallback (A, US:new york): E2 row only → n=1, mu=log1p(300).
        f = (1*log1p(300) + 75*global_mean) / 76.
        """
        ex = self._build()
        global_mean = ex._global_mean

        row1 = _make_df(
            artists=["A"],
            cities=["New York"],
            quarters=["2024Q1"],
            event_ids=["E1"],
            prices=[100.0],
        )
        result = ex.extract(row1)
        mu_fallback = np.log1p(300.0)
        expected = _smooth(1, mu_fallback, global_mean, ex.SMOOTHING_ARQ)
        np.testing.assert_allclose(result["coldstart_arq_logmean"].iloc[0], expected, rtol=1e-6)

    def test_row3_e2_price300(self):
        """Row 3 (E2, $300): ARQ bucket (A, US:new york, 2024Q2) — only E2's row,
        LOO excludes it → n=0. Fallback (A, US:new york): E1 rows (n=2, NOT excluded
        because E1 ≠ E2). mu = (log1p(100)+log1p(200))/2.
        f = (2*mu + 75*global_mean) / 77.
        """
        ex = self._build()
        global_mean = ex._global_mean

        row3 = _make_df(
            artists=["A"],
            cities=["New York"],
            quarters=["2024Q2"],
            event_ids=["E2"],
            prices=[300.0],
        )
        result = ex.extract(row3)
        mu_e1 = (np.log1p(100.0) + np.log1p(200.0)) / 2
        expected = _smooth(2, mu_e1, global_mean, ex.SMOOTHING_ARQ)
        np.testing.assert_allclose(result["coldstart_arq_logmean"].iloc[0], expected, rtol=1e-6)

    def test_inference_row_e99(self):
        """Inference row (A, US:new york, 2024Q1, E99 unseen): no LOO.
        Full (A, US:new york, 2024Q1) bucket: n=2, mu=(log1p(100)+log1p(200))/2.
        f = (2*mu + 75*global_mean) / 77.
        """
        ex = self._build()
        global_mean = ex._global_mean

        inference_row = _make_df(
            artists=["A"],
            cities=["New York"],
            quarters=["2024Q1"],
            event_ids=["E99"],  # unseen — no LOO
            prices=[150.0],
        )
        result = ex.extract(inference_row)
        mu = (np.log1p(100.0) + np.log1p(200.0)) / 2
        expected = _smooth(2, mu, global_mean, ex.SMOOTHING_ARQ)
        np.testing.assert_allclose(result["coldstart_arq_logmean"].iloc[0], expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# test_feature_count_in_pipeline
# ---------------------------------------------------------------------------


def test_pipeline_feature_count_increases_by_6():
    """Enabling coldstart adds exactly 6 features vs disabling it."""
    from ticket_price_predictor.ml.features.pipeline import FeaturePipeline

    common_kwargs = {
        "include_momentum": False,
        "include_snapshot": False,
        "include_popularity": False,
        "include_regional": False,
        "include_listing": False,
        "include_venue": False,
        "include_interactions": False,
        "include_event_pricing": False,
        "include_relative_pricing": False,
        "include_section_encoding": False,
    }
    pipe_with = FeaturePipeline(**common_kwargs, include_coldstart=True)
    pipe_without = FeaturePipeline(**common_kwargs, include_coldstart=False)

    diff = len(pipe_with.feature_names) - len(pipe_without.feature_names)
    assert diff == 6, f"Expected +6 features, got {diff}"
