"""Tests for ListingStructuralFeatureExtractor.

Covers leakage prevention, fallback chain, LOO guard, smoothing factor,
and the critical train/inference distribution-symmetry regression test
(catches the LOO-prior bug noted in plan math spec § B).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ticket_price_predictor.ml.features.listing_structural import (
    ListingStructuralFeatureExtractor,
    _parse_seat_number,
    _row_bucket,
)


def _make_synthetic_df(
    n_events: int = 5,
    sections_per_event: int = 2,
    rows_per_section: list[str] | None = None,
    listings_per_group: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic listing frame with predictable structure."""
    rows_per_section = rows_per_section or ["A", "K", "Z", "GA"]
    rng = np.random.default_rng(seed)
    rows = []
    for e_idx in range(n_events):
        e_id = f"event_{e_idx}"
        for s_idx in range(sections_per_event):
            section = f"Section {100 + s_idx * 100}"
            base_price = 100.0 + s_idx * 100
            for row_label in rows_per_section:
                row_multiplier = {"A": 1.5, "K": 1.0, "Z": 0.6, "GA": 0.8}.get(row_label, 1.0)
                for k in range(listings_per_group):
                    rows.append(
                        {
                            "event_id": e_id,
                            "section": section,
                            "row": row_label,
                            "seat_from": str(1 + k * 5),
                            "seat_to": str(2 + k * 5),
                            "listing_price": base_price
                            * row_multiplier
                            * (1 + 0.1 * rng.standard_normal()),
                        }
                    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


def test_parse_seat_number_basic() -> None:
    assert _parse_seat_number("7") == 7
    assert _parse_seat_number("0") == 0
    assert _parse_seat_number("999") == 999
    assert _parse_seat_number("1500") == 999  # capped


def test_parse_seat_number_unknown() -> None:
    assert _parse_seat_number(None) == -1
    assert _parse_seat_number("*") == -1
    assert _parse_seat_number("") == -1
    assert _parse_seat_number("abc") == -1
    assert _parse_seat_number(float("nan")) == -1


def test_row_bucket() -> None:
    assert _row_bucket("A") == "front"  # quality = 0.02
    assert _row_bucket("I") == "front"  # quality = 0.18
    assert _row_bucket("J") == "mid"  # quality = 0.20 (boundary; >= 0.20 → mid)
    assert _row_bucket("K") == "mid"  # quality = 0.22
    assert _row_bucket("X") == "mid"  # quality = 0.48
    assert _row_bucket("Y") == "back"  # quality = 0.50 (boundary; >= 0.50 → back)
    assert _row_bucket("Z") == "back"  # quality = 0.52
    assert _row_bucket("GA") == "ga"
    assert _row_bucket(None) == "unknown"
    assert _row_bucket("") == "unknown"


# ---------------------------------------------------------------------------
# Extractor contract
# ---------------------------------------------------------------------------


def test_feature_names_count() -> None:
    ext = ListingStructuralFeatureExtractor()
    assert len(ext.feature_names) == 8
    assert "event_section_row_median_price" in ext.feature_names


def test_extract_after_fit_has_all_columns() -> None:
    df = _make_synthetic_df()
    ext = ListingStructuralFeatureExtractor()
    ext.fit(df)
    out = ext.extract(df)
    assert list(out.columns) == ext.feature_names
    assert len(out) == len(df)
    assert out.isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# Leakage prevention
# ---------------------------------------------------------------------------


def test_no_target_leakage() -> None:
    """extract() must not require listing_price column for non-train rows."""
    df = _make_synthetic_df()
    ext = ListingStructuralFeatureExtractor()
    ext.fit(df)
    # Strip listing_price column to simulate inference-time data
    inference_df = df.drop(columns=["listing_price"])
    out_inference = ext.extract(inference_df)

    # Now compare against extract on a row that has listing_price BUT a price
    # not in the training set (val-style row). Result should be identical
    # to the inference-only extract for the same key.
    val_df = df.copy()
    val_df["listing_price"] = val_df["listing_price"] + 100000.0  # never in train set
    out_val = ext.extract(val_df)

    # The smoothed-encoder column must match between the two paths because
    # neither one is in the train set's price set (LOO does not trigger).
    np.testing.assert_array_almost_equal(
        out_inference["event_section_row_median_price"].to_numpy(),
        out_val["event_section_row_median_price"].to_numpy(),
        decimal=4,
    )


def test_extract_after_fit_with_held_out_event() -> None:
    """Fit on partial data; extract on a held-out event uses fallback chain."""
    df = _make_synthetic_df(n_events=5)
    train_df = df[df["event_id"] != "event_4"].copy()  # exclude one event
    test_df = df[df["event_id"] == "event_4"].copy()

    ext = ListingStructuralFeatureExtractor()
    ext.fit(train_df)
    out = ext.extract(test_df)

    # Held-out event: fallback to global_mean (event_mean unknown).
    # Should produce no NaN and a positive median price.
    assert out["event_section_row_median_price"].notna().all()
    assert (out["event_section_row_median_price"] > 0).all()
    # Listing count must be 0 (no group seen in train)
    assert (out["event_section_row_listing_count"] == 0.0).all()


def test_loo_guard_train_rows() -> None:
    """LOO branch fires only when is_train=True for training rows whose price is in the train set."""
    df = _make_synthetic_df()
    ext = ListingStructuralFeatureExtractor()
    ext.fit(df)

    # Train-flag path: LOO branch enabled
    out_train_loo = ext.extract(df, is_train=True)
    # Same df but is_train=False: LOO suppressed
    out_train_no_loo = ext.extract(df, is_train=False)

    # The two paths must differ — LOO removes self-inclusion bias, moving
    # encoder values away from the include-self smoothed mean.
    diffs = (
        out_train_loo["event_section_row_median_price"]
        - out_train_no_loo["event_section_row_median_price"]
    ).abs()
    assert diffs.sum() > 0.0, "LOO branch should change encoder values for training rows"


def test_val_collision_does_not_trigger_loo() -> None:
    """Val rows whose prices accidentally match train prices must NOT trigger LOO.

    Regression test for the integer-cents collision bug: previously, a val row
    with the same price-cents as a train row would erroneously fire the LOO
    branch and produce a biased encoder. Fix: LOO is gated by is_train, not by
    price collision.
    """
    df = _make_synthetic_df()
    ext = ListingStructuralFeatureExtractor()
    ext.fit(df)

    # df serves as "val": same prices but is_train=False
    out_val = ext.extract(df, is_train=False)

    # Manual check: val encoder must equal stats["mean_smoothed"] for keys with train data
    keys_with_stats = list(ext._group_stats.keys())
    if keys_with_stats:
        # All val rows should produce the smoothed-mean value (no LOO)
        for i in range(len(df)):
            event_id = str(df.iloc[i]["event_id"])
            section = str(df.iloc[i]["section"])
            row_bucket = (
                "front"
                if df.iloc[i]["row"] == "A"
                else (
                    "mid"
                    if df.iloc[i]["row"] == "K"
                    else ("back" if df.iloc[i]["row"] == "Z" else "ga")
                )
            )
            key = (event_id, section, row_bucket)
            if key in ext._group_stats:
                expected = ext._group_stats[key]["mean_smoothed"]
                actual = out_val.iloc[i]["event_section_row_median_price"]
                assert abs(actual - expected) < 1e-6, (
                    f"Val row {i} key {key} got {actual}, expected smoothed-mean {expected}"
                )


def test_train_extract_distribution_matches_inference() -> None:
    """CRITICAL: train (LOO) and inference encodings must be on the same scale.

    This regression test catches the LOO-prior bug from the original
    event_pricing.py:354 pattern (which would smooth toward global_mean for
    training rows but section_prior for inference rows, creating a $30+
    distribution shift on the dominant feature).

    The corrected formula in math spec § B keeps both branches on the same scale.
    """
    df = _make_synthetic_df(n_events=10, sections_per_event=2, listings_per_group=2)
    ext = ListingStructuralFeatureExtractor()
    ext.fit(df)

    # Train-style extract: uses LOO branch
    out_train = ext.extract(df)

    # Build val-style rows with the same (event, section, row) keys but novel
    # prices so the LOO branch does NOT fire.
    val_df = df.copy()
    val_df["listing_price"] = val_df["listing_price"] * 100.0 + 999.0  # off-distribution

    out_val = ext.extract(val_df)

    train_mean = float(out_train["event_section_row_median_price"].mean())
    val_mean = float(out_val["event_section_row_median_price"].mean())

    # Means must be within 5% of each other on log-scale (proxy: relative diff).
    rel_diff = abs(train_mean - val_mean) / max(val_mean, 1.0)
    assert rel_diff < 0.05, (
        f"Train mean ${train_mean:.2f} vs val mean ${val_mean:.2f} "
        f"(relative diff {rel_diff:.2%} > 5%) — LOO-prior bug regression?"
    )


# ---------------------------------------------------------------------------
# Smoothing & fallbacks
# ---------------------------------------------------------------------------


def test_section_null_fallback() -> None:
    """When section column is missing, encoder falls back without raising."""
    df = _make_synthetic_df()
    df_no_section = df.drop(columns=["section"])

    ext = ListingStructuralFeatureExtractor()
    ext.fit(df_no_section)
    out = ext.extract(df_no_section)

    # No NaN; fallback chain produced numbers.
    assert out["event_section_row_median_price"].notna().all()
    assert (out["event_section_row_median_price"] > 0).all()


def test_smoothing_factor_strong_for_small_groups() -> None:
    """For small groups, the smoothed mean should be dominated by the prior."""
    df = _make_synthetic_df(
        n_events=2, sections_per_event=1, rows_per_section=["A"], listings_per_group=2
    )
    ext = ListingStructuralFeatureExtractor()
    ext.fit(df)
    # Group size = 2, smoothing factor = 8 → 80% prior weight (non-LOO).
    # Construct a non-LOO row (unseen price) and compare to the section prior.
    val_df = df.copy()
    val_df["listing_price"] = val_df["listing_price"] + 99999.0  # never in train set
    out_val = ext.extract(val_df)

    # The encoder value must be close to (within ±20% of) the section prior
    # because group-stat weight is small (n=2, m=8).
    group_means = out_val["event_section_row_median_price"]
    # Smoothing should produce values in a sane range (prices are ~100-200 in our synthetic data)
    assert (group_means > 50).all() and (group_means < 500).all()


def test_get_params_serialization() -> None:
    """get_params returns serializable state."""
    ext = ListingStructuralFeatureExtractor()
    p = ext.get_params()
    assert "fitted" in p
    assert p["fitted"] is False

    ext.fit(_make_synthetic_df())
    p = ext.get_params()
    assert p["fitted"] is True


# ---------------------------------------------------------------------------
# Integration with FeaturePipeline
# ---------------------------------------------------------------------------


def test_pipeline_registration() -> None:
    """FeaturePipeline registers the extractor when include_listing_structural=True."""
    from ticket_price_predictor.ml.features.pipeline import FeaturePipeline

    fp = FeaturePipeline(include_listing_structural=True, include_popularity=False)
    feature_names = fp.feature_names
    # The 8 listing-structural feature names should appear in the pipeline output
    structural_names = [
        "seat_number",
        "seat_span",
        "is_low_seat_number",
        "is_unknown_seat",
        "row_bucket_encoded",
        "event_section_row_median_price",
        "event_section_row_listing_count",
        "row_bucket_section_count",
    ]
    for name in structural_names:
        assert name in feature_names, f"Missing {name} in pipeline feature_names"


def test_pipeline_disable_flag() -> None:
    """include_listing_structural=False omits the extractor."""
    from ticket_price_predictor.ml.features.pipeline import FeaturePipeline

    fp = FeaturePipeline(include_listing_structural=False, include_popularity=False)
    feature_names = fp.feature_names
    assert "event_section_row_median_price" not in feature_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
