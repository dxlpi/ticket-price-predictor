"""Tests for WithinEventDynamicsFeatureExtractor (AC5b)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ticket_price_predictor.ml.features.within_event import (
    WithinEventDynamicsFeatureExtractor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(
    event_id: str,
    timestamp: str,
    price: float,
    zone: str = "floor",
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "timestamp": pd.Timestamp(timestamp, tz="UTC"),
        "listing_price": price,
        "section": zone,
    }


def _df(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def _fit_extract(
    train_rows: list[dict[str, object]],
    extract_rows: list[dict[str, object]],
    is_train: bool,
) -> pd.DataFrame:
    ext = WithinEventDynamicsFeatureExtractor()
    ext.fit(_df(*train_rows))
    return ext.extract(_df(*extract_rows), is_train=is_train)


# ---------------------------------------------------------------------------
# test_ordinal_monotone
# ---------------------------------------------------------------------------


def test_ordinal_monotone() -> None:
    """Within an event, pos_in_sequence is non-decreasing by timestamp."""
    train = [
        _make_row("E1", "2024-01-01 10:00", 100.0),
        _make_row("E1", "2024-01-01 11:00", 110.0),
        _make_row("E1", "2024-01-01 12:00", 120.0),
        _make_row("E1", "2024-01-01 13:00", 130.0),
        _make_row("E1", "2024-01-01 14:00", 140.0),
    ]
    # Extract on the same rows in train mode (LOO)
    ext = WithinEventDynamicsFeatureExtractor()
    ext.fit(_df(*train))
    result = ext.extract(_df(*train), is_train=True)
    positions = result["we_pos_in_sequence"].tolist()
    assert positions == sorted(positions), f"Non-monotone: {positions}"


# ---------------------------------------------------------------------------
# test_no_future_peek_train
# ---------------------------------------------------------------------------


def test_no_future_peek_train() -> None:
    """Train LOO: E_i is strictly less-than, so future rows are excluded."""
    train = [
        _make_row("E1", "2024-01-01 10:00", 100.0),  # t=0
        _make_row("E1", "2024-01-01 12:00", 120.0),  # t=2h — the query row
        _make_row("E1", "2024-01-01 14:00", 140.0),  # t=4h — future
    ]
    ext = WithinEventDynamicsFeatureExtractor()
    ext.fit(_df(*train))

    # Query: the middle row (t=12:00). With is_train=True (strict-<), only t=10:00 is in E_i.
    query = _df(_make_row("E1", "2024-01-01 12:00", 120.0))
    result = ext.extract(query, is_train=True)
    assert result["we_pos_in_sequence"].iloc[0] == 1.0, (
        f"Expected only 1 prior row (strict-less-than), got {result['we_pos_in_sequence'].iloc[0]}"
    )


# ---------------------------------------------------------------------------
# test_straddling_event_consecutive
# ---------------------------------------------------------------------------


def test_straddling_event_consecutive() -> None:
    """Train has 10 listings at t=1..10h. Val row at t=6h gets pos=6 (not reset)."""
    base = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    train_rows = [
        _make_row("E1", str(base + pd.Timedelta(hours=h)), 100.0 + h)
        for h in range(1, 11)  # t=1h..10h inclusive
    ]
    ext = WithinEventDynamicsFeatureExtractor()
    ext.fit(_df(*train_rows))

    # Val row at t=6h → ≤ scoping → 6 training rows have t_j ≤ t=6h
    val_row = _make_row("E1", str(base + pd.Timedelta(hours=6)), 105.0)
    result = ext.extract(_df(val_row), is_train=False)
    assert result["we_pos_in_sequence"].iloc[0] == 6.0, (
        f"Expected pos=6, got {result['we_pos_in_sequence'].iloc[0]}"
    )


# ---------------------------------------------------------------------------
# test_age_percentile_denominator
# ---------------------------------------------------------------------------


def test_age_percentile_denominator() -> None:
    """age_percentile denominator = |H[e]| = 10, consistent across splits."""
    base = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    train_rows = [_make_row("E1", str(base + pd.Timedelta(hours=h)), 100.0) for h in range(1, 11)]
    ext = WithinEventDynamicsFeatureExtractor()
    ext.fit(_df(*train_rows))

    # Train row at t=3h → strict-less-than → k=2 (rows at t=1h, t=2h only)
    # age = 2 / |H[e]| = 2/10
    train_query = _make_row("E1", str(base + pd.Timedelta(hours=3)), 100.0)
    tr = ext.extract(_df(train_query), is_train=True)
    assert abs(tr["we_age_percentile"].iloc[0] - 2.0 / 10.0) < 1e-9, (
        f"Train age_percentile wrong: {tr['we_age_percentile'].iloc[0]}"
    )

    # Val row at t=3h → ≤ scoping → k=3 (rows at t=1h, t=2h, t=3h)
    # age = 3 / |H[e]| = 3/10 — denominator still 10, not |E_i|
    val_query = _make_row("E1", str(base + pd.Timedelta(hours=3)), 100.0)
    vr = ext.extract(_df(val_query), is_train=False)
    assert abs(vr["we_age_percentile"].iloc[0] - 3.0 / 10.0) < 1e-9, (
        f"Val age_percentile wrong: {vr['we_age_percentile'].iloc[0]}"
    )

    # Denominators are the same (|H[e]|=10), numerators differ by 1 due to LOO vs ≤
    assert tr["we_history_support"].iloc[0] == vr["we_history_support"].iloc[0] == 10.0


# ---------------------------------------------------------------------------
# test_extract_leak
# ---------------------------------------------------------------------------


def test_extract_leak() -> None:
    """Features depend only on training history, not on sibling test-row prices."""
    train = [
        _make_row("E1", "2024-01-01 10:00", 100.0),
        _make_row("E1", "2024-01-01 11:00", 110.0),
    ]
    ext = WithinEventDynamicsFeatureExtractor()
    ext.fit(_df(*train))

    # Two test batches with different sibling prices — output must be identical
    query_a = _df(
        _make_row("E1", "2024-01-01 12:00", 200.0),
        _make_row("E1", "2024-01-01 13:00", 999.0),  # sibling with very different price
    )
    query_b = _df(
        _make_row("E1", "2024-01-01 12:00", 200.0),
        _make_row("E1", "2024-01-01 13:00", 1.0),  # sibling with very different price
    )

    res_a = ext.extract(query_a, is_train=False)
    res_b = ext.extract(query_b, is_train=False)

    # First row (t=12:00) must be identical in both batches — sibling at t=13:00 must not affect it
    pd.testing.assert_series_equal(
        res_a.iloc[0],
        res_b.iloc[0],
        check_names=False,
        rtol=1e-9,
    )


# ---------------------------------------------------------------------------
# test_first_listing
# ---------------------------------------------------------------------------


def test_first_listing() -> None:
    """Train row with no prior history — all features are 0."""
    train = [_make_row("E1", "2024-01-01 10:00", 100.0)]
    ext = WithinEventDynamicsFeatureExtractor()
    ext.fit(_df(*train))

    # The first (and only) training row has no strict-less-than predecessors
    result = ext.extract(_df(*train), is_train=True)
    row = result.iloc[0]
    assert row["we_pos_in_sequence"] == 0.0
    assert row["we_time_since_first_hours"] == 0.0
    assert row["we_rolling5_dev"] == 0.0
    assert row["we_zone_price_at_t"] == 0.0


# ---------------------------------------------------------------------------
# test_unseen_event_val
# ---------------------------------------------------------------------------


def test_unseen_event_val() -> None:
    """Val row with unseen event_id gets all features at baseline (0)."""
    train = [_make_row("E1", "2024-01-01 10:00", 100.0)]
    ext = WithinEventDynamicsFeatureExtractor()
    ext.fit(_df(*train))

    result = ext.extract(_df(_make_row("E_UNSEEN", "2024-01-01 12:00", 150.0)), is_train=False)
    row = result.iloc[0]
    for col in [
        "we_pos_in_sequence",
        "we_time_since_first_hours",
        "we_age_percentile",
        "we_rolling5_dev",
        "we_zone_price_at_t",
        "we_history_support",
    ]:
        assert row[col] == 0.0, f"{col} should be 0 for unseen event, got {row[col]}"


# ---------------------------------------------------------------------------
# test_perturb_cents — sequence positions are distinct for identical (t, z, p) pairs
# ---------------------------------------------------------------------------


def test_perturb_cents() -> None:
    """Two listings with identical (t, zone, price) produce distinct pos_in_sequence."""
    # Same timestamp: searchsorted puts both at the same position.
    # But they are distinct rows so they must get distinct positions from E_i
    # (training history preserves duplicates — we never merge same-price rows).
    ts = "2024-01-01 10:00"
    train = [
        _make_row("E1", ts, 100.0, "floor"),
        _make_row("E1", ts, 100.0, "floor"),  # identical duplicate
        _make_row("E1", "2024-01-01 11:00", 110.0, "floor"),
    ]
    ext = WithinEventDynamicsFeatureExtractor()
    ext.fit(_df(*train))

    # H[E1] should have 3 entries
    assert ext._history["E1"]["size"] == 3

    # Val row at t=11h (≤) → all 3 training rows qualify
    val = _df(_make_row("E1", "2024-01-01 11:00", 110.0, "floor"))
    result = ext.extract(val, is_train=False)
    assert result["we_pos_in_sequence"].iloc[0] == 3.0, (
        f"Expected pos=3, got {result['we_pos_in_sequence'].iloc[0]}"
    )


# ---------------------------------------------------------------------------
# test_feature_names
# ---------------------------------------------------------------------------


def test_feature_names() -> None:
    """feature_names returns exactly the 6 we_* columns."""
    ext = WithinEventDynamicsFeatureExtractor()
    names = ext.feature_names
    assert len(names) == 6
    assert all(n.startswith("we_") for n in names), f"Unexpected names: {names}"
    # Verify the exact columns match the spec
    expected = {
        "we_pos_in_sequence",
        "we_time_since_first_hours",
        "we_age_percentile",
        "we_rolling5_dev",
        "we_zone_price_at_t",
        "we_history_support",
    }
    assert set(names) == expected, f"Got {names}"


# ---------------------------------------------------------------------------
# test_rolling5_dev — basic correctness
# ---------------------------------------------------------------------------


def test_rolling5_dev() -> None:
    """we_rolling5_dev = log(p_i / median(last-5 in E_i)) when |E_i|≥5."""
    base = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    prices = [100.0, 110.0, 90.0, 105.0, 95.0]  # 5 training prices
    train_rows = [
        _make_row("E1", str(base + pd.Timedelta(hours=h + 1)), p) for h, p in enumerate(prices)
    ]
    ext = WithinEventDynamicsFeatureExtractor()
    ext.fit(_df(*train_rows))

    # Val row at t = after all 5 → ≤ scoping includes all 5
    query_price = 120.0
    val_row = _make_row("E1", str(base + pd.Timedelta(hours=10)), query_price)
    result = ext.extract(_df(val_row), is_train=False)

    expected_med = float(np.median(prices))
    expected_dev = np.log(query_price / expected_med)
    got = result["we_rolling5_dev"].iloc[0]
    assert abs(got - expected_dev) < 1e-9, f"Expected {expected_dev}, got {got}"


# ---------------------------------------------------------------------------
# test_zone_price_at_t — basic correctness
# ---------------------------------------------------------------------------


def test_zone_price_at_t() -> None:
    """zone_price_at_t = log(p_i / median(same-zone in E_i))."""
    base = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    train_rows = [
        _make_row("E1", str(base + pd.Timedelta(hours=1)), 100.0, "floor"),
        _make_row("E1", str(base + pd.Timedelta(hours=2)), 120.0, "floor"),
        _make_row("E1", str(base + pd.Timedelta(hours=3)), 200.0, "vip"),  # different zone
    ]
    ext = WithinEventDynamicsFeatureExtractor()
    ext.fit(_df(*train_rows))

    # Val row: floor zone, after all training rows
    query_price = 110.0
    val_row = _make_row("E1", str(base + pd.Timedelta(hours=5)), query_price, "floor")
    result = ext.extract(_df(val_row), is_train=False)

    floor_prices = [100.0, 120.0]
    expected_med = float(np.median(floor_prices))
    expected_dev = np.log(query_price / expected_med)
    got = result["we_zone_price_at_t"].iloc[0]
    assert abs(got - expected_dev) < 1e-9, f"Expected {expected_dev}, got {got}"


# ---------------------------------------------------------------------------
# test_pipeline_integration — WithinEventDynamicsFeatureExtractor registered
# ---------------------------------------------------------------------------


def test_pipeline_integration() -> None:
    """FeaturePipeline includes we_* features when include_within_event=True."""
    from ticket_price_predictor.ml.features.pipeline import FeaturePipeline

    pipeline = FeaturePipeline(
        include_within_event=True,
        include_coldstart=False,
        include_section_encoding=False,
        include_popularity=False,
        include_regional=False,
        include_listing=False,
        include_venue=False,
        include_interactions=False,
        include_snapshot=False,
        include_event_pricing=False,
        include_relative_pricing=False,
    )
    we_features = [f for f in pipeline.feature_names if f.startswith("we_")]
    assert len(we_features) == 6, f"Expected 6 we_* features, got {we_features}"


def test_pipeline_without_within_event() -> None:
    """FeaturePipeline excludes we_* features when include_within_event=False."""
    from ticket_price_predictor.ml.features.pipeline import FeaturePipeline

    pipeline = FeaturePipeline(
        include_within_event=False,
        include_coldstart=False,
        include_section_encoding=False,
        include_popularity=False,
        include_regional=False,
        include_listing=False,
        include_venue=False,
        include_interactions=False,
        include_snapshot=False,
        include_event_pricing=False,
        include_relative_pricing=False,
    )
    we_features = [f for f in pipeline.feature_names if f.startswith("we_")]
    assert we_features == [], f"Expected no we_* features, got {we_features}"
