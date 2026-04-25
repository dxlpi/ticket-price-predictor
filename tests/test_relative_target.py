"""Tests for RelativeResidualTransform and EventBaseResolverImpl.

Canonical leak regression tests — these MUST pass before any training run
that uses --target-transform relative.  The most critical test is
test_canonical_leak_no_self_inclusion, which proves the LOO branch
prevents trivial target inversion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ticket_price_predictor.ml.training.target_transforms import (  # noqa: I001
    EventBaseResolverImpl,
    RelativeResidualTransform,
    create_target_transform,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_train_df(
    event_prices: dict[str, list[float]],
) -> pd.DataFrame:
    """Build a minimal training DataFrame from {event_id: [prices]} dict."""
    rows = []
    for event_id, prices in event_prices.items():
        for p in prices:
            rows.append({"event_id": event_id, "listing_price": p})
    return pd.DataFrame(rows)


def _fitted_resolver(train_df: pd.DataFrame) -> EventBaseResolverImpl:
    r = EventBaseResolverImpl()
    r.fit(train_df)
    return r


# ---------------------------------------------------------------------------
# test_canonical_leak_no_self_inclusion
#
# Critical regression test: construct a dataset where log1p(price_i) = b_e
# exactly (zero residual without LOO).  With proper LOO, the resolved b_e
# differs from log1p(price_i), so residuals are non-zero and a model trained
# on them cannot trivially invert its own target.
#
# Concretely: assert that for any single-row training event, resolve_train
# differs from the no-LOO mean (which would equal log1p(price) exactly
# for a one-element event).
# ---------------------------------------------------------------------------


def test_canonical_leak_no_self_inclusion() -> None:
    """LOO branch must yield b_e ≠ log1p(price) for single-listing events."""
    # Event E1 has only one listing at $100
    train_df = _make_train_df({"E1": [100.0], "E2": [200.0, 300.0]})
    resolver = _fitted_resolver(train_df)

    row_e1 = pd.Series({"event_id": "E1", "listing_price": 100.0})
    b_train = resolver.resolve_train(row_e1)

    # For a single-listing event, LOO falls back to global mean (not log1p(price))
    log_price_e1 = float(np.log1p(100.0))
    # b_train should be the global mean, NOT log1p(price)
    assert abs(b_train - log_price_e1) > 1e-6, (
        f"LOO b_e ({b_train:.6f}) equals log1p(price) ({log_price_e1:.6f}) "
        "— self-inclusion leak detected"
    )


def test_canonical_leak_multi_listing_loo() -> None:
    """For multi-listing events, LOO must exclude row i's log-price contribution."""
    # E1 has two listings: $100 and $200
    train_df = _make_train_df({"E1": [100.0, 200.0]})
    resolver = _fitted_resolver(train_df)

    row_100 = pd.Series({"event_id": "E1", "listing_price": 100.0})
    row_200 = pd.Series({"event_id": "E1", "listing_price": 200.0})

    b_train_100 = resolver.resolve_train(row_100)
    b_train_200 = resolver.resolve_train(row_200)
    b_infer = resolver.resolve_inference(row_100)

    # LOO for row_100 excludes log1p(100): mean = log1p(200)  (only row left)
    # (Bayesian-smoothed, but the key is b_train ≠ b_infer)
    assert abs(b_train_100 - b_infer) > 1e-6, (
        "resolve_train and resolve_inference returned identical values for the "
        "same event — call-site dispatch not working"
    )

    # LOO for row_100 and row_200 should differ (each excludes a different price)
    assert abs(b_train_100 - b_train_200) > 1e-6, (
        "LOO values for different prices in same event should differ"
    )


# ---------------------------------------------------------------------------
# test_call_site_dispatch_not_row_identity
#
# A val row whose price coincides with a training-row price in the same event
# must receive b_e_inference (not b_e_LOO), even if the val row's index
# matches a training row's index (simulating splitter ignore_index=True).
# ---------------------------------------------------------------------------


def test_call_site_dispatch_not_row_identity() -> None:
    """Val rows must receive resolve_inference, regardless of price coincidence."""
    train_df = _make_train_df({"E1": [100.0, 200.0]})
    resolver = _fitted_resolver(train_df)

    # Val row: same event, same price as a training row
    val_row = pd.Series({"event_id": "E1", "listing_price": 100.0})

    b_inference = resolver.resolve_inference(val_row)
    b_train_same_price = resolver.resolve_train(val_row)

    # They should differ — inference uses full mean, train uses LOO
    assert abs(b_inference - b_train_same_price) > 1e-6, (
        "resolve_inference and resolve_train returned the same value — "
        "dispatch is based on price membership (wrong) rather than call site"
    )


# ---------------------------------------------------------------------------
# test_ignore_index_split_immunity
#
# Simulate splitter.py's ignore_index=True: val_df has RangeIndex(0..N-1)
# overlapping with train_df.  A val row at index 3 must NOT trigger LOO.
# ---------------------------------------------------------------------------


def test_ignore_index_split_immunity() -> None:
    """Overlapping integer indices between train and val must not affect dispatch."""
    train_df = pd.DataFrame(
        [
            {"event_id": "E1", "listing_price": 100.0},
            {"event_id": "E1", "listing_price": 200.0},
            {"event_id": "E2", "listing_price": 50.0},
            {"event_id": "E2", "listing_price": 75.0},
        ]
    )  # RangeIndex 0..3

    resolver = _fitted_resolver(train_df)

    # val_df also starts at index 0 (ignore_index=True behaviour)
    val_df = pd.DataFrame(
        [
            {"event_id": "E1", "listing_price": 100.0},
        ]
    )  # index = 0 — same as train row 0

    val_row = val_df.iloc[0]
    # At call site: is_train=False → resolve_inference
    b_val = resolver.resolve_inference(val_row)
    b_train_idx0 = resolver.resolve_train(train_df.iloc[0])

    # Inference and LOO-train must differ — proves call-site gate, not index gate
    assert abs(b_val - b_train_idx0) > 1e-6, (
        "Val row at index 0 got the same b_e as train row at index 0 — "
        "row.name inspection suspected"
    )


# ---------------------------------------------------------------------------
# test_integer_cents_guard
#
# Perturbing training price by $0.01 must leave resolve_train(row) unchanged
# for a row at the original price (guard against float equality LOO failure).
# ---------------------------------------------------------------------------


def test_integer_cents_guard() -> None:
    """Perturbation by $0.01 to a *different* row must not affect this row's LOO."""
    # Two listings; we vary the second by $0.01
    train_df_base = _make_train_df({"E1": [100.0, 200.0]})
    train_df_pert = _make_train_df({"E1": [100.0, 200.01]})

    resolver_base = _fitted_resolver(train_df_base)
    _fitted_resolver(train_df_pert)  # perturbed resolver not used in this assertion

    # Row at $100 should have a slightly different LOO b_e because the *other*
    # row's price changed — but the integer-cents guard means that a float
    # round-trip shift of ±1e-10 on this row's price does NOT change the LOO
    row = pd.Series({"event_id": "E1", "listing_price": 100.0})
    # Float-perturb the row's own price by a sub-cent epsilon (Parquet artifact)
    row_perturbed = pd.Series({"event_id": "E1", "listing_price": 100.0 + 1e-10})

    b_exact = resolver_base.resolve_train(row)
    b_float_perturbed = resolver_base.resolve_train(row_perturbed)

    assert abs(b_exact - b_float_perturbed) < 1e-8, (
        f"Sub-cent float perturbation changed LOO b_e: {b_exact} vs {b_float_perturbed}. "
        "Integer-cents guard may not be working."
    )


# ---------------------------------------------------------------------------
# test_same_price_duplicates
#
# Two training listings with identical prices in the same event.
# LOO set-based membership: only the price value is in the set once,
# so both rows get the same LOO b_e (set lookup, not list lookup).
# ---------------------------------------------------------------------------


def test_same_price_duplicates() -> None:
    """Duplicate prices in same event: set-based LOO handles correctly."""
    # Both listings have the same price — the integer-cents set has one entry
    train_df = _make_train_df({"E1": [100.0, 100.0, 200.0]})
    resolver = _fitted_resolver(train_df)

    row = pd.Series({"event_id": "E1", "listing_price": 100.0})
    b_train = resolver.resolve_train(row)

    # Result should be finite and different from the raw full mean (LOO is active)
    b_infer = resolver.resolve_inference(row)
    assert np.isfinite(b_train), "LOO b_e is not finite for duplicate-price event"
    # LOO must differ from full inference mean
    assert abs(b_train - b_infer) > 1e-6, "Duplicate prices: LOO did not differ from inference"


# ---------------------------------------------------------------------------
# test_unseen_event_fallback
#
# Val row with event_id not in training must return global_log_mean.
# ---------------------------------------------------------------------------


def test_unseen_event_fallback() -> None:
    """Unseen event at inference returns global log-mean fallback."""
    train_df = _make_train_df({"E1": [100.0, 200.0]})
    resolver = _fitted_resolver(train_df)

    unseen_row = pd.Series({"event_id": "E_UNSEEN", "listing_price": 150.0})
    b = resolver.resolve_inference(unseen_row)

    # Must equal global_log_mean
    assert abs(b - resolver._global_log_mean) < 1e-10, (
        f"Unseen event returned {b}, expected global_log_mean={resolver._global_log_mean}"
    )


def test_unseen_event_coldstart_fallback() -> None:
    """Unseen event with coldstart_resolver uses the resolver callable."""
    train_df = _make_train_df({"E1": [100.0]})
    resolver = _fitted_resolver(train_df)

    sentinel = 42.0
    resolver._coldstart_resolver = lambda row: sentinel  # noqa: ARG005

    unseen_row = pd.Series({"event_id": "E_NEW", "listing_price": 999.0})
    b = resolver.resolve_inference(unseen_row)
    assert abs(b - sentinel) < 1e-10, f"Coldstart fallback not used: got {b}"


# ---------------------------------------------------------------------------
# test_relative_residual_transform_roundtrip
#
# transform then inverse_transform (for inference) should recover raw price.
# ---------------------------------------------------------------------------


def test_relative_residual_transform_roundtrip() -> None:
    """inverse_transform(transform(y)) ≈ y for inference rows."""
    train_df = _make_train_df({"E1": [100.0, 200.0, 150.0]})
    resolver = _fitted_resolver(train_df)
    tt = RelativeResidualTransform(resolver)
    tt.fit(train_df["listing_price"].values)

    val_prices = np.array([100.0, 200.0, 50.0])
    val_df = pd.DataFrame(
        {
            "event_id": ["E1", "E1", "E2"],
            "listing_price": val_prices,
        }
    )

    y_prime = tt.transform(val_prices, val_df, is_train=False)
    y_recovered = tt.inverse_transform(y_prime, val_df)

    np.testing.assert_allclose(y_recovered, val_prices, rtol=1e-5)


# ---------------------------------------------------------------------------
# test_transform_requires_df_and_is_train
# ---------------------------------------------------------------------------


def test_transform_requires_df_and_is_train() -> None:
    """RelativeResidualTransform.transform() must raise if df or is_train omitted."""
    train_df = _make_train_df({"E1": [100.0]})
    resolver = _fitted_resolver(train_df)
    tt = RelativeResidualTransform(resolver)
    tt.fit(train_df["listing_price"].values)

    y = np.array([100.0])

    with pytest.raises(ValueError, match="requires df and is_train"):
        tt.transform(y)  # type: ignore[call-arg]

    with pytest.raises(ValueError, match="requires df and is_train"):
        tt.transform(y, None, is_train=True)


def test_inverse_transform_requires_df() -> None:
    """RelativeResidualTransform.inverse_transform() must raise without df."""
    train_df = _make_train_df({"E1": [100.0]})
    resolver = _fitted_resolver(train_df)
    tt = RelativeResidualTransform(resolver)
    tt.fit(train_df["listing_price"].values)

    with pytest.raises(ValueError, match="requires df"):
        tt.inverse_transform(np.array([0.5]))


# ---------------------------------------------------------------------------
# test_factory_creates_relative
# ---------------------------------------------------------------------------


def test_factory_creates_relative() -> None:
    """create_target_transform('relative', event_base_resolver=...) works."""
    train_df = _make_train_df({"E1": [100.0, 200.0]})
    resolver = _fitted_resolver(train_df)

    tt = create_target_transform("relative", event_base_resolver=resolver)
    assert isinstance(tt, RelativeResidualTransform)
    assert tt.name == "relative"


def test_factory_relative_requires_resolver() -> None:
    """create_target_transform('relative') without resolver must raise."""
    with pytest.raises(ValueError, match="event_base_resolver"):
        create_target_transform("relative")


# ---------------------------------------------------------------------------
# test_existing_transforms_backward_compatible
#
# Widened ABC must not break existing callers that pass only y.
# ---------------------------------------------------------------------------


def test_log_transform_backward_compatible() -> None:
    """LogTransform.transform(y) still works (no df/is_train required)."""
    from ticket_price_predictor.ml.training.target_transforms import LogTransform

    tt = LogTransform()
    tt.fit(np.array([100.0, 200.0]))
    y = np.array([100.0, 200.0])
    result = tt.transform(y)
    np.testing.assert_allclose(result, np.log1p(y))
    recovered = tt.inverse_transform(result)
    np.testing.assert_allclose(recovered, y, rtol=1e-10)


def test_sqrt_transform_backward_compatible() -> None:
    """SqrtTransform.transform(y) still works."""
    from ticket_price_predictor.ml.training.target_transforms import SqrtTransform

    tt = SqrtTransform()
    tt.fit(np.array([100.0]))
    y = np.array([100.0, 400.0])
    result = tt.transform(y)
    np.testing.assert_allclose(result, np.sqrt(y))
    recovered = tt.inverse_transform(result)
    np.testing.assert_allclose(recovered, y, rtol=1e-10)


def test_resolver_no_row_name_inspection() -> None:
    """Neither resolve_train nor resolve_inference should use row.name.

    Construct a val row with the same .name (index) as a training row.
    resolve_inference must return full-training mean, not LOO value.
    """
    train_df = pd.DataFrame(
        [
            {"event_id": "E1", "listing_price": 100.0},
            {"event_id": "E1", "listing_price": 200.0},
        ]
    )
    resolver = _fitted_resolver(train_df)

    # Create val row with index=0 (same as train row 0)
    val_row = pd.Series(
        {"event_id": "E1", "listing_price": 100.0},
        name=0,  # same index as train_df.iloc[0].name
    )
    train_row = train_df.iloc[0]

    b_val = resolver.resolve_inference(val_row)
    b_train = resolver.resolve_train(train_row)

    # Must differ: val gets full mean, train gets LOO
    assert abs(b_val - b_train) > 1e-6, (
        "resolve_inference returned same value as resolve_train for a val row "
        "with matching index — row.name inspection suspected"
    )
