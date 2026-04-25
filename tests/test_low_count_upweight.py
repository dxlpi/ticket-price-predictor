"""Unit tests for the low_count_upweight sample-weight strategy.

Tests cover weight normalization, low-count identification, degenerate cases,
and weight bounds — without running the full training loop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers — replicate the weight computation from trainer.py in isolation
# ---------------------------------------------------------------------------

_UPWEIGHT_FACTOR = 2.0


def _compute_low_count_weights(
    train_df: pd.DataFrame,
    upweight_factor: float = _UPWEIGHT_FACTOR,
) -> np.ndarray:
    """Compute low_count_upweight sample weights for a training DataFrame.

    Mirrors the implementation in ModelTrainer.train() so tests exercise the
    exact same logic without needing a full training pipeline.

    Args:
        train_df: DataFrame with ``artist_or_team`` and ``event_id`` columns.
        upweight_factor: Weight assigned to rows from low-count artists.

    Returns:
        Float array of shape (n,) with sum == n (mean weight == 1).
        Falls back to uniform weights if required columns are absent.
    """
    n = len(train_df)
    if "artist_or_team" not in train_df.columns or "event_id" not in train_df.columns:
        return np.ones(n, dtype=float)

    artist_event_counts = train_df.groupby("artist_or_team")["event_id"].nunique()
    median_count = float(np.median(artist_event_counts.values))
    low_count_artists = set(artist_event_counts[artist_event_counts < median_count].index)

    raw_weights = (
        train_df["artist_or_team"]
        .map(lambda a: upweight_factor if a in low_count_artists else 1.0)
        .values.astype(float)
    )

    weights: np.ndarray = raw_weights / raw_weights.sum() * n
    return weights


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mixed_df() -> pd.DataFrame:
    """10-row DataFrame with a clear low-count / high-count split.

    Artists A and B each appear in 1 training event (low-count).
    Artists C, D, E each appear in 4 training events (high-count).
    Median training-event count = 4, so A and B are strictly below median.
    """
    rows: list[dict[str, object]] = []
    # Low-count artists: 1 event, 2 listings each
    for artist in ("A", "B"):
        rows.append({"artist_or_team": artist, "event_id": f"{artist}_e0", "listing_price": 100.0})
        rows.append({"artist_or_team": artist, "event_id": f"{artist}_e0", "listing_price": 110.0})
    # High-count artists: 4 events, 1 listing each
    for artist in ("C", "D", "E"):
        for i in range(4):
            rows.append(
                {"artist_or_team": artist, "event_id": f"{artist}_e{i}", "listing_price": 200.0}
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWeightsNormalized:
    def test_sum_equals_n_train(self, mixed_df: pd.DataFrame) -> None:
        weights = _compute_low_count_weights(mixed_df)
        assert len(weights) == len(mixed_df)
        assert abs(weights.sum() - len(mixed_df)) < 1e-9, (
            f"Expected sum={len(mixed_df)}, got {weights.sum()}"
        )

    def test_mean_weight_is_one(self, mixed_df: pd.DataFrame) -> None:
        weights = _compute_low_count_weights(mixed_df)
        assert abs(weights.mean() - 1.0) < 1e-9


class TestLowCountIdentification:
    def test_low_count_artists_get_higher_weight(self, mixed_df: pd.DataFrame) -> None:
        weights = _compute_low_count_weights(mixed_df)
        low_mask = mixed_df["artist_or_team"].isin(["A", "B"])
        high_mask = ~low_mask

        mean_low = weights[low_mask].mean()
        mean_high = weights[high_mask].mean()
        assert mean_low > 1.0, f"Low-count mean weight should be > 1, got {mean_low}"
        assert mean_high < 1.0, f"High-count mean weight should be < 1, got {mean_high}"

    def test_all_rows_within_artist_have_same_weight(self, mixed_df: pd.DataFrame) -> None:
        """All rows for the same artist must receive identical weight."""
        weights = _compute_low_count_weights(mixed_df)
        for artist in mixed_df["artist_or_team"].unique():
            mask = mixed_df["artist_or_team"] == artist
            artist_weights = weights[mask]
            assert np.allclose(artist_weights, artist_weights[0]), (
                f"Artist {artist} has inconsistent weights: {artist_weights}"
            )


class TestNoWeightWhenSingleArtist:
    def test_single_artist_all_ones(self) -> None:
        """Degenerate case: only one artist — no low/high split possible."""
        df = pd.DataFrame(
            {
                "artist_or_team": ["Solo"] * 5,
                "event_id": [f"e{i}" for i in range(5)],
                "listing_price": [100.0] * 5,
            }
        )
        weights = _compute_low_count_weights(df)
        # With a single artist, median == that artist's count, so no artist
        # is strictly below median → all weights should be 1.0 after normalization
        assert np.allclose(weights, 1.0), f"Single-artist weights should all be 1.0, got {weights}"


class TestUpweightFactorBounds:
    def test_weights_in_sensible_range(self, mixed_df: pd.DataFrame) -> None:
        weights = _compute_low_count_weights(mixed_df)
        assert (weights > 0).all(), "All weights must be positive"
        assert (weights < 5.0).all(), (
            f"Weight exceeded 5.0 — possible extreme skew: max={weights.max()}"
        )

    def test_custom_upweight_factor(self, mixed_df: pd.DataFrame) -> None:
        weights = _compute_low_count_weights(mixed_df, upweight_factor=3.0)
        # Normalized sum must still equal n
        assert abs(weights.sum() - len(mixed_df)) < 1e-9
        # Low-count rows should still be upweighted
        low_mask = mixed_df["artist_or_team"].isin(["A", "B"])
        assert weights[low_mask].mean() > 1.0

    def test_missing_columns_returns_uniform(self) -> None:
        df = pd.DataFrame({"listing_price": [100.0, 200.0, 300.0]})
        weights = _compute_low_count_weights(df)
        assert np.allclose(weights, 1.0)
