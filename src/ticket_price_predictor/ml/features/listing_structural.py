"""Listing structural feature extraction.

Consumes per-listing structural columns currently unused by other extractors:
- seat_from / seat_to (77.9% / 76.7% populated): numeric seat range within row
- row (99% populated): row letter/number
- section + event_id: keys for the (event, section, row_bucket) target encoder

Adds 8 features capturing within-section variance the smoothed
event_section_median_price cannot resolve.

Design decisions documented in plan §"Mathematical Specification → B" and
§"Implementation Phases → Phase 1":

- Encoder uses MEAN (not median) of group prices for sum-decomposable LOO.
- LOO formula smooths toward (e, s) prior — NOT global mean — so train and
  inference encodings stay on the same scale (corrects event_pricing.py:354
  pattern that pulls training rows toward global mean).
- Internal (event, section) prior recomputed inside fit() (not borrowed from
  EventPricingFeatureExtractor) to keep extractors decoupled.
- row_bucket_encoded is ordinal {front:0, mid:1, back:2, ga:3, unknown:4} —
  acceptable for tree-only consumers (LightGBM); a future linear consumer
  would need one-hot.

seat_description (100% populated) is excluded: empirically it's a
deterministic concatenation of section + row + seats (e.g.
"Section 414, Row 2, Seats 19-20"), so it adds no signal beyond the
structured columns we already consume.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ticket_price_predictor.ml.features.base import FeatureExtractor
from ticket_price_predictor.ml.features.section_encoding import _row_quality

# Smoothing factors — see plan math spec § B.
_GROUP_SMOOTHING_FACTOR = 8  # Bayesian smoothing for (event, section, row_bucket) groups
_SECTION_PRIOR_SMOOTHING_FACTOR = 20  # Internal (event, section) prior smoothing

# Sentinel for unknown seat number
_UNKNOWN_SEAT = -1

# Cap seat numbers to avoid pathological values (e.g. "Seats 9001-9004" placeholder)
_MAX_SEAT_NUMBER = 999
_MAX_SEAT_SPAN = 50

# Aisle proxy: low seat numbers in the row are typically near the aisle
_AISLE_THRESHOLD = 5

# Cap distinct-section count per (event, row_bucket)
_MAX_SECTION_COUNT = 50


def _parse_seat_number(s: str | None) -> int:
    """Parse a seat number string from seat_from/seat_to.

    Returns -1 (sentinel) for missing, "*" wildcard, or non-numeric values.
    Caps at 999.
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return _UNKNOWN_SEAT
    s_str = str(s).strip()
    if not s_str or s_str == "*":
        return _UNKNOWN_SEAT
    try:
        n = int(s_str)
    except (ValueError, TypeError):
        return _UNKNOWN_SEAT
    if n < 0:
        return _UNKNOWN_SEAT
    return min(n, _MAX_SEAT_NUMBER)


def _row_bucket(row: str | None) -> str:
    """Bin a row string into 5 levels via section_encoding._row_quality.

    front:    quality < 0.20  (rows A-J / numeric 1-10)
    mid:      0.20 <= quality < 0.50  (rows K-Y / 11-25)
    back:     quality >= 0.50  (rows Z+ / 26+)
    ga:       row indicates GA / standing
    unknown:  row missing or unparseable
    """
    if row is None or (isinstance(row, float) and np.isnan(row)):
        return "unknown"
    r = str(row).strip().upper()
    if not r or r in ("N/A", "-"):
        return "unknown"
    if r in ("GA", "GENERAL", "STANDING"):
        return "ga"
    q = _row_quality(r)
    if q < 0.20:
        return "front"
    if q < 0.50:
        return "mid"
    return "back"


_ROW_BUCKET_ENCODING = {
    "front": 0,
    "mid": 1,
    "back": 2,
    "ga": 3,
    "unknown": 4,
}


class ListingStructuralFeatureExtractor(FeatureExtractor):
    """Per-listing structural features + Bayesian-smoothed (e, s, r) encoder.

    See module docstring for design rationale.
    """

    def __init__(self) -> None:
        # (event_id, section, row_bucket) -> {"mean_smoothed": float, "count": int}
        self._group_stats: dict[tuple[str, str, str], dict[str, float]] = {}
        # (event_id, section, row_bucket) -> sum of training prices (for LOO)
        self._group_price_sums: dict[tuple[str, str, str], float] = {}
        # (event_id, section, row_bucket) -> count of training prices (for LOO)
        self._group_price_counts: dict[tuple[str, str, str], int] = {}
        # (event_id, section, row_bucket) -> set of integer-cents prices
        # (LOO guard: only train rows match; survives Parquet float round-trip)
        self._train_group_prices: dict[tuple[str, str, str], set[int]] = {}
        # (event_id, section) -> smoothed mean (internal prior)
        self._event_section_prior: dict[tuple[str, str], float] = {}
        # event_id -> raw mean (fallback)
        self._event_mean: dict[str, float] = {}
        # Global fallback
        self._global_mean: float = 150.0
        # (event_id, row_bucket) -> distinct section count
        self._row_bucket_section_count: dict[tuple[str, str], int] = {}
        self._fitted = False

    @property
    def feature_names(self) -> list[str]:
        return [
            "seat_number",
            "seat_span",
            "is_low_seat_number",
            "is_unknown_seat",
            "row_bucket_encoded",
            "event_section_row_median_price",
            "event_section_row_listing_count",
            "row_bucket_section_count",
        ]

    def fit(self, df: pd.DataFrame) -> ListingStructuralFeatureExtractor:
        """Fit on training data: compute group stats and section prior."""
        if "event_id" not in df.columns or "listing_price" not in df.columns:
            self._fitted = True
            return self

        work = df.copy()
        prices = work["listing_price"].dropna()
        self._global_mean = float(prices.mean()) if len(prices) > 0 else 150.0

        # Per-event raw mean (fallback)
        self._event_mean = {}
        for event_id, group in work.groupby("event_id"):
            grp = group["listing_price"].dropna()
            if len(grp) > 0:
                self._event_mean[str(event_id)] = float(grp.mean())

        has_section = "section" in work.columns

        # Internal (event, section) prior: smoothed mean toward event mean
        # μ̂_{e,s} = (n_{e,s} · mean_{e,s} + 20 · μ_e) / (n_{e,s} + 20)
        self._event_section_prior = {}
        if has_section:
            m_es = _SECTION_PRIOR_SMOOTHING_FACTOR
            for (event_id, section), group in work.groupby(["event_id", "section"]):
                grp = group["listing_price"].dropna()
                n = len(grp)
                if n == 0:
                    continue
                event_id_str = str(event_id)
                event_mean = self._event_mean.get(event_id_str, self._global_mean)
                smoothed = (n * float(grp.mean()) + m_es * event_mean) / (n + m_es)
                self._event_section_prior[(event_id_str, str(section))] = smoothed

        # Compute row_bucket column
        if "row" in work.columns:
            work["_row_bucket"] = work["row"].apply(_row_bucket)
        else:
            work["_row_bucket"] = "unknown"

        # Per-(event, section, row_bucket) stats
        self._group_stats = {}
        self._group_price_sums = {}
        self._group_price_counts = {}
        self._train_group_prices = {}

        if has_section:
            m = _GROUP_SMOOTHING_FACTOR
            for (event_id, section, row_bucket), group in work.groupby(
                ["event_id", "section", "_row_bucket"]
            ):
                grp_prices = group["listing_price"].dropna()
                n = len(grp_prices)
                if n == 0:
                    continue
                event_id_str = str(event_id)
                section_str = str(section)
                row_bucket_str = str(row_bucket)
                key = (event_id_str, section_str, row_bucket_str)
                prior = self._event_section_prior.get(
                    (event_id_str, section_str),
                    self._event_mean.get(event_id_str, self._global_mean),
                )
                group_mean = float(grp_prices.mean())
                smoothed = (n * group_mean + m * prior) / (n + m)
                self._group_stats[key] = {
                    "mean_smoothed": smoothed,
                    "count": float(n),
                }
                self._group_price_sums[key] = float(grp_prices.sum())
                self._group_price_counts[key] = n
                # Integer-cents set for LOO guard (survives Parquet float round-trip)
                self._train_group_prices[key] = {int(round(p * 100)) for p in grp_prices.tolist()}

        # Per-(event, row_bucket) distinct section count
        self._row_bucket_section_count = {}
        if has_section:
            for (event_id, row_bucket), group in work.groupby(["event_id", "_row_bucket"]):
                self._row_bucket_section_count[(str(event_id), str(row_bucket))] = min(
                    int(group["section"].nunique()), _MAX_SECTION_COUNT
                )

        self._fitted = True
        return self

    def extract(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """Extract 8 listing-structural features for each row.

        Args:
            df: Input DataFrame.
            is_train: When True, the LOO branch is enabled — for training
                rows whose price is in the train set, the encoder uses the
                leave-one-out mean smoothed toward the (event, section)
                prior. When False (default — val/test/inference), the LOO
                branch is disabled regardless of price collisions, so val
                rows that happen to share a price-cents value with a
                training listing in the same group do not trigger the
                LOO formula. This avoids a price-collision-proxy bias in
                seen-event MAE measurement.
        """
        n_rows = len(df)
        result = pd.DataFrame(index=df.index)

        # Quick path columns (vectorizable)
        if "seat_from" in df.columns:
            seat_numbers = df["seat_from"].apply(_parse_seat_number).to_numpy()
            is_unknown_seat = (seat_numbers == _UNKNOWN_SEAT).astype(int)
        else:
            seat_numbers = np.full(n_rows, _UNKNOWN_SEAT, dtype=np.int64)
            is_unknown_seat = np.ones(n_rows, dtype=int)

        if "seat_to" in df.columns and "seat_from" in df.columns:
            seat_to_nums = df["seat_to"].apply(_parse_seat_number).to_numpy()
            # span = seat_to - seat_from + 1 when both known; else 1
            span = np.where(
                (seat_numbers != _UNKNOWN_SEAT) & (seat_to_nums != _UNKNOWN_SEAT),
                np.clip(seat_to_nums - seat_numbers + 1, 1, _MAX_SEAT_SPAN),
                1,
            ).astype(np.int64)
        else:
            span = np.ones(n_rows, dtype=np.int64)

        is_low_seat = np.where(
            (seat_numbers != _UNKNOWN_SEAT) & (seat_numbers <= _AISLE_THRESHOLD), 1, 0
        ).astype(int)

        # Replace -1 sentinel with 0 for downstream model consumers (negative
        # values would surprise tree splits less than -1 but we keep it explicit).
        seat_numbers_out = np.where(seat_numbers == _UNKNOWN_SEAT, 0, seat_numbers)

        # Row buckets (slow path — per-row apply)
        if "row" in df.columns:
            row_buckets = df["row"].apply(_row_bucket).to_numpy()
        else:
            row_buckets = np.full(n_rows, "unknown", dtype=object)
        row_bucket_encoded = np.array(
            [_ROW_BUCKET_ENCODING[rb] for rb in row_buckets], dtype=np.int64
        )

        # Group-level encoder + counts (per-row resolution with LOO guard)
        event_section_row_median_prices = np.zeros(n_rows, dtype=np.float64)
        event_section_row_listing_counts = np.zeros(n_rows, dtype=np.float64)
        row_bucket_section_counts = np.zeros(n_rows, dtype=np.float64)

        m = _GROUP_SMOOTHING_FACTOR

        event_ids = df["event_id"].astype(str).to_numpy() if "event_id" in df.columns else None
        sections = df["section"].astype(str).to_numpy() if "section" in df.columns else None
        prices = (
            df["listing_price"].to_numpy(dtype=np.float64)
            if "listing_price" in df.columns
            else np.full(n_rows, np.nan, dtype=np.float64)
        )

        for i in range(n_rows):
            event_id = event_ids[i] if event_ids is not None else ""
            section = sections[i] if sections is not None else ""
            row_bucket = row_buckets[i]

            # Resolve smoothed encoder via fallback chain.
            # Prior at the (event, section) level — used both for fallback
            # and as the LOO smoothing target.
            section_prior = self._event_section_prior.get(
                (event_id, section),
                self._event_mean.get(event_id, self._global_mean),
            )

            key = (event_id, section, row_bucket)
            stats = self._group_stats.get(key)

            if stats is None:
                # Fallback: μ̂_{e,s} → μ_e → μ_global
                ev_section_row_mean = section_prior
                ev_section_row_count = 0.0
            else:
                p = prices[i]
                # LOO branch only fires for training rows (is_train=True). Val/test
                # rows whose prices happen to collide with train set prices in
                # integer-cents do NOT trigger LOO — that would compute a bogus
                # "leave-one-train-row-out" estimate biased toward the train mean.
                use_loo = (
                    is_train
                    and not np.isnan(p)
                    and int(round(float(p) * 100)) in self._train_group_prices.get(key, set())
                )
                if use_loo:
                    # loo_mean = (sum - y_i) / (n - 1); re-smooth toward (e, s) prior.
                    n_loo = self._group_price_counts[key]
                    if n_loo == 1:
                        # Removing self leaves no signal — fall back to section prior.
                        ev_section_row_mean = section_prior
                    else:
                        loo_mean = (self._group_price_sums[key] - float(p)) / (n_loo - 1)
                        ev_section_row_mean = ((n_loo - 1) * loo_mean + m * section_prior) / (
                            n_loo - 1 + m
                        )
                else:
                    # Non-LOO (val/test/inference): use the smoothed group mean.
                    ev_section_row_mean = stats["mean_smoothed"]
                ev_section_row_count = stats["count"]

            event_section_row_median_prices[i] = ev_section_row_mean
            event_section_row_listing_counts[i] = float(np.log1p(ev_section_row_count))

            row_bucket_section_counts[i] = float(
                self._row_bucket_section_count.get((event_id, row_bucket), 0)
            )

        result["seat_number"] = seat_numbers_out.astype(np.int64)
        result["seat_span"] = span
        result["is_low_seat_number"] = is_low_seat
        result["is_unknown_seat"] = is_unknown_seat
        result["row_bucket_encoded"] = row_bucket_encoded
        result["event_section_row_median_price"] = event_section_row_median_prices
        result["event_section_row_listing_count"] = event_section_row_listing_counts
        result["row_bucket_section_count"] = row_bucket_section_counts

        return result

    def get_params(self) -> dict[str, Any]:
        """Return parameters for serialization."""
        return {"fitted": self._fitted}
