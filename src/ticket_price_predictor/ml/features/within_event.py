"""Within-event dynamics feature extractor.

Caches per-event listing history H[e] at fit() time (training data only).
At extract() time, computes E_i — the causal prefix of that history for
each row — and emits sequence-position and price-deviation features.

Signature deviation from FeatureExtractor.extract(df):
    extract(df, is_train=False)

The extra ``is_train`` parameter controls E_i scoping:
  - is_train=True  → strict-less-than LOO  (E_i = {j ∈ H[e] : t_j < t_i})
  - is_train=False → ≤ t_i prefix          (E_i = {j ∈ H[e] : t_j ≤ t_i})

Calling extract(df) without the flag defaults to is_train=False, preserving
backward compatibility with callers that don't know about the flag.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pandas as pd

from ticket_price_predictor.ml.features.base import FeatureExtractor

# Output column names — no "price"/"avg"/"median" substrings so the trainer's
# log-transform heuristic leaves them untouched.
_FEATURE_NAMES: list[str] = [
    "we_pos_in_sequence",
    "we_time_since_first_hours",
    "we_age_percentile",
    "we_rolling5_dev",
    "we_zone_price_at_t",
    "we_history_support",
]


class WithinEventDynamicsFeatureExtractor(FeatureExtractor):
    """Per-listing within-event dynamics features.

    Fit: builds H[e] — sorted (timestamp, zone, price) triples — from the
    training DataFrame for every event_id.

    Extract: for each row i, scopes E_i to the causal prefix of H[e_i],
    then emits:

      we_pos_in_sequence      |E_i|
      we_time_since_first_hrs t_i − min(t_j) in hours; 0 when |E_i|=0
      we_age_percentile       |E_i| / max(|H[e]|, 1)   — same denominator
                              across splits for seen events
      we_rolling5_median_dev  log(p_i / median(last-5 in E_i)) if |E_i|≥5
      we_zone_price_at_t      log(p_i / median zone-match in E_i) if any
      we_history_support      |H[e_i]|  (raw training-set size for this event)

    Performance: O(log|H|) per row via np.searchsorted on sorted timestamps.
    Zone-match scan is O(k) where k=|E_i| ≤ |H[e]| (typically ≪ 500).
    """

    def __init__(self) -> None:
        # Maps event_id (str) → dict with sorted numpy arrays aligned by index:
        #   timestamps_np : float64 (seconds since epoch, sorted ascending)
        #   prices_np     : float64 (listing prices)
        #   zones_np      : object  (zone/section strings)
        #   size          : int     (|H[e]|, used as age_percentile denominator)
        self._history: dict[str, dict[str, Any]] = {}

    @property
    def feature_names(self) -> list[str]:
        return list(_FEATURE_NAMES)

    def fit(self, df: pd.DataFrame) -> WithinEventDynamicsFeatureExtractor:
        """Cache per-event listing history from training data.

        Only rows with non-null event_id, timestamp, and listing_price are
        included. Rows within an event are sorted by timestamp ascending.
        """
        self._history = {}

        if df.empty or "event_id" not in df.columns:
            return self

        required = {"event_id", "timestamp", "listing_price"}
        missing = required - set(df.columns)
        if missing:
            return self

        work = df[["event_id", "timestamp", "listing_price"]].copy()

        # Carry zone column when available (falls back to empty strings)
        zone_col = next((c for c in ("seat_zone", "section", "zone") if c in df.columns), None)
        if zone_col:
            work["_zone"] = df[zone_col].fillna("").astype(str)
        else:
            work["_zone"] = ""

        work = work.dropna(subset=["event_id", "timestamp", "listing_price"])
        work["_ts_f"] = pd.to_datetime(work["timestamp"], utc=True).astype(np.int64) / 1e9

        for event_id, grp in work.groupby("event_id", sort=False):
            grp_sorted = grp.sort_values("_ts_f")
            self._history[str(event_id)] = {
                "timestamps_np": grp_sorted["_ts_f"].to_numpy(dtype=np.float64),
                "prices_np": grp_sorted["listing_price"].to_numpy(dtype=np.float64),
                "zones_np": grp_sorted["_zone"].to_numpy(dtype=object),
                "size": len(grp_sorted),
            }

        return self

    def extract(
        self,
        df: pd.DataFrame,
        is_train: bool = False,
    ) -> pd.DataFrame:
        """Extract within-event dynamics features.

        Args:
            df: Input DataFrame (one split — train, val, or test).
            is_train: When True, uses strict-less-than E_i scoping (LOO).
                      When False (default), uses ≤ t_i scoping.

        Returns:
            DataFrame with ``we_*`` feature columns, same row count as df.
        """
        n = len(df)
        out: dict[str, np.ndarray[Any, Any]] = {
            "we_pos_in_sequence": np.zeros(n, dtype=np.float64),
            "we_time_since_first_hours": np.zeros(n, dtype=np.float64),
            "we_age_percentile": np.zeros(n, dtype=np.float64),
            "we_rolling5_dev": np.zeros(n, dtype=np.float64),
            "we_zone_price_at_t": np.zeros(n, dtype=np.float64),
            "we_history_support": np.zeros(n, dtype=np.float64),
        }

        if (
            df.empty
            or not self._history
            or "event_id" not in df.columns
            or "timestamp" not in df.columns
            or "listing_price" not in df.columns
        ):
            return pd.DataFrame(out, index=df.index)

        zone_col = next((c for c in ("seat_zone", "section", "zone") if c in df.columns), None)

        ts_seconds = pd.to_datetime(df["timestamp"], utc=True).astype(np.int64).to_numpy() / 1e9
        prices = df["listing_price"].to_numpy(dtype=np.float64)
        event_ids = df["event_id"].astype(str).to_numpy(dtype=object)
        zones = (
            df[zone_col].fillna("").astype(str).to_numpy(dtype=object)
            if zone_col
            else np.full(n, "", dtype=object)
        )

        for i in range(n):
            eid = event_ids[i]
            hist = self._history.get(eid)
            if hist is None:
                continue  # unseen event — all features remain 0

            h_ts: np.ndarray[Any, Any] = hist["timestamps_np"]
            h_px: np.ndarray[Any, Any] = hist["prices_np"]
            h_zn: np.ndarray[Any, Any] = hist["zones_np"]
            h_size: int = hist["size"]

            t_i = ts_seconds[i]

            # E_i scoping: train → strict-less-than; val/test → ≤ t_i
            if is_train:
                k = int(np.searchsorted(h_ts, t_i, side="left"))
            else:
                k = int(np.searchsorted(h_ts, t_i, side="right"))

            out["we_history_support"][i] = float(h_size)
            out["we_pos_in_sequence"][i] = float(k)
            out["we_age_percentile"][i] = float(k) / max(h_size, 1)

            if k == 0:
                continue  # no history before t_i — remaining features stay 0

            # time_since_first_hours: t_i − earliest timestamp in H[e]
            # Denominator is |H[e]| (not |E_i|) for consistent age_percentile;
            # time_since_first uses the global first listing for this event.
            first_ts = h_ts[0]
            out["we_time_since_first_hours"][i] = (t_i - first_ts) / 3600.0

            p_i = prices[i]

            # rolling5_median_dev: log(p_i / median(last-5 in E_i)) if k≥5
            if k >= 5:
                tail = h_px[max(0, k - 5) : k]
                tail_median = float(np.median(tail))
                if tail_median > 0 and p_i > 0:
                    out["we_rolling5_dev"][i] = np.log(p_i / tail_median)

            # zone_price_at_t: log(p_i / median(same-zone prices in E_i))
            z_i = zones[i]
            if z_i:
                mask = h_zn[:k] == z_i
                zone_px = h_px[:k][mask]
                if len(zone_px) > 0:
                    zone_med = float(np.median(zone_px))
                    if zone_med > 0 and p_i > 0:
                        out["we_zone_price_at_t"][i] = np.log(p_i / zone_med)

        return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------------------
# Introspection helper used by FeaturePipeline.transform_with_train_flag()
# ---------------------------------------------------------------------------


def _accepts_is_train(extractor: FeatureExtractor) -> bool:
    """Return True if extractor.extract() accepts an ``is_train`` keyword."""
    sig = inspect.signature(extractor.extract)
    return "is_train" in sig.parameters
