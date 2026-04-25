"""Cold-start sibling-event aggregate features for temporally unseen events.

Three Bayesian-smoothed log-price means keyed on sibling events (same group,
different event_id) to provide signal for events with no direct training history.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ticket_price_predictor.ml.features.base import FeatureExtractor
from ticket_price_predictor.ml.features.geo_mapping import get_region_key
from ticket_price_predictor.ml.features.section_encoding import _section_hash

logger = logging.getLogger(__name__)

# Smoothing factors per group (match existing codebase convention)
_SMOOTHING_ARQ = 75  # (artist, region, year_quarter)
_SMOOTHING_VDOW = 100  # (venue, day_of_week)
_SMOOTHING_SZ = 50  # (section_type_hash, zone)

# Sentinel used when a row has no section/zone data
_UNKNOWN_ZONE = "unknown"


def _year_quarter(dt: pd.Timestamp) -> str:
    """Return 'YYYYQn' string for a timestamp."""
    q = ((dt.month - 1) // 3) + 1
    return f"{dt.year}Q{q}"


def _safe_str(val: object) -> str:
    """Return str(val) or '' for None/NaN."""
    if val is None:
        return ""
    if isinstance(val, float) and np.isnan(val):
        return ""
    return str(val)


# ---------------------------------------------------------------------------
# Per-group accumulated statistics (count and log-price sum) stored during fit.
# Key is the group tuple; value is (total_count, total_log_sum).
# We also store per-event contributions so LOO exclusion is O(1) at extract time.
# ---------------------------------------------------------------------------

_GroupStats = dict[tuple[Any, ...], tuple[int, float]]
_EventContribs = dict[tuple[Any, ...], dict[str, tuple[int, float]]]


class ColdStartFeatureExtractor(FeatureExtractor):
    """Bayesian-smoothed sibling-event log-price means for cold-start events.

    Three partition keys generalise to temporally unseen events:
      - ARQ: (artist, region, year_quarter) — same artist priced in same
             region at a similar time of year, different event.
      - VDOW: (venue, day_of_week) — same venue on same weekday, any artist.
      - SZ: (section_type_hash, zone) — same section archetype across the
            full catalog.

    Each feature is the Bayesian-smoothed mean of log1p(price) over training
    rows from sibling events (same group key, different event_id). A fallback
    chain ensures no NaN output even for fully unseen combinations.

    Integer-cents LOO guard matches event_pricing.py:141-143: price membership
    is tested via `int(round(price * 100))` to survive Parquet float round-trips.
    """

    SMOOTHING_ARQ: int = _SMOOTHING_ARQ
    SMOOTHING_VDOW: int = _SMOOTHING_VDOW
    SMOOTHING_SZ: int = _SMOOTHING_SZ

    def __init__(self) -> None:
        self._global_mean: float = 0.0

        # Total (count, log_sum) per group key — full training set.
        # Primary group stats: 3-/2-tuple keys.
        self._arq_stats: _GroupStats = {}  # (artist, region, quarter)
        self._arq_ar_stats: _GroupStats = {}  # (artist, region) — fallback level 1
        self._arq_a_stats: _GroupStats = {}  # (artist,) — fallback level 2
        self._vdow_stats: _GroupStats = {}  # (venue, dow)
        self._vdow_v_stats: _GroupStats = {}  # (venue,) — fallback level 1
        self._sz_stats: _GroupStats = {}  # (section_hash, zone)
        self._sz_z_stats: _GroupStats = {}  # (zone,) — fallback level 1

        # Per-event contributions: group_key -> event_id -> (count, log_sum).
        # Used to subtract the target event's own rows for LOO.
        self._arq_event: _EventContribs = {}
        self._arq_ar_event: _EventContribs = {}
        self._arq_a_event: _EventContribs = {}
        self._vdow_event: _EventContribs = {}
        self._vdow_v_event: _EventContribs = {}
        self._sz_event: _EventContribs = {}
        self._sz_z_event: _EventContribs = {}

        # Integer-cents price sets per event (LOO guard, matches event_pricing.py:141-143).
        self._train_event_prices: dict[str, set[int]] = {}

        self._fitted = False

    # ------------------------------------------------------------------
    # FeatureExtractor interface
    # ------------------------------------------------------------------

    @property
    def feature_names(self) -> list[str]:
        return [
            "coldstart_arq_logmean",
            "coldstart_vdow_logmean",
            "coldstart_sz_logmean",
            "coldstart_arq_support",
            "coldstart_vdow_support",
            "coldstart_sz_support",
        ]

    def fit(self, df: pd.DataFrame) -> ColdStartFeatureExtractor:
        """Learn group statistics from training data.

        Populates three group-stats dicts and per-event contribution dicts
        for O(1) LOO exclusion at extract() time.

        Args:
            df: Training DataFrame. Required columns: listing_price, event_id.
                Optional but important: artist_or_team, city, event_datetime,
                venue_name, section, seat_zone.
        """
        if "listing_price" not in df.columns or "event_id" not in df.columns:
            logger.warning(
                "ColdStartFeatureExtractor.fit(): missing listing_price or event_id — "
                "using global-mean fallback only."
            )
            self._global_mean = np.log1p(150.0)
            self._fitted = True
            return self

        df = df.copy()
        prices = df["listing_price"].dropna()
        self._global_mean = float(np.log1p(prices).mean()) if len(prices) > 0 else np.log1p(150.0)

        # Build integer-cents sets per event for LOO guard.
        self._train_event_prices = {}
        for event_id, grp in df.groupby("event_id"):
            grp_prices = grp["listing_price"].dropna()
            self._train_event_prices[str(event_id)] = {
                int(round(p * 100)) for p in grp_prices.tolist()
            }

        # Derive grouping columns, filling missing with sentinel values.
        df["_event_id"] = df["event_id"].astype(str)
        df["_log_price"] = np.log1p(df["listing_price"].fillna(0.0))

        # --- ARQ keys ---
        df["_artist"] = (
            df["artist_or_team"].apply(_safe_str) if "artist_or_team" in df.columns else ""
        )
        if "city" in df.columns:
            df["_region"] = df["city"].apply(
                lambda c: get_region_key(_safe_str(c)) if _safe_str(c) else "US:unknown"
            )
        else:
            df["_region"] = "US:unknown"
        if "event_datetime" in df.columns:
            df["_quarter"] = pd.to_datetime(df["event_datetime"]).apply(
                lambda dt: _year_quarter(dt) if not pd.isnull(dt) else "unknown"
            )
        else:
            df["_quarter"] = "unknown"

        # --- VDOW keys ---
        df["_venue"] = df["venue_name"].apply(_safe_str) if "venue_name" in df.columns else ""
        if "event_datetime" in df.columns:
            df["_dow"] = pd.to_datetime(df["event_datetime"]).apply(
                lambda dt: str(int(dt.dayofweek)) if not pd.isnull(dt) else "unknown"
            )
        else:
            df["_dow"] = "unknown"

        # --- SZ keys ---
        df["_section_hash"] = (
            df["section"].apply(lambda s: str(_section_hash(_safe_str(s))))
            if "section" in df.columns
            else "0"
        )
        df["_zone"] = (
            df["seat_zone"].apply(_safe_str) if "seat_zone" in df.columns else _UNKNOWN_ZONE
        )

        # Accumulate stats for primary keys and all fallback levels.
        self._arq_stats, self._arq_event = _accumulate(
            df, ("_artist", "_region", "_quarter"), "_event_id", "_log_price"
        )
        self._arq_ar_stats, self._arq_ar_event = _accumulate(
            df, ("_artist", "_region"), "_event_id", "_log_price"
        )
        self._arq_a_stats, self._arq_a_event = _accumulate(
            df, ("_artist",), "_event_id", "_log_price"
        )
        self._vdow_stats, self._vdow_event = _accumulate(
            df, ("_venue", "_dow"), "_event_id", "_log_price"
        )
        self._vdow_v_stats, self._vdow_v_event = _accumulate(
            df, ("_venue",), "_event_id", "_log_price"
        )
        self._sz_stats, self._sz_event = _accumulate(
            df, ("_section_hash", "_zone"), "_event_id", "_log_price"
        )
        self._sz_z_stats, self._sz_z_event = _accumulate(df, ("_zone",), "_event_id", "_log_price")

        self._fitted = True
        return self

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute cold-start features for each row.

        For training-set rows (event_id seen during fit AND integer-cents price
        in the training price set for that event), applies LOO exclusion to
        remove the target event's contribution before computing the group mean.
        Walks fallback chains so output is never NaN.

        Args:
            df: Input DataFrame (train, val, or test split).

        Returns:
            DataFrame with six feature columns.
        """
        result = pd.DataFrame(index=df.index)
        g = self._global_mean

        arq_logmeans: list[float] = []
        vdow_logmeans: list[float] = []
        sz_logmeans: list[float] = []
        arq_supports: list[float] = []
        vdow_supports: list[float] = []
        sz_supports: list[float] = []

        for _, row in df.iterrows():
            event_id = _safe_str(row.get("event_id"))
            price_raw = row.get("listing_price", np.nan)
            price_cents = int(round(float(price_raw) * 100)) if not pd.isnull(price_raw) else -1
            # A row triggers LOO exclusion only if it was seen during fit
            # (its integer-cents price is in the training price set for that event).
            is_train_row = (
                event_id in self._train_event_prices
                and price_cents in self._train_event_prices[event_id]
            )

            # --- ARQ ---
            artist = _safe_str(row.get("artist_or_team")) if "artist_or_team" in df.columns else ""
            city_raw = _safe_str(row.get("city")) if "city" in df.columns else ""
            region = get_region_key(city_raw) if city_raw else "US:unknown"
            dt_raw = row.get("event_datetime")
            quarter = (
                _year_quarter(pd.Timestamp(dt_raw))
                if dt_raw is not None and not pd.isnull(dt_raw)
                else "unknown"
            )
            arq_key = (artist, region, quarter)
            arq_n, arq_mu = _resolve_loo(
                arq_key,
                event_id,
                is_train_row,
                self._arq_stats,
                self._arq_event,
            )
            arq_supports.append(float(self._arq_stats.get(arq_key, (0, 0.0))[0]))

            if arq_n == 0:
                # Fallback 1: (artist, region)
                fb1 = (artist, region)
                arq_n, arq_mu = _resolve_loo(
                    fb1,
                    event_id,
                    is_train_row,
                    self._arq_ar_stats,
                    self._arq_ar_event,
                )
            if arq_n == 0:
                # Fallback 2: (artist,)
                fb2 = (artist,)
                arq_n, arq_mu = _resolve_loo(
                    fb2,
                    event_id,
                    is_train_row,
                    self._arq_a_stats,
                    self._arq_a_event,
                )
            arq_logmeans.append(_smooth(arq_n, arq_mu, g, self.SMOOTHING_ARQ))

            # --- VDOW ---
            venue = _safe_str(row.get("venue_name")) if "venue_name" in df.columns else ""
            dt_raw2 = row.get("event_datetime")
            dow = (
                str(int(pd.Timestamp(dt_raw2).dayofweek))
                if dt_raw2 is not None and not pd.isnull(dt_raw2)
                else "unknown"
            )
            vdow_key = (venue, dow)
            vdow_n, vdow_mu = _resolve_loo(
                vdow_key,
                event_id,
                is_train_row,
                self._vdow_stats,
                self._vdow_event,
            )
            vdow_supports.append(float(self._vdow_stats.get(vdow_key, (0, 0.0))[0]))

            if vdow_n == 0:
                # Fallback 1: (venue,)
                fb_v = (venue,)
                vdow_n, vdow_mu = _resolve_loo(
                    fb_v,
                    event_id,
                    is_train_row,
                    self._vdow_v_stats,
                    self._vdow_v_event,
                )
            vdow_logmeans.append(_smooth(vdow_n, vdow_mu, g, self.SMOOTHING_VDOW))

            # --- SZ ---
            section_raw = _safe_str(row.get("section")) if "section" in df.columns else ""
            section_hash_str = str(_section_hash(section_raw))
            zone = _safe_str(row.get("seat_zone")) if "seat_zone" in df.columns else _UNKNOWN_ZONE
            if not zone:
                zone = _UNKNOWN_ZONE
            sz_key = (section_hash_str, zone)
            sz_n, sz_mu = _resolve_loo(
                sz_key,
                event_id,
                is_train_row,
                self._sz_stats,
                self._sz_event,
            )
            sz_supports.append(float(self._sz_stats.get(sz_key, (0, 0.0))[0]))

            if sz_n == 0:
                # Fallback 1: (zone,)
                fb_z = (zone,)
                sz_n, sz_mu = _resolve_loo(
                    fb_z,
                    event_id,
                    is_train_row,
                    self._sz_z_stats,
                    self._sz_z_event,
                )
            sz_logmeans.append(_smooth(sz_n, sz_mu, g, self.SMOOTHING_SZ))

        result["coldstart_arq_logmean"] = arq_logmeans
        result["coldstart_vdow_logmean"] = vdow_logmeans
        result["coldstart_sz_logmean"] = sz_logmeans
        result["coldstart_arq_support"] = arq_supports
        result["coldstart_vdow_support"] = vdow_supports
        result["coldstart_sz_support"] = sz_supports

        return result

    def get_params(self) -> dict[str, Any]:
        return {"fitted": self._fitted, "global_mean": self._global_mean}


# ---------------------------------------------------------------------------
# Module-level helpers (no class state, easily testable)
# ---------------------------------------------------------------------------


def _accumulate(
    df: pd.DataFrame,
    key_cols: tuple[str, ...],
    event_col: str,
    log_price_col: str,
) -> tuple[_GroupStats, _EventContribs]:
    """Accumulate (count, log_sum) dicts for a set of grouping columns.

    Returns:
        group_stats: key -> (total_count, total_log_sum) over all training rows.
        event_contribs: key -> event_id -> (count, log_sum) for LOO subtraction.
    """
    group_stats: _GroupStats = {}
    event_contribs: _EventContribs = {}

    for row_idx in df.index:
        row = df.loc[row_idx]
        key = tuple(str(row[c]) for c in key_cols)
        eid = str(row[event_col])
        lp = float(row[log_price_col])

        # Accumulate into group total.
        if key in group_stats:
            n, s = group_stats[key]
            group_stats[key] = (n + 1, s + lp)
        else:
            group_stats[key] = (1, lp)

        # Accumulate per-event contribution.
        if key not in event_contribs:
            event_contribs[key] = {}
        if eid in event_contribs[key]:
            en, es = event_contribs[key][eid]
            event_contribs[key][eid] = (en + 1, es + lp)
        else:
            event_contribs[key][eid] = (1, lp)

    return group_stats, event_contribs


def _resolve_loo(
    key: tuple[Any, ...],
    event_id: str,
    is_train_row: bool,
    group_stats: _GroupStats,
    event_contribs: _EventContribs,
) -> tuple[int, float]:
    """Return (sibling_count, sibling_log_mean) with LOO for training rows.

    For training rows, subtracts the target event's contribution from the
    group total so the feature reflects only sibling-event price history.
    For val/test rows, returns the full group statistics (no exclusion needed).

    Returns (0, 0.0) when there are no sibling rows after exclusion.
    """
    if key not in group_stats:
        return 0, 0.0

    total_n, total_s = group_stats[key]

    if is_train_row and key in event_contribs and event_id in event_contribs[key]:
        excl_n, excl_s = event_contribs[key][event_id]
        sibling_n = total_n - excl_n
        sibling_s = total_s - excl_s
    else:
        sibling_n = total_n
        sibling_s = total_s

    if sibling_n <= 0:
        return 0, 0.0

    return sibling_n, sibling_s / sibling_n


def _smooth(n: int, mu: float, global_mean: float, m: int) -> float:
    """Bayesian-smoothed estimate: (n * mu + m * global_mean) / (n + m).

    When n == 0 (no sibling rows), returns global_mean directly.
    """
    if n == 0:
        return global_mean
    return (n * mu + m * global_mean) / (n + m)
