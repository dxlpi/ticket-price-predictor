"""Event-level pricing features computed from training data."""

import logging
from typing import Any

import numpy as np
import pandas as pd

from ticket_price_predictor.ml.features.base import FeatureExtractor
from ticket_price_predictor.normalization.seat_zones import SeatZoneMapper

logger = logging.getLogger(__name__)

_ZONE_MAPPER = SeatZoneMapper()


class EventPricingFeatureExtractor(FeatureExtractor):
    """Extract event-level pricing context features.

    Computes per-event and per-(event, zone) price statistics from
    training data. Uses a fallback chain for unseen events:
    event_zone → event → artist_zone → artist → global.

    These features capture "what do tickets cost at THIS event?"
    which is the strongest single predictor (r=0.789).
    """

    SMOOTHING_FACTOR = 20  # Bayesian smoothing for small-sample events

    def __init__(self) -> None:
        self._event_stats: dict[
            str, dict[str, float]
        ] = {}  # event_id -> {median, mean, std, count}
        self._event_zone_stats: dict[
            tuple[str, str], dict[str, float]
        ] = {}  # (event_id, zone) -> {median, count}
        self._event_section_stats: dict[
            tuple[str, str], dict[str, float]
        ] = {}  # (event_id, section) -> {median, count}
        self._artist_stats: dict[str, dict[str, float]] = {}  # artist -> {median}
        self._artist_zone_stats: dict[
            tuple[str, str], dict[str, float]
        ] = {}  # (artist, zone) -> {median}
        self._global_stats: dict[str, float] = {}
        # LOO encoding: per-event sums and counts for leave-one-out mean computation.
        # Populated during fit() for training rows; empty during inference.
        self._event_price_sums: dict[str, float] = {}
        self._event_price_counts: dict[str, int] = {}
        # Set of prices per event seen during fit(). Used in extract() to guard
        # against val/test rows from shared events incorrectly triggering LOO.
        self._train_event_prices: dict[str, set[float]] = {}
        self._fitted = False

    @property
    def feature_names(self) -> list[str]:
        """Return list of feature names this extractor produces."""
        return [
            "event_median_price",
            "event_zone_median_price",
            "event_listing_count",
            "event_price_cv",
            "event_zone_price_ratio",
        ]

    def fit(self, df: pd.DataFrame) -> "EventPricingFeatureExtractor":
        """Fit on training data to compute event and zone price statistics.

        Args:
            df: Training DataFrame with columns: event_id, listing_price,
                and optionally section, artist_or_team.

        Returns:
            self
        """
        if "event_id" not in df.columns or "listing_price" not in df.columns:
            logger.warning(
                "EventPricingFeatureExtractor.fit(): missing required columns "
                "(event_id, listing_price). Using global defaults only."
            )
            self._global_stats = {"median": 150.0, "mean": 150.0, "std": 50.0}
            self._fitted = True
            return self

        df = df.copy()
        prices = df["listing_price"].dropna()
        global_median = float(prices.median()) if len(prices) > 0 else 150.0
        global_mean = float(prices.mean()) if len(prices) > 0 else 150.0
        global_std = float(prices.std()) if len(prices) > 0 else 50.0
        self._global_stats = {
            "median": global_median,
            "mean": global_mean,
            "std": global_std,
        }

        m = self.SMOOTHING_FACTOR

        # Map sections to zones if section column is present
        has_section = "section" in df.columns
        if has_section:
            df["_zone"] = (
                df["section"]
                .fillna("")
                .apply(
                    lambda s: _ZONE_MAPPER.normalize_zone_name(str(s)).value if s else "upper_tier"
                )
            )

        # Per-event stats (smoothed toward global median)
        self._event_stats = {}
        self._event_price_sums = {}
        self._event_price_counts = {}
        self._train_event_prices = {}
        for event_id, group in df.groupby("event_id"):
            grp_prices = group["listing_price"].dropna()
            n = len(grp_prices)
            if n == 0:
                continue
            group_median = float(grp_prices.median())
            group_mean = float(grp_prices.mean())
            group_std = float(grp_prices.std()) if n > 1 else 0.0
            smoothed_median = (n * group_median + m * global_median) / (n + m)
            event_id_str = str(event_id)
            self._train_event_prices[event_id_str] = set(grp_prices.tolist())
            self._event_stats[event_id_str] = {
                "median": smoothed_median,
                "mean": group_mean,
                "std": group_std,
                "count": float(n),
            }
            # LOO: raw sum and count for leave-one-out mean computation in extract()
            self._event_price_sums[event_id_str] = float(grp_prices.sum())
            self._event_price_counts[event_id_str] = n

        # Per-(event, zone) stats (smoothed toward event median)
        self._event_zone_stats = {}
        if has_section:
            for (event_id, zone), group in df.groupby(["event_id", "_zone"]):
                grp_prices = group["listing_price"].dropna()
                n = len(grp_prices)
                if n == 0:
                    continue
                group_median = float(grp_prices.median())
                event_id_str = str(event_id)
                event_median = self._event_stats.get(event_id_str, {}).get("median", global_median)
                smoothed_median = (n * group_median + m * event_median) / (n + m)
                self._event_zone_stats[(event_id_str, str(zone))] = {
                    "median": smoothed_median,
                    "count": float(n),
                }

        # Per-(event, section) stats (smoothed toward event_zone median)
        # Sections explain 49.8% of within-event variance vs zone's 20.2%.
        self._event_section_stats = {}
        if has_section:
            for (event_id, section), group in df.groupby(["event_id", "section"]):
                grp_prices = group["listing_price"].dropna()
                n = len(grp_prices)
                if n == 0:
                    continue
                group_median = float(grp_prices.median())
                event_id_str = str(event_id)
                section_str = str(section)
                # Determine zone for this section to use as smoothing prior
                zone_str = _ZONE_MAPPER.normalize_zone_name(section_str).value
                ez_key = (event_id_str, zone_str)
                prior = self._event_zone_stats.get(ez_key, {}).get(
                    "median",
                    self._event_stats.get(event_id_str, {}).get("median", global_median),
                )
                smoothed_median = (n * group_median + m * prior) / (n + m)
                self._event_section_stats[(event_id_str, section_str)] = {
                    "median": smoothed_median,
                    "count": float(n),
                }

        # Per-artist stats (fallback for unseen events)
        self._artist_stats = {}
        if "artist_or_team" in df.columns:
            for artist, group in df.groupby("artist_or_team"):
                grp_prices = group["listing_price"].dropna()
                n = len(grp_prices)
                if n == 0:
                    continue
                group_median = float(grp_prices.median())
                group_mean = float(grp_prices.mean())
                group_std = float(grp_prices.std()) if n > 1 else 0.0
                smoothed_median = (n * group_median + m * global_median) / (n + m)
                self._artist_stats[str(artist)] = {
                    "median": smoothed_median,
                    "mean": group_mean,
                    "std": group_std,
                    "count": float(n),
                }

        # Per-(artist, zone) stats
        self._artist_zone_stats = {}
        if "artist_or_team" in df.columns and has_section:
            for (artist, zone), group in df.groupby(["artist_or_team", "_zone"]):
                grp_prices = group["listing_price"].dropna()
                n = len(grp_prices)
                if n == 0:
                    continue
                group_median = float(grp_prices.median())
                artist_str = str(artist)
                artist_median = self._artist_stats.get(artist_str, {}).get("median", global_median)
                smoothed_median = (n * group_median + m * artist_median) / (n + m)
                self._artist_zone_stats[(artist_str, str(zone))] = {
                    "median": smoothed_median,
                    "count": float(n),
                }

        self._fitted = True
        return self

    def _get_zone_str(self, row: pd.Series) -> str:
        """Map a row's section to a zone string."""
        section = row.get("section", None)
        if section is None or (isinstance(section, float) and np.isnan(section)):
            return "upper_tier"
        return _ZONE_MAPPER.normalize_zone_name(str(section)).value

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract event-level pricing features for each row.

        Args:
            df: Input DataFrame with raw data

        Returns:
            DataFrame with 5 event pricing feature columns
        """
        result = pd.DataFrame(index=df.index)

        global_median = self._global_stats.get("median", 150.0)
        global_mean = self._global_stats.get("mean", 150.0)
        global_std = self._global_stats.get("std", 50.0)

        event_median_prices = []
        event_zone_median_prices = []
        event_listing_counts = []
        event_price_cvs = []
        event_zone_price_ratios = []

        for _, row in df.iterrows():
            event_id = str(row.get("event_id", "")) if row.get("event_id") is not None else ""
            artist = (
                str(row.get("artist_or_team", "")) if row.get("artist_or_team") is not None else ""
            )
            zone = self._get_zone_str(row)

            # Resolve event-level stats with fallback: event → artist → global
            event_stats = self._event_stats.get(event_id)
            if event_stats is not None:
                ev_median = event_stats["median"]
                ev_mean = event_stats["mean"]
                ev_std = event_stats["std"]
                ev_count = event_stats["count"]
            else:
                artist_stats = self._artist_stats.get(artist)
                if artist_stats is not None:
                    ev_median = artist_stats["median"]
                    ev_mean = artist_stats["mean"]
                    ev_std = artist_stats["std"]
                    ev_count = artist_stats["count"]
                else:
                    ev_median = global_median
                    ev_mean = global_mean
                    ev_std = global_std
                    ev_count = 1.0

            # LOO: for training-set rows, replace event_median_price with a
            # leave-one-out mean to remove self-inclusion leakage. Uses mean
            # (not median) because means are sum-decomposable: LOO_mean =
            # (sum - y_i) / (n - 1). Val/test/inference rows are unaffected.
            price = row.get("listing_price", np.nan)
            if (
                event_id in self._event_price_sums
                and not pd.isna(price)
                and float(price) in self._train_event_prices.get(event_id, set())
            ):
                total = self._event_price_sums[event_id]
                n_loo = self._event_price_counts[event_id]
                if n_loo > 1:
                    loo_mean = (total - float(price)) / (n_loo - 1)
                    m_factor = self.SMOOTHING_FACTOR
                    global_mean_val = self._global_stats.get("mean", global_mean)
                    ev_median = (((n_loo - 1) * loo_mean) + (m_factor * global_mean_val)) / (
                        n_loo - 1 + m_factor
                    )
                else:
                    ev_median = self._global_stats.get("mean", global_mean)

            # Resolve zone-level median with fallback: event_zone → artist_zone → event → global
            zone_stats = self._event_zone_stats.get((event_id, zone))
            if zone_stats is not None:
                ez_median = zone_stats["median"]
            else:
                artist_zone_stats = self._artist_zone_stats.get((artist, zone))
                if artist_zone_stats is not None:
                    ez_median = artist_zone_stats["median"]
                else:
                    ez_median = ev_median

            # Compute CV (coefficient of variation): std / mean, clamped to [0, 3]
            cv = float(np.clip(ev_std / ev_mean, 0.0, 3.0)) if ev_mean > 0 else 0.0

            # Zone price ratio: zone_median / event_median, clamped to [0.1, 10.0]
            zone_ratio = float(np.clip(ez_median / ev_median, 0.1, 10.0)) if ev_median > 0 else 1.0

            event_median_prices.append(ev_median)
            event_zone_median_prices.append(ez_median)
            event_listing_counts.append(float(np.log1p(ev_count)))
            event_price_cvs.append(cv)
            event_zone_price_ratios.append(zone_ratio)

        result["event_median_price"] = event_median_prices
        result["event_zone_median_price"] = event_zone_median_prices
        result["event_listing_count"] = event_listing_counts
        result["event_price_cv"] = event_price_cvs
        result["event_zone_price_ratio"] = event_zone_price_ratios

        return result

    def get_params(self) -> dict[str, Any]:
        """Return extractor parameters for serialization."""
        return {"fitted": self._fitted}
