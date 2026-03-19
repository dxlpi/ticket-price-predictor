"""Relative pricing features — where a listing sits within its event/zone/section."""

import logging
from typing import Any

import numpy as np
import pandas as pd

from ticket_price_predictor.ml.features.base import FeatureExtractor
from ticket_price_predictor.normalization.seat_zones import SeatZoneMapper

logger = logging.getLogger(__name__)

_ZONE_MAPPER = SeatZoneMapper()


class RelativePricingFeatureExtractor(FeatureExtractor):
    """Extract features capturing a listing's relative position within its event.

    These features answer: "How does this section/zone compare to the event
    average?" — providing within-event discrimination that absolute price
    features cannot.

    All statistics are computed from training data during fit(). At extract()
    time, each row looks up pre-computed group statistics to derive its
    relative position. No listing_price is used at extract() time (no leakage).
    """

    SMOOTHING_FACTOR = 20

    def __init__(self) -> None:
        # Per-event stats: event_id -> {median, mean, std, count}
        self._event_stats: dict[str, dict[str, float]] = {}
        # Per-(event, zone) stats: (event_id, zone) -> {median, count}
        self._event_zone_stats: dict[tuple[str, str], dict[str, float]] = {}
        # Per-(event, section) stats: (event_id, section) -> {median, count}
        self._event_section_stats: dict[tuple[str, str], dict[str, float]] = {}
        # Per-artist stats: artist -> {median}
        self._artist_stats: dict[str, dict[str, float]] = {}
        # Global stats
        self._global_stats: dict[str, float] = {}
        self._fitted = False

    @property
    def feature_names(self) -> list[str]:
        """Return list of feature names this extractor produces."""
        return [
            "section_price_deviation",
            "zone_price_deviation",
            "section_zone_residual",
            "zone_event_residual",
            "event_artist_residual",
        ]

    def fit(self, df: pd.DataFrame) -> "RelativePricingFeatureExtractor":
        """Fit on training data to compute group-level price statistics.

        Args:
            df: Training DataFrame with columns: event_id, listing_price,
                and optionally section, artist_or_team.

        Returns:
            self
        """
        if "event_id" not in df.columns or "listing_price" not in df.columns:
            logger.warning(
                "RelativePricingFeatureExtractor.fit(): missing required columns. "
                "Using global defaults only."
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

        # Map sections to zones
        has_section = "section" in df.columns
        if has_section:
            df["_zone"] = (
                df["section"]
                .fillna("")
                .apply(
                    lambda s: _ZONE_MAPPER.normalize_zone_name(str(s)).value if s else "upper_tier"
                )
            )

        # Per-event stats (smoothed toward global)
        self._event_stats = {}
        for event_id, group in df.groupby("event_id"):
            grp_prices = group["listing_price"].dropna()
            n = len(grp_prices)
            if n == 0:
                continue
            group_median = float(grp_prices.median())
            group_mean = float(grp_prices.mean())
            group_std = float(grp_prices.std()) if n > 1 else 0.0
            smoothed_median = (n * group_median + m * global_median) / (n + m)
            self._event_stats[str(event_id)] = {
                "median": smoothed_median,
                "mean": group_mean,
                "std": group_std,
                "count": float(n),
            }

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

        # Per-(event, section) stats (smoothed toward zone median)
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
                smoothed_median = (n * group_median + m * global_median) / (n + m)
                self._artist_stats[str(artist)] = {"median": smoothed_median}

        self._fitted = True
        return self

    def _get_zone_str(self, row: pd.Series) -> str:
        """Map a row's section to a zone string."""
        section = row.get("section", None)
        if section is None or (isinstance(section, float) and np.isnan(section)):
            return "upper_tier"
        return _ZONE_MAPPER.normalize_zone_name(str(section)).value

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract relative pricing features for each row.

        All features are derived from pre-computed group medians (no listing_price
        access at extract time), so there is no target leakage.

        Args:
            df: Input DataFrame with raw data

        Returns:
            DataFrame with 5 relative pricing feature columns
        """
        result = pd.DataFrame(index=df.index)
        global_median = self._global_stats.get("median", 150.0)

        section_price_deviations: list[float] = []
        zone_price_deviations: list[float] = []
        section_zone_residuals: list[float] = []
        zone_event_residuals: list[float] = []
        event_artist_residuals: list[float] = []

        for _, row in df.iterrows():
            event_id = str(row.get("event_id", "")) if row.get("event_id") is not None else ""
            artist = (
                str(row.get("artist_or_team", "")) if row.get("artist_or_team") is not None else ""
            )
            zone = self._get_zone_str(row)
            section = str(row.get("section", "")) if row.get("section") is not None else ""

            # Resolve event median (with artist fallback)
            ev_stats = self._event_stats.get(event_id)
            if ev_stats is not None:
                ev_median = ev_stats["median"]
            else:
                artist_data = self._artist_stats.get(artist)
                ev_median = artist_data["median"] if artist_data else global_median

            # Resolve zone median (with event fallback)
            zone_stats = self._event_zone_stats.get((event_id, zone))
            ez_median = zone_stats["median"] if zone_stats is not None else ev_median

            # Resolve section median (with zone fallback)
            section_stats = self._event_section_stats.get((event_id, section))
            es_median = section_stats["median"] if section_stats is not None else ez_median

            # Resolve artist median
            artist_data = self._artist_stats.get(artist)
            artist_median = artist_data["median"] if artist_data else global_median

            # --- Features ---

            # section_price_deviation: (section_median - zone_median) / zone_median
            # How premium is this section vs its zone
            if ez_median > 0:
                section_price_deviations.append((es_median - ez_median) / ez_median)
            else:
                section_price_deviations.append(0.0)

            # zone_price_deviation: (zone_median - event_median) / event_median
            # How premium is this zone vs the event
            if ev_median > 0:
                zone_price_deviations.append((ez_median - ev_median) / ev_median)
            else:
                zone_price_deviations.append(0.0)

            # section_zone_residual: section_median - zone_median (absolute)
            section_zone_residuals.append(es_median - ez_median)

            # zone_event_residual: zone_median - event_median (absolute)
            zone_event_residuals.append(ez_median - ev_median)

            # event_artist_residual: event_median - artist_median (absolute)
            event_artist_residuals.append(ev_median - artist_median)

        result["section_price_deviation"] = section_price_deviations
        result["zone_price_deviation"] = zone_price_deviations
        result["section_zone_residual"] = section_zone_residuals
        result["zone_event_residual"] = zone_event_residuals
        result["event_artist_residual"] = event_artist_residuals

        return result

    def get_params(self) -> dict[str, Any]:
        """Return extractor parameters for serialization."""
        return {"fitted": self._fitted}
