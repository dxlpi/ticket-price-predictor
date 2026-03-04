"""Feature pipeline orchestration."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ticket_price_predictor.ml.features.base import FeatureExtractor
from ticket_price_predictor.ml.features.event import EventFeatureExtractor
from ticket_price_predictor.ml.features.event_pricing import EventPricingFeatureExtractor
from ticket_price_predictor.ml.features.performer import PerformerFeatureExtractor
from ticket_price_predictor.ml.features.popularity import PopularityFeatureExtractor
from ticket_price_predictor.ml.features.regional import RegionalPopularityFeatureExtractor
from ticket_price_predictor.ml.features.seating import SeatingFeatureExtractor
from ticket_price_predictor.ml.features.snapshot import SnapshotFeatureExtractor
from ticket_price_predictor.ml.features.timeseries import (
    MomentumFeatureExtractor,
    TimeSeriesFeatureExtractor,
)

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Orchestrates all feature extractors.

    Supports two-stage extraction: base extractors run on raw data, then
    post-extractors (e.g. interactions) run on the concatenated base features.
    """

    def __init__(
        self,
        include_momentum: bool = False,
        include_snapshot: bool = True,
        include_popularity: bool = True,
        include_regional: bool = True,
        include_listing: bool = True,
        include_venue: bool = True,
        include_interactions: bool = True,
        include_event_pricing: bool = True,
        popularity_service: Any | None = None,
        extractor_params: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize feature pipeline.

        Args:
            include_momentum: Whether to include legacy momentum features (deprecated,
                default False — replaced by snapshot features)
            include_snapshot: Whether to include temporal snapshot features (default True)
            include_popularity: Whether to include external popularity features
            include_regional: Whether to include regional popularity features
            include_listing: Whether to include listing context features
            include_venue: Whether to include venue features
            include_interactions: Whether to include interaction features (post-extraction)
            include_event_pricing: Whether to include event-level pricing features
            popularity_service: PopularityService instance for API-based features
            extractor_params: Per-extractor parameter overrides, keyed by extractor class
                name. Values are dicts of attribute name → value applied via setattr after
                extractor construction. Extractors that wrap nested caches must expose
                forwarding properties (e.g. PerformerFeatureExtractor.artist_stats_smoothing).
                Example: {"EventPricingFeatureExtractor": {"SMOOTHING_FACTOR": 10}}
        """
        self._extractors: list[FeatureExtractor] = [
            PerformerFeatureExtractor(),
            EventFeatureExtractor(),
            SeatingFeatureExtractor(),
            TimeSeriesFeatureExtractor(),
        ]

        if include_event_pricing:
            ep_kwargs: dict[str, Any] = {}
            if extractor_params and "EventPricingFeatureExtractor" in extractor_params:
                ep_init = extractor_params["EventPricingFeatureExtractor"]
                if "include_section_feature" in ep_init:
                    ep_kwargs["include_section_feature"] = ep_init.pop("include_section_feature")
            self._extractors.append(EventPricingFeatureExtractor(**ep_kwargs))

        if include_momentum:
            self._extractors.append(MomentumFeatureExtractor())

        if include_snapshot:
            self._extractors.append(SnapshotFeatureExtractor())

        if include_regional:
            self._extractors.append(RegionalPopularityFeatureExtractor())

        if include_popularity:
            self._extractors.append(PopularityFeatureExtractor(popularity_service))

        # Optional listing context extractor (lazy import)
        if include_listing:
            try:
                from ticket_price_predictor.ml.features.listing import (
                    ListingContextFeatureExtractor,
                )

                self._extractors.append(ListingContextFeatureExtractor())
            except ImportError:
                logger.warning(
                    "ListingContextFeatureExtractor not available; skipping listing features"
                )

        # Optional venue extractor (lazy import)
        if include_venue:
            try:
                from ticket_price_predictor.ml.features.venue import (
                    VenueFeatureExtractor,
                )

                self._extractors.append(VenueFeatureExtractor())
            except ImportError:
                logger.warning("VenueFeatureExtractor not available; skipping venue features")

        # Apply per-extractor parameter overrides (e.g. smoothing factors).
        # Keys must match extractor class names in self._extractors, not nested cache names.
        if extractor_params:
            for extractor in self._extractors:
                cls_name = type(extractor).__name__
                if cls_name in extractor_params:
                    for key, value in extractor_params[cls_name].items():
                        setattr(extractor, key, value)

        # Post-extractors operate on concatenated base features
        self._post_extractors: list[FeatureExtractor] = []

        if include_interactions:
            try:
                from ticket_price_predictor.ml.features.interactions import (
                    InteractionFeatureExtractor,
                )

                self._post_extractors.append(InteractionFeatureExtractor())
            except ImportError:
                logger.warning(
                    "InteractionFeatureExtractor not available; skipping interaction features"
                )

        self._fitted = False

    @property
    def feature_names(self) -> list[str]:
        """Return all feature names from all extractors."""
        names: list[str] = []
        for extractor in self._extractors:
            names.extend(extractor.feature_names)
        for post_extractor in self._post_extractors:
            names.extend(post_extractor.feature_names)
        return names

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        """Fit all extractors on training data.

        Base extractors are fit on the raw DataFrame. Post-extractors are fit
        on the concatenated base feature output so they can learn interaction
        statistics from the training set.

        Args:
            df: Training DataFrame

        Returns:
            self
        """
        for extractor in self._extractors:
            extractor.fit(df)

        # Post-extractors need the base feature output for fitting
        if self._post_extractors:
            base_features = self._compute_base_features(df)
            for post_extractor in self._post_extractors:
                post_extractor.fit(base_features)

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all features from input DataFrame.

        First computes base features from the raw data, then passes the
        concatenated base features through post-extractors.

        Args:
            df: Input DataFrame with raw data

        Returns:
            DataFrame with all extracted features
        """
        base_features = self._compute_base_features(df)

        if not self._post_extractors:
            return base_features

        post_list: list[pd.DataFrame] = [base_features]
        for post_extractor in self._post_extractors:
            post_list.append(post_extractor.extract(base_features))

        return pd.concat(post_list, axis=1)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            df: Training DataFrame

        Returns:
            DataFrame with all extracted features
        """
        self.fit(df)
        return self.transform(df)

    def save(self, path: Path) -> None:
        """Serialize fitted pipeline to disk using joblib.

        PopularityFeatureExtractor handles unpicklable state via __getstate__,
        so the live pipeline remains unaffected after save.

        Args:
            path: Destination file path (e.g. lightgbm_v31_pipeline.joblib)
        """
        import joblib

        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "FeaturePipeline":
        """Load a fitted pipeline from disk.

        Args:
            path: Path to saved pipeline file

        Returns:
            Loaded FeaturePipeline with _fitted=True

        Raises:
            TypeError: If the file does not contain a FeaturePipeline
        """
        import joblib

        pipeline = joblib.load(path)
        if not isinstance(pipeline, cls):
            raise TypeError(f"Expected FeaturePipeline, got {type(pipeline)}")
        return pipeline

    def get_params(self) -> dict[str, Any]:
        """Get parameters for all extractors."""
        params: dict[str, Any] = {
            f"extractor_{i}": extractor.get_params() for i, extractor in enumerate(self._extractors)
        }
        for i, post_extractor in enumerate(self._post_extractors):
            params[f"post_extractor_{i}"] = post_extractor.get_params()
        return params

    def _compute_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features from all base extractors.

        Args:
            df: Input DataFrame with raw data

        Returns:
            DataFrame with concatenated base features
        """
        features_list: list[pd.DataFrame] = []
        for extractor in self._extractors:
            features_list.append(extractor.extract(df))
        return pd.concat(features_list, axis=1)
