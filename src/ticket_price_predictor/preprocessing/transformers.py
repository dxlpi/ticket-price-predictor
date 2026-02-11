"""Data transformers for preprocessing pipeline."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ticket_price_predictor.normalization.seat_zones import SeatZoneMapper
from ticket_price_predictor.preprocessing.base import Preprocessor, ProcessingResult
from ticket_price_predictor.preprocessing.config import PreprocessingConfig
from ticket_price_predictor.schemas import SeatZone

logger = logging.getLogger(__name__)


class EventMetadataJoiner(Preprocessor):
    """Join venue_capacity from EventMetadata to TicketListing."""

    def __init__(self, events_df: pd.DataFrame, config: PreprocessingConfig | None = None):
        """Initialize joiner with events dataframe.

        Args:
            events_df: EventMetadata DataFrame with event_id and venue_capacity
            config: Optional preprocessing configuration
        """
        self.events_df = events_df
        self.config = config or PreprocessingConfig()

    def process(self, df: pd.DataFrame) -> ProcessingResult:
        """Join venue_capacity to listings via event_id.

        Args:
            df: TicketListing DataFrame

        Returns:
            ProcessingResult with venue_capacity column added
        """
        issues: list[str] = []
        metrics: dict[str, Any] = {}

        # Track original row count
        original_rows = len(df)

        # Select only event_id and venue_capacity from events
        events_subset = self.events_df[["event_id", "venue_capacity"]].copy()

        # Perform left join
        result_df = df.merge(events_subset, on="event_id", how="left")

        # Track missing event IDs
        missing_event_ids = result_df["venue_capacity"].isna().sum()
        if missing_event_ids > 0:
            issues.append(
                f"{missing_event_ids} listings have no matching event_id in EventMetadata"
            )

        metrics["rows_joined"] = len(result_df)
        metrics["missing_event_ids_count"] = int(missing_event_ids)

        # Verify no duplicate rows created
        if len(result_df) != original_rows:
            issues.append(f"Row count changed: {original_rows} -> {len(result_df)}")

        return ProcessingResult(data=result_df, issues=issues, metrics=metrics)


class MissingValueImputer(Preprocessor):
    """Impute missing values in EventMetadata and TicketListing."""

    def __init__(self, config: PreprocessingConfig | None = None):
        """Initialize imputer with configuration.

        Args:
            config: Preprocessing configuration with imputation settings
        """
        self.config = config or PreprocessingConfig()

    def process(self, df: pd.DataFrame) -> ProcessingResult:
        """Impute missing values in the dataframe.

        Args:
            df: DataFrame to impute (EventMetadata or TicketListing)

        Returns:
            ProcessingResult with imputed values and metadata columns
        """
        issues: list[str] = []
        metrics: dict[str, Any] = {"imputation_counts": {}}
        result_df = df.copy()

        # Impute venue_capacity if present
        if "venue_capacity" in result_df.columns:
            missing_mask = result_df["venue_capacity"].isna()
            missing_count = missing_mask.sum()

            if missing_count > 0:
                # Add imputed tracking column before imputation
                result_df["venue_capacity_imputed"] = missing_mask

                # Try median by city if city column exists
                if "city" in result_df.columns:
                    city_medians = result_df.groupby("city")["venue_capacity"].median()
                    for city in result_df.loc[missing_mask, "city"].unique():
                        city_mask = missing_mask & (result_df["city"] == city)
                        if city in city_medians and not pd.isna(city_medians[city]):
                            result_df.loc[city_mask, "venue_capacity"] = city_medians[city]

                # Fallback to global default for remaining missing
                still_missing = result_df["venue_capacity"].isna()
                result_df.loc[still_missing, "venue_capacity"] = self.config.venue_capacity_default

                metrics["imputation_counts"]["venue_capacity"] = int(missing_count)
                logger.info(f"Imputed {missing_count} venue_capacity values")

        # Impute face_value if present
        if "face_value" in result_df.columns and "listing_price" in result_df.columns:
            missing_mask = result_df["face_value"].isna()
            missing_count = missing_mask.sum()

            if missing_count > 0:
                # Add imputed tracking column
                result_df["face_value_imputed"] = missing_mask

                # Impute as 50% of listing_price
                result_df.loc[missing_mask, "face_value"] = (
                    result_df.loc[missing_mask, "listing_price"] * 0.5
                )

                metrics["imputation_counts"]["face_value"] = int(missing_count)
                logger.info(f"Imputed {missing_count} face_value values")

        return ProcessingResult(data=result_df, issues=issues, metrics=metrics)


class TypeConverter(Preprocessor):
    """Convert columns to proper types."""

    def __init__(self, config: PreprocessingConfig | None = None):
        """Initialize type converter.

        Args:
            config: Optional preprocessing configuration
        """
        self.config = config or PreprocessingConfig()

    def process(self, df: pd.DataFrame) -> ProcessingResult:
        """Convert columns to appropriate types.

        Args:
            df: DataFrame to convert

        Returns:
            ProcessingResult with typed columns
        """
        issues: list[str] = []
        metrics: dict[str, Any] = {"conversions_applied": {}}
        result_df = df.copy()

        # Datetime columns - ensure timezone-aware UTC
        datetime_cols = ["timestamp", "event_datetime"]
        for col in datetime_cols:
            if col in result_df.columns:
                try:
                    # Convert to datetime if not already
                    if not pd.api.types.is_datetime64_any_dtype(result_df[col]):
                        result_df[col] = pd.to_datetime(result_df[col])
                        metrics["conversions_applied"][col] = "to_datetime"

                    # Make timezone-aware UTC if not already
                    if result_df[col].dt.tz is None:
                        result_df[col] = result_df[col].dt.tz_localize("UTC")
                        metrics["conversions_applied"][f"{col}_tz"] = "localize_utc"
                    elif str(result_df[col].dt.tz) != "UTC":
                        result_df[col] = result_df[col].dt.tz_convert("UTC")
                        metrics["conversions_applied"][f"{col}_tz"] = "convert_utc"

                except Exception as e:
                    issues.append(f"Failed to convert {col} to datetime: {e}")

        # Price columns - ensure float64
        price_cols = ["listing_price", "face_value", "total_price"]
        for col in price_cols:
            if col in result_df.columns:
                try:
                    if result_df[col].dtype != "float64":
                        result_df[col] = result_df[col].astype("float64")
                        metrics["conversions_applied"][col] = "to_float64"
                except Exception as e:
                    issues.append(f"Failed to convert {col} to float64: {e}")

        # Categorical columns
        categorical_cols = ["seat_zone", "normalized_seat_zone", "event_type"]
        for col in categorical_cols:
            if col in result_df.columns:
                try:
                    if not isinstance(result_df[col].dtype, pd.CategoricalDtype):
                        result_df[col] = result_df[col].astype("category")
                        metrics["conversions_applied"][col] = "to_category"
                except Exception as e:
                    issues.append(f"Failed to convert {col} to category: {e}")

        return ProcessingResult(data=result_df, issues=issues, metrics=metrics)


class SeatZoneEnricher(Preprocessor):
    """Enrich data with normalized seat zones."""

    def __init__(self, config: PreprocessingConfig | None = None):
        """Initialize seat zone enricher.

        Args:
            config: Optional preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        self.mapper = SeatZoneMapper()

    def process(self, df: pd.DataFrame) -> ProcessingResult:
        """Add normalized_seat_zone column based on section.

        Args:
            df: DataFrame with section column

        Returns:
            ProcessingResult with normalized_seat_zone column added
        """
        issues: list[str] = []
        metrics: dict[str, Any] = {}
        result_df = df.copy()

        if "section" not in result_df.columns:
            issues.append("No 'section' column found for seat zone normalization")
            return ProcessingResult(data=result_df, issues=issues, metrics=metrics)

        normalized_zones = []
        unmappable_count = 0

        for section in result_df["section"]:
            try:
                if pd.isna(section) or section == "":
                    # Default to UPPER_TIER for missing sections
                    normalized_zones.append(SeatZone.UPPER_TIER)
                    unmappable_count += 1
                else:
                    zone = self.mapper.normalize_zone_name(str(section))
                    normalized_zones.append(zone)
            except Exception as e:
                logger.warning(f"Failed to normalize section '{section}': {e}")
                normalized_zones.append(SeatZone.UPPER_TIER)
                unmappable_count += 1

        result_df["normalized_seat_zone"] = normalized_zones

        metrics["zones_normalized_count"] = len(result_df)
        metrics["unmappable_count"] = unmappable_count

        if unmappable_count > 0:
            issues.append(
                f"{unmappable_count} sections could not be mapped (defaulted to UPPER_TIER)"
            )

        return ProcessingResult(data=result_df, issues=issues, metrics=metrics)


class TemporalFeatureEnricher(Preprocessor):
    """Add temporal features derived from timestamps."""

    def __init__(self, config: PreprocessingConfig | None = None):
        """Initialize temporal feature enricher.

        Args:
            config: Optional preprocessing configuration
        """
        self.config = config or PreprocessingConfig()

    def process(self, df: pd.DataFrame) -> ProcessingResult:
        """Add temporal features: hour_of_day, days_to_event, is_weekend.

        Args:
            df: DataFrame with timestamp and event_datetime columns

        Returns:
            ProcessingResult with temporal features added
        """
        issues: list[str] = []
        metrics: dict[str, Any] = {}
        result_df = df.copy()
        features_added = 0

        # Add hour_of_day from timestamp
        if "timestamp" in result_df.columns:
            try:
                result_df["hour_of_day"] = result_df["timestamp"].dt.hour
                features_added += 1
            except Exception as e:
                issues.append(f"Failed to extract hour_of_day: {e}")

        # Add/recalculate days_to_event
        if "timestamp" in result_df.columns and "event_datetime" in result_df.columns:
            try:
                time_delta = result_df["event_datetime"] - result_df["timestamp"]
                result_df["days_to_event"] = (time_delta.dt.total_seconds() / (24 * 3600)).round(2)
                features_added += 1
            except Exception as e:
                issues.append(f"Failed to calculate days_to_event: {e}")

        # Add is_weekend from event_datetime
        if "event_datetime" in result_df.columns:
            try:
                # Saturday=5, Sunday=6 in Python's weekday() (Monday=0)
                result_df["is_weekend"] = result_df["event_datetime"].dt.weekday.isin([5, 6])
                features_added += 1
            except Exception as e:
                issues.append(f"Failed to calculate is_weekend: {e}")

        metrics["features_added"] = features_added

        return ProcessingResult(data=result_df, issues=issues, metrics=metrics)
