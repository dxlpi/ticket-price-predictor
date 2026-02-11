"""Data cleaning preprocessors for ticket listing data."""

from __future__ import annotations

import pandas as pd

from .base import Preprocessor, ProcessingResult
from .config import PreprocessingConfig


class TextNormalizer(Preprocessor):
    """Normalize text fields for consistent matching and analysis.

    Adds normalized versions of artist, venue, and city fields to facilitate
    matching and grouping while preserving original values.
    """

    def __init__(self, config: PreprocessingConfig | None = None):
        """Initialize text normalizer.

        Args:
            config: Preprocessing configuration (uses defaults if None)
        """
        self.config = config or PreprocessingConfig()

        # City name mappings for common variations
        self.city_mappings = {
            "nyc": "new york",
            "la": "los angeles",
            "sf": "san francisco",
            "dc": "washington",
            "nola": "new orleans",
            "vegas": "las vegas",
            "philly": "philadelphia",
        }

    def process(self, df: pd.DataFrame) -> ProcessingResult:
        """Normalize text fields and add normalized columns.

        Args:
            df: Input DataFrame with artist_or_team, venue_name, city columns

        Returns:
            ProcessingResult with normalized columns added
        """
        if df.empty:
            return ProcessingResult(
                data=df,
                issues=["Input DataFrame is empty"],
                metrics={"rows_processed": 0, "columns_added": 0},
            )

        result_df = df.copy()
        issues = []
        columns_added = 0

        # Normalize artist_or_team
        if "artist_or_team" in result_df.columns:
            result_df["artist_normalized"] = (
                result_df["artist_or_team"].fillna("").str.strip().str.lower()
            )
            columns_added += 1
        else:
            issues.append("Missing column: artist_or_team")

        # Normalize venue_name
        if "venue_name" in result_df.columns:
            result_df["venue_normalized"] = (
                result_df["venue_name"].fillna("").str.strip().str.lower()
            )
            columns_added += 1
        else:
            issues.append("Missing column: venue_name")

        # Normalize city with common mappings
        if "city" in result_df.columns:
            normalized_city = result_df["city"].fillna("").str.strip().str.lower()
            # Apply city mappings
            result_df["city_normalized"] = normalized_city.replace(self.city_mappings)
            columns_added += 1
        else:
            issues.append("Missing column: city")

        metrics = {
            "rows_processed": len(result_df),
            "columns_added": columns_added,
        }

        return ProcessingResult(data=result_df, issues=issues, metrics=metrics)


class PriceOutlierHandler(Preprocessor):
    """Detect and flag price outliers using IQR method and absolute bounds.

    Identifies outliers based on:
    - IQR (Interquartile Range) method for statistical outliers
    - Absolute min/max bounds for invalid prices

    Does not remove outliers, only flags them for downstream handling.
    """

    def __init__(self, config: PreprocessingConfig | None = None):
        """Initialize price outlier handler.

        Args:
            config: Preprocessing configuration (uses defaults if None)
        """
        self.config = config or PreprocessingConfig()

    def process(self, df: pd.DataFrame) -> ProcessingResult:
        """Detect and flag price outliers.

        Args:
            df: Input DataFrame with listing_price column

        Returns:
            ProcessingResult with outlier flags and reasons added
        """
        if df.empty:
            return ProcessingResult(
                data=df,
                issues=["Input DataFrame is empty"],
                metrics={"outlier_count": 0, "outlier_percentage": 0.0, "by_reason": {}},
            )

        result_df = df.copy()
        issues = []

        if "listing_price" not in result_df.columns:
            issues.append("Missing column: listing_price")
            result_df["is_price_outlier"] = False
            result_df["outlier_reason"] = None
            return ProcessingResult(
                data=result_df,
                issues=issues,
                metrics={"outlier_count": 0, "outlier_percentage": 0.0, "by_reason": {}},
            )

        prices = result_df["listing_price"].dropna()

        if len(prices) == 0:
            issues.append("No valid prices found in listing_price column")
            result_df["is_price_outlier"] = False
            result_df["outlier_reason"] = None
            return ProcessingResult(
                data=result_df,
                issues=issues,
                metrics={"outlier_count": 0, "outlier_percentage": 0.0, "by_reason": {}},
            )

        # Calculate IQR bounds
        q1 = prices.quantile(0.25)
        q3 = prices.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (self.config.iqr_multiplier * iqr)
        upper_bound = q3 + (self.config.iqr_multiplier * iqr)

        # Initialize outlier columns
        result_df["is_price_outlier"] = False
        result_df["outlier_reason"] = None

        # Flag outliers with reasons (priority: absolute bounds, then IQR)
        mask_below_min = result_df["listing_price"] < self.config.price_min
        mask_above_max = result_df["listing_price"] > self.config.price_max
        mask_below_iqr = (result_df["listing_price"] < lower_bound) & ~mask_below_min
        mask_above_iqr = (result_df["listing_price"] > upper_bound) & ~mask_above_max

        result_df.loc[mask_below_min, "is_price_outlier"] = True
        result_df.loc[mask_below_min, "outlier_reason"] = "below_min"

        result_df.loc[mask_above_max, "is_price_outlier"] = True
        result_df.loc[mask_above_max, "outlier_reason"] = "above_max"

        result_df.loc[mask_below_iqr, "is_price_outlier"] = True
        result_df.loc[mask_below_iqr, "outlier_reason"] = "below_iqr"

        result_df.loc[mask_above_iqr, "is_price_outlier"] = True
        result_df.loc[mask_above_iqr, "outlier_reason"] = "above_iqr"

        # Calculate metrics
        outlier_count = result_df["is_price_outlier"].sum()
        outlier_percentage = (outlier_count / len(result_df)) * 100 if len(result_df) > 0 else 0.0

        by_reason = (
            result_df[result_df["is_price_outlier"]]["outlier_reason"].value_counts().to_dict()
        )

        metrics = {
            "outlier_count": int(outlier_count),
            "outlier_percentage": round(outlier_percentage, 2),
            "by_reason": by_reason,
            "iqr_lower_bound": round(lower_bound, 2),
            "iqr_upper_bound": round(upper_bound, 2),
            "q1": round(q1, 2),
            "q3": round(q3, 2),
        }

        return ProcessingResult(data=result_df, issues=issues, metrics=metrics)


class DuplicateHandler(Preprocessor):
    """Detect and flag duplicate listings within a time window.

    Identifies duplicates based on:
    - event_id
    - section, row, seat_from, seat_to
    - Within 6-hour timestamp window

    Keeps the first occurrence, flags subsequent duplicates.
    """

    def __init__(self, time_window_hours: int = 6):
        """Initialize duplicate handler.

        Args:
            time_window_hours: Time window in hours for considering duplicates
        """
        self.time_window_hours = time_window_hours

    def process(self, df: pd.DataFrame) -> ProcessingResult:
        """Detect and flag duplicate listings.

        Args:
            df: Input DataFrame with event_id, section, row, seat_from, seat_to, timestamp

        Returns:
            ProcessingResult with is_duplicate column added
        """
        if df.empty:
            return ProcessingResult(
                data=df,
                issues=["Input DataFrame is empty"],
                metrics={"duplicate_count": 0, "duplicate_percentage": 0.0},
            )

        result_df = df.copy()
        issues = []

        # Check required columns
        required_cols = ["event_id", "section", "row", "timestamp"]
        missing_cols = [col for col in required_cols if col not in result_df.columns]

        if missing_cols:
            issues.append(f"Missing required columns: {', '.join(missing_cols)}")
            result_df["is_duplicate"] = False
            return ProcessingResult(
                data=result_df,
                issues=issues,
                metrics={"duplicate_count": 0, "duplicate_percentage": 0.0},
            )

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(result_df["timestamp"]):
            try:
                result_df["timestamp"] = pd.to_datetime(result_df["timestamp"])
            except Exception as e:
                issues.append(f"Failed to convert timestamp to datetime: {e}")
                result_df["is_duplicate"] = False
                return ProcessingResult(
                    data=result_df,
                    issues=issues,
                    metrics={"duplicate_count": 0, "duplicate_percentage": 0.0},
                )

        # Fill None values for seat columns with empty string for grouping
        if "seat_from" in result_df.columns:
            result_df["seat_from"] = result_df["seat_from"].fillna("")
        else:
            result_df["seat_from"] = ""

        if "seat_to" in result_df.columns:
            result_df["seat_to"] = result_df["seat_to"].fillna("")
        else:
            result_df["seat_to"] = ""

        # Sort by timestamp to keep first occurrence
        result_df = result_df.sort_values("timestamp")

        # Initialize is_duplicate column
        result_df["is_duplicate"] = False

        # Group by identifying fields
        group_cols = ["event_id", "section", "row", "seat_from", "seat_to"]

        for _name, group in result_df.groupby(group_cols):
            if len(group) <= 1:
                continue

            # Check time window for duplicates
            timestamps = group["timestamp"].values
            indices = group.index.tolist()

            for i in range(len(timestamps)):
                if result_df.loc[indices[i], "is_duplicate"]:
                    # Already marked as duplicate
                    continue

                # Compare with all subsequent records in the group
                for j in range(i + 1, len(timestamps)):
                    time_diff = pd.Timedelta(timestamps[j] - timestamps[i])

                    if time_diff <= pd.Timedelta(hours=self.time_window_hours):
                        # Mark the later one as duplicate
                        result_df.loc[indices[j], "is_duplicate"] = True

        # Calculate metrics
        duplicate_count = result_df["is_duplicate"].sum()
        duplicate_percentage = (
            (duplicate_count / len(result_df)) * 100 if len(result_df) > 0 else 0.0
        )

        metrics = {
            "duplicate_count": int(duplicate_count),
            "duplicate_percentage": round(duplicate_percentage, 2),
            "time_window_hours": self.time_window_hours,
        }

        return ProcessingResult(data=result_df, issues=issues, metrics=metrics)
