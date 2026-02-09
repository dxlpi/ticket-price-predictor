"""Data validation classes for preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import pandas as pd

from ticket_price_predictor.normalization.seat_zones import SeatZoneMapper
from ticket_price_predictor.preprocessing.base import Preprocessor, ProcessingResult
from ticket_price_predictor.schemas import SeatZone


@dataclass
class ValidationIssue:
    """Represents a data validation issue."""

    severity: Literal["ERROR", "WARNING"]
    column: str
    issue_type: str
    message: str
    row_indices: list[int]


class SchemaValidator(Preprocessor):
    """Validates DataFrame schema and structure.

    Checks:
    - Required columns are present
    - Column dtypes match expected types
    - Null patterns in required columns
    """

    # Column requirements by data type
    REQUIRED_COLUMNS = {
        "listings": [
            "listing_id",
            "event_id",
            "timestamp",
            "event_datetime",
            "section",
            "row",
            "quantity",
            "listing_price",
            "total_price",
            "days_to_event",
        ],
        "events": [
            "event_id",
            "event_type",
            "event_datetime",
            "artist_or_team",
            "venue_id",
            "venue_name",
            "city",
        ],
        "snapshots": [
            "event_id",
            "seat_zone",
            "timestamp",
            "price_min",
            "days_to_event",
        ],
    }

    EXPECTED_DTYPES = {
        "timestamp": "datetime64[ns]",
        "event_datetime": "datetime64[ns]",
        "listing_price": "float64",
        "total_price": "float64",
        "price_min": "float64",
        "price_avg": "float64",
        "price_max": "float64",
        "days_to_event": "int64",
        "quantity": "int64",
        "venue_capacity": "int64",
        "inventory_remaining": "int64",
    }

    def __init__(self, data_type: Literal["listings", "events", "snapshots"]):
        """Initialize validator.

        Args:
            data_type: Type of data being validated (listings, events, or snapshots)
        """
        self.data_type = data_type

    def process(self, df: pd.DataFrame) -> ProcessingResult:
        """Validate DataFrame schema.

        Args:
            df: Input DataFrame to validate

        Returns:
            ProcessingResult with validation issues
        """
        issues: list[ValidationIssue] = []

        # Check required columns
        required = self.REQUIRED_COLUMNS.get(self.data_type, [])
        missing_columns = [col for col in required if col not in df.columns]

        if missing_columns:
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    column=",".join(missing_columns),
                    issue_type="missing_columns",
                    message=f"Required columns missing: {missing_columns}",
                    row_indices=[],
                )
            )

        # Check dtypes for present columns
        for col, expected_dtype in self.EXPECTED_DTYPES.items():
            if col not in df.columns:
                continue

            actual_dtype = str(df[col].dtype)

            # Flexible dtype checking (allow various datetime and numeric representations)
            is_valid = False
            if "datetime" in expected_dtype and "datetime" in actual_dtype:
                is_valid = True
            elif "float" in expected_dtype and ("float" in actual_dtype or "int" in actual_dtype):
                is_valid = True
            elif "int" in expected_dtype and "int" in actual_dtype:
                is_valid = True
            elif actual_dtype == expected_dtype:
                is_valid = True

            if not is_valid:
                issues.append(
                    ValidationIssue(
                        severity="WARNING",
                        column=col,
                        issue_type="dtype_mismatch",
                        message=f"Column '{col}' has dtype {actual_dtype}, expected {expected_dtype}",
                        row_indices=[],
                    )
                )

        # Check null patterns in required columns
        for col in required:
            if col not in df.columns:
                continue

            null_pct = df[col].isna().sum() / len(df) if len(df) > 0 else 0

            if null_pct > 0.5:
                null_indices = df[df[col].isna()].index.tolist()
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        column=col,
                        issue_type="high_null_rate",
                        message=f"Column '{col}' has {null_pct:.1%} null values (>50% threshold)",
                        row_indices=null_indices[:100],  # Limit to first 100
                    )
                )

        # Convert issues to string format for ProcessingResult.issues
        issue_strings = [
            f"[{issue.severity}] {issue.issue_type} in '{issue.column}': {issue.message}"
            for issue in issues
        ]

        return ProcessingResult(
            data=df,
            issues=issue_strings,
            metrics={
                "validation_issues": len(issues),
                "error_count": sum(1 for issue in issues if issue.severity == "ERROR"),
                "warning_count": sum(1 for issue in issues if issue.severity == "WARNING"),
            },
        )


class ReferentialValidator(Preprocessor):
    """Validates referential integrity and enum values.

    Checks:
    - For listings: section names can be mapped to valid SeatZone
    - For snapshots: seat_zone contains valid SeatZone enum values
    - Optional: event_id references exist in events DataFrame
    """

    def __init__(
        self,
        data_type: Literal["listings", "snapshots"],
        events_df: pd.DataFrame | None = None,
    ):
        """Initialize validator.

        Args:
            data_type: Type of data being validated
            events_df: Optional events DataFrame for event_id validation
        """
        self.data_type = data_type
        self.events_df = events_df
        self.mapper = SeatZoneMapper()

    def process(self, df: pd.DataFrame) -> ProcessingResult:
        """Validate referential integrity.

        Args:
            df: Input DataFrame to validate

        Returns:
            ProcessingResult with validation issues
        """
        issues: list[ValidationIssue] = []

        # Validate seat zone references
        if self.data_type == "listings" and "section" in df.columns:
            # For listings, validate section can be mapped to SeatZone
            invalid_sections = []
            invalid_indices = []

            for idx, section in df["section"].items():
                if pd.isna(section):
                    invalid_indices.append(idx)
                    continue

                try:
                    # SeatZoneMapper.normalize_zone_name always returns a valid zone
                    # (defaults to UPPER_TIER), so we check for empty/invalid strings
                    if not isinstance(section, str) or not section.strip():
                        invalid_sections.append(section)
                        invalid_indices.append(idx)
                except Exception:
                    invalid_sections.append(section)
                    invalid_indices.append(idx)

            if invalid_indices:
                issues.append(
                    ValidationIssue(
                        severity="WARNING",
                        column="section",
                        issue_type="invalid_section",
                        message=f"Found {len(invalid_indices)} listings with invalid/empty section names",
                        row_indices=invalid_indices[:100],
                    )
                )

        elif self.data_type == "snapshots" and "seat_zone" in df.columns:
            # For snapshots, validate seat_zone contains valid enum values
            valid_zones = {zone.value for zone in SeatZone}
            invalid_indices = []

            for idx, zone in df["seat_zone"].items():
                if pd.isna(zone) or zone not in valid_zones:
                    invalid_indices.append(idx)

            if invalid_indices:
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        column="seat_zone",
                        issue_type="invalid_seat_zone",
                        message=f"Found {len(invalid_indices)} snapshots with invalid seat_zone values. Valid: {valid_zones}",
                        row_indices=invalid_indices[:100],
                    )
                )

        # Validate event_id references
        if self.events_df is not None and "event_id" in df.columns:
            valid_event_ids = set(self.events_df["event_id"]) if "event_id" in self.events_df.columns else set()

            if valid_event_ids:
                invalid_indices = []
                for idx, event_id in df["event_id"].items():
                    if pd.isna(event_id) or event_id not in valid_event_ids:
                        invalid_indices.append(idx)

                if invalid_indices:
                    issues.append(
                        ValidationIssue(
                            severity="ERROR",
                            column="event_id",
                            issue_type="invalid_event_reference",
                            message=f"Found {len(invalid_indices)} rows with event_id not in events table",
                            row_indices=invalid_indices[:100],
                        )
                    )

        # Convert issues to string format
        issue_strings = [
            f"[{issue.severity}] {issue.issue_type} in '{issue.column}': {issue.message}"
            for issue in issues
        ]

        return ProcessingResult(
            data=df,
            issues=issue_strings,
            metrics={
                "validation_issues": len(issues),
                "error_count": sum(1 for issue in issues if issue.severity == "ERROR"),
                "warning_count": sum(1 for issue in issues if issue.severity == "WARNING"),
            },
        )


class TemporalValidator(Preprocessor):
    """Validates temporal data consistency.

    Checks:
    - Timestamps are not in the future
    - Events have not already passed (event_datetime >= now)
    - days_to_event calculation is accurate
    """

    def __init__(self, allow_past_events: bool = False):
        """Initialize validator.

        Args:
            allow_past_events: If True, don't flag past events as errors
        """
        self.allow_past_events = allow_past_events

    def process(self, df: pd.DataFrame) -> ProcessingResult:
        """Validate temporal data.

        Args:
            df: Input DataFrame to validate

        Returns:
            ProcessingResult with validation issues
        """
        issues: list[ValidationIssue] = []
        now = datetime.now(UTC)

        # Check for future timestamps
        if "timestamp" in df.columns:
            future_indices = df[df["timestamp"] > now].index.tolist()

            if future_indices:
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        column="timestamp",
                        issue_type="future_timestamp",
                        message=f"Found {len(future_indices)} rows with timestamps in the future",
                        row_indices=future_indices[:100],
                    )
                )

        # Check for past events
        if "event_datetime" in df.columns and not self.allow_past_events:
            past_indices = df[df["event_datetime"] < now].index.tolist()

            if past_indices:
                issues.append(
                    ValidationIssue(
                        severity="WARNING",
                        column="event_datetime",
                        issue_type="past_event",
                        message=f"Found {len(past_indices)} rows for events that already occurred",
                        row_indices=past_indices[:100],
                    )
                )

        # Validate days_to_event calculation
        if all(col in df.columns for col in ["timestamp", "event_datetime", "days_to_event"]):
            invalid_indices = []

            for idx, row in df.iterrows():
                try:
                    timestamp = pd.to_datetime(row["timestamp"])
                    event_datetime = pd.to_datetime(row["event_datetime"])
                    days_to_event = row["days_to_event"]

                    # Calculate expected days_to_event
                    expected_days = (event_datetime - timestamp).days

                    # Allow 2-day tolerance for timezone/rounding differences
                    if abs(expected_days - days_to_event) >= 2:
                        invalid_indices.append(idx)
                except Exception:
                    invalid_indices.append(idx)

            if invalid_indices:
                issues.append(
                    ValidationIssue(
                        severity="WARNING",
                        column="days_to_event",
                        issue_type="incorrect_days_calculation",
                        message=f"Found {len(invalid_indices)} rows with incorrect days_to_event calculation (>2 day difference)",
                        row_indices=invalid_indices[:100],
                    )
                )

        # Convert issues to string format
        issue_strings = [
            f"[{issue.severity}] {issue.issue_type} in '{issue.column}': {issue.message}"
            for issue in issues
        ]

        return ProcessingResult(
            data=df,
            issues=issue_strings,
            metrics={
                "validation_issues": len(issues),
                "error_count": sum(1 for issue in issues if issue.severity == "ERROR"),
                "warning_count": sum(1 for issue in issues if issue.severity == "WARNING"),
            },
        )
