"""Quality reporting and metrics for preprocessing pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .base import ProcessingResult
from .config import PreprocessingConfig


class AlertLevel(Enum):
    """Alert severity levels for quality checks."""

    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for preprocessing results."""

    # Row counts
    input_rows: int = 0
    output_rows: int = 0
    dropped_rows: int = 0
    flagged_rows: int = 0

    # Column completeness (percentage)
    column_completeness: dict[str, float] = field(default_factory=dict)

    # Outlier counts by column
    outlier_counts: dict[str, int] = field(default_factory=dict)

    # Validation errors by type
    validation_errors: dict[str, int] = field(default_factory=dict)

    # Additional custom metrics
    custom_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to JSON-serializable dictionary."""
        return asdict(self)

    @property
    def drop_rate(self) -> float:
        """Calculate percentage of dropped rows."""
        if self.input_rows == 0:
            return 0.0
        return (self.dropped_rows / self.input_rows) * 100

    @property
    def retention_rate(self) -> float:
        """Calculate percentage of retained rows."""
        if self.input_rows == 0:
            return 0.0
        return (self.output_rows / self.input_rows) * 100


@dataclass
class QualityThresholds:
    """Configurable thresholds for quality alerts."""

    # Drop rate thresholds
    drop_rate_warning: float = 5.0  # >5% drops = warning
    drop_rate_error: float = 20.0  # >20% drops = error

    # Null rate thresholds
    null_rate_warning: float = 10.0  # >10% nulls = warning
    null_rate_error: float = 50.0  # >50% nulls = error

    # Outlier rate thresholds
    outlier_rate_warning: float = 5.0  # >5% outliers = warning
    outlier_rate_error: float = 15.0  # >15% outliers = error

    # Duplicate rate thresholds
    duplicate_rate_warning: float = 5.0  # >5% duplicates = warning
    duplicate_rate_error: float = 20.0  # >20% duplicates = error


class QualityReporter:
    """Generate quality reports from preprocessing results."""

    def __init__(
        self,
        config: PreprocessingConfig | None = None,
        thresholds: QualityThresholds | None = None,
    ):
        """Initialize reporter with configuration.

        Args:
            config: Preprocessing configuration
            thresholds: Quality thresholds for alerting
        """
        self.config = config or PreprocessingConfig()
        self.thresholds = thresholds or QualityThresholds()

    def extract_metrics(self, result: ProcessingResult) -> QualityMetrics:
        """Extract quality metrics from processing result.

        Args:
            result: ProcessingResult from pipeline execution

        Returns:
            QualityMetrics with comprehensive statistics
        """
        metrics = QualityMetrics()

        # Extract row counts
        metrics.output_rows = len(result.data)
        metrics.input_rows = result.metrics.get("input_rows", metrics.output_rows)
        metrics.dropped_rows = result.metrics.get("dropped_rows", 0)
        metrics.flagged_rows = result.metrics.get("flagged_rows", 0)

        # Calculate column completeness
        for col in result.data.columns:
            non_null = result.data[col].notna().sum()
            completeness = (non_null / len(result.data) * 100) if len(result.data) > 0 else 0.0
            metrics.column_completeness[col] = round(completeness, 2)

        # Extract outlier counts from metrics
        if "outliers" in result.metrics:
            metrics.outlier_counts = result.metrics["outliers"]

        # Extract validation errors from issues
        for issue in result.issues:
            # Parse issue strings like "duplicate_rows: 5 duplicates found"
            if ":" in issue:
                error_type = issue.split(":")[0].strip()
                metrics.validation_errors[error_type] = (
                    metrics.validation_errors.get(error_type, 0) + 1
                )

        # Copy custom metrics
        for key, value in result.metrics.items():
            if key not in ["input_rows", "dropped_rows", "flagged_rows", "outliers"]:
                metrics.custom_metrics[key] = value

        return metrics

    def generate_text_summary(self, metrics: QualityMetrics) -> str:
        """Generate human-readable text report.

        Args:
            metrics: Quality metrics to report

        Returns:
            Formatted text summary
        """
        lines = ["=" * 60, "PREPROCESSING QUALITY REPORT", "=" * 60, ""]

        # Row statistics
        lines.append("ROW STATISTICS:")
        lines.append(f"  Input rows:     {metrics.input_rows:,}")
        lines.append(f"  Output rows:    {metrics.output_rows:,}")
        lines.append(f"  Dropped rows:   {metrics.dropped_rows:,} ({metrics.drop_rate:.2f}%)")
        lines.append(f"  Flagged rows:   {metrics.flagged_rows:,}")
        lines.append(f"  Retention rate: {metrics.retention_rate:.2f}%")
        lines.append("")

        # Column completeness
        if metrics.column_completeness:
            lines.append("COLUMN COMPLETENESS:")
            for col, pct in sorted(metrics.column_completeness.items()):
                status = "✓" if pct >= 95.0 else "⚠" if pct >= 80.0 else "✗"
                lines.append(f"  {status} {col:30s} {pct:6.2f}%")
            lines.append("")

        # Outliers
        if metrics.outlier_counts:
            lines.append("OUTLIER DETECTION:")
            total_outliers = sum(metrics.outlier_counts.values())
            lines.append(f"  Total outliers: {total_outliers:,}")
            for col, count in sorted(
                metrics.outlier_counts.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"    - {col}: {count:,}")
            lines.append("")

        # Validation errors
        if metrics.validation_errors:
            lines.append("VALIDATION ERRORS:")
            for error_type, count in sorted(
                metrics.validation_errors.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"  - {error_type}: {count}")
            lines.append("")

        # Custom metrics
        if metrics.custom_metrics:
            lines.append("ADDITIONAL METRICS:")
            for key, value in sorted(metrics.custom_metrics.items()):
                lines.append(f"  - {key}: {value}")
            lines.append("")

        # Alert level
        alert = self.check_thresholds(metrics)
        alert_symbol = {"ok": "✓", "warning": "⚠", "error": "✗"}[alert.value]
        lines.append(f"OVERALL STATUS: {alert_symbol} {alert.value.upper()}")
        lines.append("=" * 60)

        return "\n".join(lines)

    def generate_json_export(self, metrics: QualityMetrics) -> str:
        """Generate JSON export for monitoring systems.

        Args:
            metrics: Quality metrics to export

        Returns:
            JSON string
        """
        data = {
            "metrics": metrics.to_dict(),
            "alert_level": self.check_thresholds(metrics).value,
            "thresholds": asdict(self.thresholds),
        }
        return json.dumps(data, indent=2)

    def check_thresholds(self, metrics: QualityMetrics) -> AlertLevel:
        """Check metrics against thresholds and return alert level.

        Args:
            metrics: Quality metrics to check

        Returns:
            AlertLevel (OK, WARNING, or ERROR)
        """
        alert = AlertLevel.OK

        # Check drop rate
        if metrics.drop_rate > self.thresholds.drop_rate_error:
            return AlertLevel.ERROR
        elif metrics.drop_rate > self.thresholds.drop_rate_warning:
            alert = AlertLevel.WARNING

        # Check null rates
        for _col, completeness in metrics.column_completeness.items():
            null_rate = 100.0 - completeness
            if null_rate > self.thresholds.null_rate_error:
                return AlertLevel.ERROR
            elif null_rate > self.thresholds.null_rate_warning:
                alert = AlertLevel.WARNING

        # Check outlier rates
        if metrics.input_rows > 0:
            total_outliers = sum(metrics.outlier_counts.values())
            outlier_rate = (total_outliers / metrics.input_rows) * 100
            if outlier_rate > self.thresholds.outlier_rate_error:
                return AlertLevel.ERROR
            elif outlier_rate > self.thresholds.outlier_rate_warning:
                alert = AlertLevel.WARNING

        # Check for duplicate errors
        dup_count = metrics.validation_errors.get("duplicate_rows", 0)
        if metrics.input_rows > 0:
            dup_rate = (dup_count / metrics.input_rows) * 100
            if dup_rate > self.thresholds.duplicate_rate_error:
                return AlertLevel.ERROR
            elif dup_rate > self.thresholds.duplicate_rate_warning:
                alert = AlertLevel.WARNING

        return alert

    def compare_against_baseline(
        self, current: QualityMetrics, baseline: QualityMetrics
    ) -> dict[str, Any]:
        """Compare current metrics against a baseline.

        Args:
            current: Current quality metrics
            baseline: Baseline metrics to compare against

        Returns:
            Dictionary with comparison results and deltas
        """
        comparison: dict[str, Any] = {
            "row_count_delta": current.output_rows - baseline.output_rows,
            "drop_rate_delta": current.drop_rate - baseline.drop_rate,
            "retention_rate_delta": current.retention_rate - baseline.retention_rate,
            "completeness_changes": {},
            "outlier_changes": {},
            "improved": [],
            "degraded": [],
        }

        # Compare column completeness
        all_columns = set(current.column_completeness.keys()) | set(
            baseline.column_completeness.keys()
        )
        for col in all_columns:
            curr_val = current.column_completeness.get(col, 0.0)
            base_val = baseline.column_completeness.get(col, 0.0)
            delta = curr_val - base_val
            comparison["completeness_changes"][col] = round(delta, 2)

            if delta > 1.0:
                comparison["improved"].append(f"completeness:{col}")
            elif delta < -1.0:
                comparison["degraded"].append(f"completeness:{col}")

        # Compare outlier counts
        all_outlier_cols = set(current.outlier_counts.keys()) | set(baseline.outlier_counts.keys())
        for col in all_outlier_cols:
            curr_val = current.outlier_counts.get(col, 0)
            base_val = baseline.outlier_counts.get(col, 0)
            delta = curr_val - base_val
            comparison["outlier_changes"][col] = delta

            if delta < -5:  # At least 5 fewer outliers
                comparison["improved"].append(f"outliers:{col}")
            elif delta > 5:
                comparison["degraded"].append(f"outliers:{col}")

        # Overall assessment
        comparison["overall"] = (
            "improved"
            if len(comparison["improved"]) > len(comparison["degraded"])
            else "degraded"
            if len(comparison["degraded"]) > len(comparison["improved"])
            else "stable"
        )

        return comparison
