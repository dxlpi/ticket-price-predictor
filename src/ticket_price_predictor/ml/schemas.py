"""ML schemas for predictions and training metrics."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

import pyarrow as pa
from pydantic import BaseModel, Field


class PriceDirection(StrEnum):
    """Predicted price movement direction."""

    UP = "UP"
    DOWN = "DOWN"
    STABLE = "STABLE"


class PricePrediction(BaseModel):
    """Model prediction with uncertainty estimates."""

    event_id: str = Field(..., description="Event identifier")
    seat_zone: str = Field(..., description="Normalized seat zone")
    target_days_to_event: int = Field(..., ge=0, description="Days until event for prediction")

    # Point estimate
    predicted_price: float = Field(..., ge=0, description="Predicted ticket price")

    # 95% confidence interval
    price_lower_bound: float = Field(..., ge=0, description="Lower bound of 95% CI")
    price_upper_bound: float = Field(..., ge=0, description="Upper bound of 95% CI")
    confidence_score: float = Field(
        ..., ge=0, le=1, description="Confidence score (0-1, higher = more confident)"
    )

    # Direction prediction
    predicted_direction: Literal["UP", "DOWN", "STABLE"] = Field(
        ..., description="Predicted price direction"
    )
    direction_probability: float = Field(
        ..., ge=0, le=1, description="Probability of predicted direction"
    )

    # Metadata
    model_version: str = Field(..., description="Model version used for prediction")
    prediction_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="When prediction was made"
    )

    @classmethod
    def parquet_schema(cls) -> pa.Schema:
        """Return PyArrow schema for Parquet storage."""
        return pa.schema(
            [
                pa.field("event_id", pa.string(), nullable=False),
                pa.field("seat_zone", pa.string(), nullable=False),
                pa.field("target_days_to_event", pa.int32(), nullable=False),
                pa.field("predicted_price", pa.float64(), nullable=False),
                pa.field("price_lower_bound", pa.float64(), nullable=False),
                pa.field("price_upper_bound", pa.float64(), nullable=False),
                pa.field("confidence_score", pa.float64(), nullable=False),
                pa.field("predicted_direction", pa.string(), nullable=False),
                pa.field("direction_probability", pa.float64(), nullable=False),
                pa.field("model_version", pa.string(), nullable=False),
                pa.field("prediction_timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
            ]
        )


class RankedListing(BaseModel):
    """A listing ranked by predicted value relative to its asking price.

    Used by ListingRanker to surface underpriced listings (value_score > 1.0).
    Analogous to ranked item scoring in commerce recommendation systems.
    """

    event_id: str = Field(..., description="Event identifier")
    listing_id: str | None = Field(None, description="Listing identifier from marketplace")
    section: str = Field(..., description="Seat section")
    row: str = Field(..., description="Seat row")
    listing_price: float = Field(..., ge=0, description="Actual asking price")
    predicted_fair_price: float = Field(
        ..., ge=0, description="Model's predicted fair market price"
    )
    value_score: float = Field(
        ..., description="predicted_fair_price / listing_price. >1.0 means underpriced (good value)"
    )
    savings_estimate: float = Field(
        ..., description="predicted_fair_price - listing_price. Positive means potential savings"
    )
    confidence_score: float = Field(
        ..., ge=0, le=1, description="Price prediction confidence (from PricePrediction)"
    )
    rank: int = Field(..., ge=1, description="Rank position (1 = best value)")


class TrainingMetrics(BaseModel):
    """Metrics from model training and evaluation."""

    model_version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Model type (baseline, lightgbm, etc.)")
    trained_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Dataset info
    n_train_samples: int = Field(..., ge=0)
    n_val_samples: int = Field(..., ge=0)
    n_test_samples: int = Field(..., ge=0)
    n_features: int = Field(..., ge=0)

    # Regression metrics
    mae: float = Field(..., ge=0, description="Mean Absolute Error")
    rmse: float = Field(..., ge=0, description="Root Mean Squared Error")
    mape: float = Field(..., ge=0, description="Mean Absolute Percentage Error")
    r2: float = Field(..., description="R-squared score")

    # Direction classification metrics (optional)
    direction_accuracy: float | None = Field(
        None, ge=0, le=1, description="Accuracy of direction prediction"
    )

    # Classification metrics (sale probability model)
    auc_roc: float | None = Field(None, ge=0, le=1, description="AUC-ROC for binary classifier")
    precision: float | None = Field(
        None, ge=0, le=1, description="Precision at default threshold (0.5)"
    )
    recall: float | None = Field(None, ge=0, le=1, description="Recall at default threshold (0.5)")
    f1: float | None = Field(None, ge=0, le=1, description="F1 score at default threshold (0.5)")
    calibration_error: float | None = Field(
        None, ge=0, description="Expected calibration error (ECE)"
    )

    # Training info
    training_time_seconds: float = Field(..., ge=0)
    best_iteration: int | None = Field(None, ge=0)

    # Feature importance (top 10)
    feature_importance: dict[str, float] = Field(
        default_factory=dict, description="Feature importance scores"
    )

    # Per-quartile and per-zone MAE breakdowns (optional — absent in old metrics files)
    quartile_mae: dict[str, float] = Field(
        default_factory=dict, description="MAE per price quartile (Q1-Q4)"
    )
    zone_mae: dict[str, float] = Field(default_factory=dict, description="MAE per seat zone")

    # Seen/unseen event breakdown (optional — populated by evaluate_with_breakdown).
    # primary_mae is the production metric: MAE on the in-scope (seen-event) test slice.
    # q4_top_decile_mae is diagnostic only — note this differs from quartile_mae.Q4
    # (top quartile via compute_metrics) and is defined as MAE where y >= 0.9*p95(y).
    primary_mae: float | None = Field(
        None, description="Headline metric: MAE on seen-event slice of test set"
    )
    seen_mae: float | None = Field(None, description="Alias of primary_mae")
    unseen_mae: float | None = Field(None, description="MAE on unseen-event slice (diagnostic)")
    q4_top_decile_mae: float | None = Field(
        None, description="MAE where y_test >= 0.9*p95(y_test) (top decile by price; diagnostic)"
    )
    unseen_event_pct_by_event: float | None = Field(
        None, description="Fraction of distinct test event_ids not in train_events"
    )

    # Stacking diagnostic (optional — populated by stacking_v2 q75_tail meta-feature)
    gate_on_rate: float | None = Field(
        None, description="Fraction of training rows where huber_pred >= log1p($310) (Q4 gate)"
    )
