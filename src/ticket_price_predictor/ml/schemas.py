"""ML schemas for predictions and training metrics."""

from datetime import datetime
from enum import Enum
from typing import Literal

import pyarrow as pa
from pydantic import BaseModel, Field


class PriceDirection(str, Enum):
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
        default_factory=datetime.utcnow, description="When prediction was made"
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
                pa.field(
                    "prediction_timestamp", pa.timestamp("us", tz="UTC"), nullable=False
                ),
            ]
        )


class TrainingMetrics(BaseModel):
    """Metrics from model training and evaluation."""

    model_version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Model type (baseline, lightgbm, etc.)")
    trained_at: datetime = Field(default_factory=datetime.utcnow)

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

    # Training info
    training_time_seconds: float = Field(..., ge=0)
    best_iteration: int | None = Field(None, ge=0)

    # Feature importance (top 10)
    feature_importance: dict[str, float] = Field(
        default_factory=dict, description="Feature importance scores"
    )
