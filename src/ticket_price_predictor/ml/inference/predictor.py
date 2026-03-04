"""Price prediction service."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ticket_price_predictor.config import get_ml_config
from ticket_price_predictor.ml.features.pipeline import FeaturePipeline
from ticket_price_predictor.ml.inference.cold_start import ColdStartHandler
from ticket_price_predictor.ml.models.base import PriceModel
from ticket_price_predictor.ml.models.lightgbm_model import QuantileLightGBMModel
from ticket_price_predictor.ml.schemas import PricePrediction
from ticket_price_predictor.ml.training.trainer import ModelTrainer
from ticket_price_predictor.normalization.seat_zones import SeatZone

_config = get_ml_config()


class PricePredictor:
    """Service for making price predictions."""

    def __init__(
        self,
        model: PriceModel,
        model_version: str = "v1",
        cold_start_handler: ColdStartHandler | None = None,
        popularity_service: Any | None = None,
    ) -> None:
        """Initialize predictor.

        Args:
            model: Trained model
            model_version: Model version string
            cold_start_handler: Handler for new events/performers
            popularity_service: PopularityService instance for popularity features
        """
        self._model = model
        self._model_version = model_version
        self._cold_start = cold_start_handler or ColdStartHandler()
        self._feature_pipeline = FeaturePipeline(
            include_momentum=False,
            # Snapshot features require pre-joined _snap_* columns from ModelTrainer;
            # at inference time no snapshot join occurs, so disable to avoid
            # all predictions using training-mean defaults for these features.
            include_snapshot=False,
            popularity_service=popularity_service,
        )

    @classmethod
    def from_path(
        cls,
        model_path: Path,
        model_type: str = "lightgbm",
        model_version: str = "v1",
    ) -> "PricePredictor":
        """Load predictor from saved model.

        Args:
            model_path: Path to saved model
            model_type: Type of model
            model_version: Version string

        Returns:
            PricePredictor instance
        """
        model = ModelTrainer.load(model_path, model_type)  # type: ignore
        return cls(model, model_version)

    def predict(
        self,
        event_id: str,
        artist_or_team: str,
        venue_name: str,
        city: str,
        event_datetime: datetime,
        section: str,
        row: str = "10",
        days_to_event: int = 14,
        event_type: str = "CONCERT",
        quantity: int = 2,
    ) -> PricePrediction:
        """Make a price prediction.

        Args:
            event_id: Event identifier
            artist_or_team: Artist or team name
            venue_name: Venue name
            city: City
            event_datetime: Event date/time
            section: Seat section
            row: Row
            days_to_event: Days until event
            event_type: Event type
            quantity: Number of tickets

        Returns:
            PricePrediction object
        """
        # Create input DataFrame
        df = pd.DataFrame(
            [
                {
                    "event_id": event_id,
                    "artist_or_team": artist_or_team,
                    "venue_name": venue_name,
                    "city": city,
                    "event_datetime": event_datetime,
                    "section": section,
                    "row": row,
                    "days_to_event": days_to_event,
                    "event_type": event_type,
                    "quantity": quantity,
                }
            ]
        )

        # Extract features
        X = self._feature_pipeline.transform(df)

        # Get seat zone from section
        from ticket_price_predictor.normalization.seat_zones import SeatZoneMapper

        mapper = SeatZoneMapper()
        seat_zone = mapper.normalize_zone_name(section)

        # Make prediction
        if isinstance(self._model, QuantileLightGBMModel):
            median, lower, upper = self._model.predict_with_uncertainty(X)
            predicted_price = float(median[0])
            price_lower = float(lower[0])
            price_upper = float(upper[0])
        else:
            preds = self._model.predict(X)
            predicted_price = float(preds[0])
            # Estimate bounds using config margin for non-quantile models
            price_lower = predicted_price * (1 - _config.price_bound_margin)
            price_upper = predicted_price * (1 + _config.price_bound_margin)

        # Calculate confidence based on interval width
        if price_upper > 0:
            confidence = 1.0 - min((price_upper - price_lower) / predicted_price, 1.0)
        else:
            confidence = _config.default_confidence

        # Determine direction (simplified: compare to cold-start estimate)
        cold_start = self._cold_start.get_estimate(artist_or_team, event_type)
        baseline = cold_start.prices_by_zone.get(seat_zone, predicted_price)

        if predicted_price > baseline * _config.direction_up_threshold:
            direction = "UP"
            direction_prob = min(
                _config.direction_max_probability, 0.5 + (predicted_price - baseline) / baseline
            )
        elif predicted_price < baseline * _config.direction_down_threshold:
            direction = "DOWN"
            direction_prob = min(
                _config.direction_max_probability, 0.5 + (baseline - predicted_price) / baseline
            )
        else:
            direction = "STABLE"
            direction_prob = _config.direction_stable_probability

        return PricePrediction(
            event_id=event_id,
            seat_zone=seat_zone.value,
            target_days_to_event=days_to_event,
            predicted_price=predicted_price,
            price_lower_bound=price_lower,
            price_upper_bound=price_upper,
            confidence_score=confidence,
            predicted_direction=direction,
            direction_probability=direction_prob,
            model_version=self._model_version,
            prediction_timestamp=datetime.now(UTC),
        )

    def predict_batch(
        self,
        df: pd.DataFrame,
    ) -> list[PricePrediction]:
        """Make predictions for multiple listings.

        Args:
            df: DataFrame with listing data

        Returns:
            List of PricePrediction objects
        """
        predictions = []

        for _, row in df.iterrows():
            pred = self.predict(
                event_id=row.get("event_id", "unknown"),
                artist_or_team=row.get("artist_or_team", "Unknown"),
                venue_name=row.get("venue_name", "Unknown"),
                city=row.get("city", "Unknown"),
                event_datetime=row.get("event_datetime", datetime.now(UTC)),
                section=row.get("section", "Upper Level"),
                row=row.get("row", "10"),
                days_to_event=row.get("days_to_event", 14),
                event_type=row.get("event_type", "CONCERT"),
                quantity=row.get("quantity", 2),
            )
            predictions.append(pred)

        return predictions

    def predict_for_zones(
        self,
        event_id: str,
        artist_or_team: str,
        venue_name: str,
        city: str,
        event_datetime: datetime,
        days_to_event: int = 14,
        event_type: str = "CONCERT",
    ) -> dict[SeatZone, PricePrediction]:
        """Predict prices for all seat zones.

        Args:
            event_id: Event identifier
            artist_or_team: Artist or team name
            venue_name: Venue name
            city: City
            event_datetime: Event date/time
            days_to_event: Days until event
            event_type: Event type

        Returns:
            Dictionary mapping zones to predictions
        """
        # Representative sections for each zone
        zone_sections = {
            SeatZone.FLOOR_VIP: "Floor VIP",
            SeatZone.LOWER_TIER: "Lower Level 100",
            SeatZone.UPPER_TIER: "Upper Level 200",
            SeatZone.BALCONY: "Balcony 400",
        }

        predictions = {}
        for zone, section in zone_sections.items():
            pred = self.predict(
                event_id=event_id,
                artist_or_team=artist_or_team,
                venue_name=venue_name,
                city=city,
                event_datetime=event_datetime,
                section=section,
                row="10",
                days_to_event=days_to_event,
                event_type=event_type,
            )
            predictions[zone] = pred

        return predictions
