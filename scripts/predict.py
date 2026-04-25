#!/usr/bin/env python3
"""Make price predictions using a trained model.

Usage:
    python scripts/predict.py --artist "Bruno Mars" --section "Lower Level 100"
    python scripts/predict.py --event-id "abc123" --days-ahead 7
    python scripts/predict.py --all-zones --artist "BTS" --city "Los Angeles"
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

from ticket_price_predictor.ml.inference import UnknownEventError
from ticket_price_predictor.ml.inference.predictor import PricePredictor
from ticket_price_predictor.normalization.seat_zones import SeatZone


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Make price predictions")

    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("data/models/lightgbm_v1.joblib"),
        help="Path to trained model",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        choices=["baseline", "lightgbm", "quantile"],
        default="lightgbm",
        help="Model type (default: lightgbm)",
    )

    parser.add_argument(
        "--artist",
        type=str,
        default="Unknown Artist",
        help="Artist or team name",
    )

    parser.add_argument(
        "--venue",
        type=str,
        default="Unknown Venue",
        help="Venue name",
    )

    parser.add_argument(
        "--city",
        type=str,
        default="New York",
        help="City (default: New York)",
    )

    parser.add_argument(
        "--section",
        type=str,
        default="Lower Level 100",
        help="Seat section (default: Lower Level 100)",
    )

    parser.add_argument(
        "--row",
        type=str,
        default="10",
        help="Row (default: 10)",
    )

    parser.add_argument(
        "--days-ahead",
        type=int,
        default=14,
        help="Days until event (default: 14)",
    )

    parser.add_argument(
        "--event-type",
        type=str,
        choices=["CONCERT", "SPORTS", "THEATER"],
        default="CONCERT",
        help="Event type (default: CONCERT)",
    )

    parser.add_argument(
        "--all-zones",
        action="store_true",
        help="Predict for all seat zones",
    )

    parser.add_argument(
        "--event-id",
        type=str,
        default="event_001",
        help="Event ID (default: event_001)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("TICKET PRICE PREDICTION")
    print("=" * 60)
    print()

    # Check if model exists
    if not args.model_path.exists():
        print(f"ERROR: Model not found at {args.model_path}")
        print("Train a model first: python scripts/train_model.py")
        return

    # Load model
    print(f"Loading model from: {args.model_path}")
    predictor = PricePredictor.from_path(args.model_path, args.model_type)

    # Event datetime
    event_datetime = datetime.now() + timedelta(days=args.days_ahead)

    print()
    print(f"Artist: {args.artist}")
    print(f"Venue: {args.venue}")
    print(f"City: {args.city}")
    print(f"Event: {event_datetime.strftime('%Y-%m-%d')}")
    print(f"Days to event: {args.days_ahead}")
    print()

    if args.all_zones:
        # Predict for all zones
        print("Predictions by zone:")
        print("-" * 50)

        try:
            predictions = predictor.predict_for_zones(
                event_id=args.event_id,
                artist_or_team=args.artist,
                venue_name=args.venue,
                city=args.city,
                event_datetime=event_datetime,
                days_to_event=args.days_ahead,
                event_type=args.event_type,
            )
        except UnknownEventError as exc:
            print(f"ERROR: {exc}")
            sys.exit(2)

        for zone, pred in predictions.items():
            print(
                f"  {zone.value:15} | ${pred.predicted_price:>8.2f} "
                f"(${pred.price_lower_bound:.0f} - ${pred.price_upper_bound:.0f}) "
                f"| {pred.predicted_direction} ({pred.direction_probability:.0%})"
            )

    else:
        # Single prediction
        print(f"Section: {args.section}, Row: {args.row}")
        print("-" * 50)

        try:
            pred = predictor.predict(
                event_id=args.event_id,
                artist_or_team=args.artist,
                venue_name=args.venue,
                city=args.city,
                event_datetime=event_datetime,
                section=args.section,
                row=args.row,
                days_to_event=args.days_ahead,
                event_type=args.event_type,
            )
        except UnknownEventError as exc:
            print(f"ERROR: {exc}")
            sys.exit(2)

        print()
        print(f"Predicted Price: ${pred.predicted_price:.2f}")
        print(f"95% CI: ${pred.price_lower_bound:.2f} - ${pred.price_upper_bound:.2f}")
        print(f"Confidence: {pred.confidence_score:.0%}")
        print()
        print(f"Direction: {pred.predicted_direction} ({pred.direction_probability:.0%})")
        print(f"Zone: {pred.seat_zone}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
