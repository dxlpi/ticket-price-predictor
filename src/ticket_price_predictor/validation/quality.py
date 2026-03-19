"""Data quality validation for events and snapshots."""

from dataclasses import dataclass, field
from datetime import UTC, datetime

from ticket_price_predictor.schemas import EventMetadata, PriceSnapshot, TicketListing


@dataclass
class ValidationResult:
    """Result of validating a single record."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class BatchValidationResult:
    """Result of validating a batch of records."""

    total_records: int
    valid_records: int
    invalid_records: int
    errors: list[tuple[int, str]] = field(default_factory=list)  # (index, error)
    warnings: list[tuple[int, str]] = field(default_factory=list)  # (index, warning)

    @property
    def is_valid(self) -> bool:
        """Return True if all records are valid."""
        return self.invalid_records == 0

    @property
    def error_rate(self) -> float:
        """Return the percentage of invalid records."""
        if self.total_records == 0:
            return 0.0
        return self.invalid_records / self.total_records


class DataValidator:
    """Validates event and snapshot data for quality issues."""

    def __init__(self, allow_past_events: bool = False) -> None:
        """Initialize the validator.

        Args:
            allow_past_events: If True, don't flag past events as errors
        """
        self._allow_past_events = allow_past_events

    def validate_event(self, event: EventMetadata) -> ValidationResult:
        """Validate an event metadata record.

        Args:
            event: Event to validate

        Returns:
            ValidationResult with errors and warnings
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Required field checks
        if not event.event_id:
            errors.append("event_id is empty")

        if not event.venue_id:
            errors.append("venue_id is empty")

        if not event.artist_or_team:
            warnings.append("artist_or_team is empty")

        if not event.city:
            warnings.append("city is empty")

        # Date validation
        now = datetime.now(UTC)
        if event.event_datetime < now and not self._allow_past_events:
            warnings.append(f"event_datetime {event.event_datetime} is in the past")

        # Capacity validation
        if event.venue_capacity is not None and event.venue_capacity < 0:
            errors.append(f"venue_capacity {event.venue_capacity} is negative")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_snapshot(self, snapshot: PriceSnapshot) -> ValidationResult:
        """Validate a price snapshot record.

        Args:
            snapshot: Snapshot to validate

        Returns:
            ValidationResult with errors and warnings
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Required field checks
        if not snapshot.event_id:
            errors.append("event_id is empty")

        # Price validation (Pydantic already checks >= 0, but double-check ordering)
        if snapshot.price_avg is not None and snapshot.price_avg < snapshot.price_min:
            errors.append(
                f"price_avg ({snapshot.price_avg}) is less than price_min ({snapshot.price_min})"
            )

        if snapshot.price_max is not None and snapshot.price_max < snapshot.price_min:
            errors.append(
                f"price_max ({snapshot.price_max}) is less than price_min ({snapshot.price_min})"
            )

        if (
            snapshot.price_avg is not None
            and snapshot.price_max is not None
            and snapshot.price_avg > snapshot.price_max
        ):
            errors.append(
                f"price_avg ({snapshot.price_avg}) is greater than price_max ({snapshot.price_max})"
            )

        # Days to event validation
        if snapshot.days_to_event < 0:
            errors.append(f"days_to_event ({snapshot.days_to_event}) is negative")

        # Timestamp validation
        now = datetime.now(UTC)
        if snapshot.timestamp > now:
            warnings.append(f"timestamp {snapshot.timestamp} is in the future")

        # Inventory validation
        if snapshot.inventory_remaining is not None and snapshot.inventory_remaining < 0:
            errors.append(f"inventory_remaining ({snapshot.inventory_remaining}) is negative")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_listing(self, listing: TicketListing) -> ValidationResult:
        """Validate a ticket listing record.

        Args:
            listing: Listing to validate

        Returns:
            ValidationResult with errors and warnings
        """
        errors: list[str] = []
        warnings: list[str] = []

        if listing.listing_price <= 0:
            errors.append(f"listing_price {listing.listing_price} is <= 0")

        if listing.total_price <= 0:
            errors.append(f"total_price {listing.total_price} is <= 0")

        if not listing.event_id:
            errors.append("event_id is empty")

        if listing.days_to_event == 0:
            warnings.append("days_to_event is 0 — may indicate defaulted event_datetime")

        if not listing.section or listing.section == "Unknown":
            warnings.append(f"section is '{listing.section}'")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)

    def validate_events(self, events: list[EventMetadata]) -> BatchValidationResult:
        """Validate a batch of events.

        Args:
            events: List of events to validate

        Returns:
            BatchValidationResult with aggregated results
        """
        all_errors: list[tuple[int, str]] = []
        all_warnings: list[tuple[int, str]] = []
        valid_count = 0

        for i, event in enumerate(events):
            result = self.validate_event(event)

            if result.is_valid:
                valid_count += 1

            for error in result.errors:
                all_errors.append((i, f"Event {event.event_id}: {error}"))

            for warning in result.warnings:
                all_warnings.append((i, f"Event {event.event_id}: {warning}"))

        return BatchValidationResult(
            total_records=len(events),
            valid_records=valid_count,
            invalid_records=len(events) - valid_count,
            errors=all_errors,
            warnings=all_warnings,
        )

    def validate_snapshots(self, snapshots: list[PriceSnapshot]) -> BatchValidationResult:
        """Validate a batch of snapshots.

        Args:
            snapshots: List of snapshots to validate

        Returns:
            BatchValidationResult with aggregated results
        """
        all_errors: list[tuple[int, str]] = []
        all_warnings: list[tuple[int, str]] = []
        valid_count = 0

        for i, snapshot in enumerate(snapshots):
            result = self.validate_snapshot(snapshot)

            if result.is_valid:
                valid_count += 1

            for error in result.errors:
                all_errors.append(
                    (i, f"Snapshot {snapshot.event_id}/{snapshot.seat_zone.value}: {error}")
                )

            for warning in result.warnings:
                all_warnings.append(
                    (i, f"Snapshot {snapshot.event_id}/{snapshot.seat_zone.value}: {warning}")
                )

        return BatchValidationResult(
            total_records=len(snapshots),
            valid_records=valid_count,
            invalid_records=len(snapshots) - valid_count,
            errors=all_errors,
            warnings=all_warnings,
        )
