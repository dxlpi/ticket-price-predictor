"""High-level data access repositories for events and snapshots."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from ticket_price_predictor.schemas import EventMetadata, EventType, PriceSnapshot, SeatZone
from ticket_price_predictor.storage.parquet import append_parquet, read_parquet, table_to_dicts


@dataclass
class EventFilters:
    """Filters for querying events."""

    event_type: EventType | None = None
    city: str | None = None
    venue_id: str | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None


class EventRepository:
    """Repository for event metadata storage and retrieval."""

    def __init__(self, data_dir: Path) -> None:
        """Initialize the repository.

        Args:
            data_dir: Base directory for data storage
        """
        self._data_dir = data_dir
        self._events_path = data_dir / "raw" / "events" / "events.parquet"

    def save_events(self, events: list[EventMetadata]) -> int:
        """Save events to storage, deduplicating by event_id.

        Args:
            events: List of events to save

        Returns:
            Number of new events saved
        """
        if not events:
            return 0

        # Get existing event IDs
        existing_ids = set(self.list_event_ids())

        # Filter to only new events
        new_events = [e for e in events if e.event_id not in existing_ids]

        if not new_events:
            return 0

        # Append new events
        append_parquet(new_events, self._events_path, EventMetadata.parquet_schema())

        return len(new_events)

    def get_event(self, event_id: str) -> EventMetadata | None:
        """Get a single event by ID.

        Args:
            event_id: Event ID to look up

        Returns:
            EventMetadata or None if not found
        """
        table = read_parquet(self._events_path, filters=[("event_id", "=", event_id)])

        if table is None or table.num_rows == 0:
            return None

        records = table_to_dicts(table)
        return self._dict_to_event(records[0])

    def get_events(self, filters: EventFilters | None = None) -> list[EventMetadata]:
        """Get events matching optional filters.

        Args:
            filters: Optional filters to apply

        Returns:
            List of matching events
        """
        pq_filters: list[tuple[str, str, Any]] = []

        if filters:
            if filters.event_type:
                pq_filters.append(("event_type", "=", filters.event_type.value))
            if filters.city:
                pq_filters.append(("city", "=", filters.city))
            if filters.venue_id:
                pq_filters.append(("venue_id", "=", filters.venue_id))

        table = read_parquet(self._events_path, filters=pq_filters if pq_filters else None)

        if table is None:
            return []

        records = table_to_dicts(table)
        events = [self._dict_to_event(r) for r in records]

        # Apply date filters in Python (PyArrow timestamp filtering can be tricky)
        if filters:
            if filters.start_date:
                events = [e for e in events if e.event_datetime >= filters.start_date]
            if filters.end_date:
                events = [e for e in events if e.event_datetime <= filters.end_date]

        return events

    def list_event_ids(self) -> list[str]:
        """Get all stored event IDs.

        Returns:
            List of event IDs
        """
        table = read_parquet(self._events_path)

        if table is None:
            return []

        return cast(list[str], table.column("event_id").to_pylist())

    def _dict_to_event(self, data: dict[str, Any]) -> EventMetadata:
        """Convert a dictionary to EventMetadata."""
        # Handle timezone-naive datetimes from Parquet
        event_datetime = data["event_datetime"]
        if isinstance(event_datetime, datetime) and event_datetime.tzinfo is None:
            event_datetime = event_datetime.replace(tzinfo=UTC)

        return EventMetadata(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            event_datetime=event_datetime,
            artist_or_team=data["artist_or_team"],
            venue_id=data["venue_id"],
            venue_name=data["venue_name"],
            city=data["city"],
            country=data["country"],
            venue_capacity=data.get("venue_capacity"),
        )


class SnapshotRepository:
    """Repository for price snapshot storage and retrieval."""

    def __init__(self, data_dir: Path) -> None:
        """Initialize the repository.

        Args:
            data_dir: Base directory for data storage
        """
        self._data_dir = data_dir
        self._snapshots_dir = data_dir / "raw" / "snapshots"

    def save_snapshots(self, snapshots: list[PriceSnapshot]) -> int:
        """Save snapshots to storage, partitioned by year/month.

        Args:
            snapshots: List of snapshots to save

        Returns:
            Number of snapshots saved
        """
        if not snapshots:
            return 0

        # Group snapshots by year/month
        by_partition: dict[tuple[int, int], list[PriceSnapshot]] = {}
        for snapshot in snapshots:
            key = (snapshot.timestamp.year, snapshot.timestamp.month)
            if key not in by_partition:
                by_partition[key] = []
            by_partition[key].append(snapshot)

        total_saved = 0
        for (year, month), partition_snapshots in by_partition.items():
            path = self._snapshots_dir / f"year={year}" / f"month={month:02d}" / "snapshots.parquet"
            append_parquet(partition_snapshots, path, PriceSnapshot.parquet_schema())
            total_saved += len(partition_snapshots)

        return total_saved

    def get_snapshots(
        self,
        event_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[PriceSnapshot]:
        """Get snapshots matching criteria.

        Args:
            event_id: Optional event ID filter
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            List of matching snapshots
        """
        if not self._snapshots_dir.exists():
            return []

        all_snapshots: list[PriceSnapshot] = []

        # Read all partition files
        for parquet_file in self._snapshots_dir.rglob("*.parquet"):
            pq_filters: list[tuple[str, str, Any]] = []
            if event_id:
                pq_filters.append(("event_id", "=", event_id))

            table = read_parquet(parquet_file, filters=pq_filters if pq_filters else None)

            if table is None:
                continue

            records = table_to_dicts(table)
            for record in records:
                snapshot = self._dict_to_snapshot(record)

                # Apply time filters
                if start_time and snapshot.timestamp < start_time:
                    continue
                if end_time and snapshot.timestamp > end_time:
                    continue

                all_snapshots.append(snapshot)

        # Sort by timestamp
        all_snapshots.sort(key=lambda s: s.timestamp)

        return all_snapshots

    def get_latest_snapshot(
        self, event_id: str, seat_zone: SeatZone | None = None
    ) -> PriceSnapshot | None:
        """Get the most recent snapshot for an event.

        Args:
            event_id: Event ID to look up
            seat_zone: Optional seat zone filter

        Returns:
            Latest PriceSnapshot or None if not found
        """
        snapshots = self.get_snapshots(event_id=event_id)

        if seat_zone:
            snapshots = [s for s in snapshots if s.seat_zone == seat_zone]

        if not snapshots:
            return None

        return max(snapshots, key=lambda s: s.timestamp)

    def _dict_to_snapshot(self, data: dict[str, Any]) -> PriceSnapshot:
        """Convert a dictionary to PriceSnapshot."""
        # Handle timezone-naive datetimes from Parquet
        timestamp = data["timestamp"]
        if isinstance(timestamp, datetime) and timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)

        return PriceSnapshot(
            event_id=data["event_id"],
            seat_zone=SeatZone(data["seat_zone"]),
            timestamp=timestamp,
            price_min=data["price_min"],
            price_avg=data.get("price_avg"),
            price_max=data.get("price_max"),
            inventory_remaining=data.get("inventory_remaining"),
            days_to_event=data["days_to_event"],
        )
