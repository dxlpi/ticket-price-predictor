"""High-level data access repositories for events and snapshots."""

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from ticket_price_predictor.schemas import (
    EventMetadata,
    EventType,
    PriceSnapshot,
    SeatZone,
    TicketListing,
)
from ticket_price_predictor.storage.parquet import append_parquet, read_parquet, table_to_dicts

logger = logging.getLogger(__name__)


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
            source=data.get("source", "vividseats"),
        )


class ListingRepository:
    """Repository for seat-level ticket listings."""

    def __init__(self, data_dir: Path) -> None:
        """Initialize the repository.

        Args:
            data_dir: Base directory for data storage
        """
        self._data_dir = data_dir
        self._listings_dir = data_dir / "raw" / "listings"
        self._dedup_hashes_file = data_dir / "raw" / "listings" / ".listing_hashes.txt"
        self._cached_hashes: set[str] | None = None

    @staticmethod
    def compute_listing_hash(listing: TicketListing) -> str:
        """Compute a hash for a listing based on its composite key.

        Composite key: (event_id, section, row, seat_from, seat_to, source)

        This identifies the "same" listing even if scraped multiple times.

        Args:
            listing: The listing to hash

        Returns:
            MD5 hash string of the composite key
        """
        key_parts = [
            listing.event_id,
            listing.section.lower().strip(),
            (listing.row or "").lower().strip(),
            str(listing.seat_from or ""),
            str(listing.seat_to or ""),
            listing.source.lower().strip(),
            f"{listing.listing_price:.2f}",  # Include price to preserve price change signal
        ]
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _load_existing_hashes(self) -> set[str]:
        """Load existing listing hashes from disk.

        Returns:
            Set of known listing hashes
        """
        if self._cached_hashes is not None:
            return self._cached_hashes

        hashes: set[str] = set()

        if self._dedup_hashes_file.exists():
            with open(self._dedup_hashes_file) as f:
                hashes = {line.strip() for line in f if line.strip()}

        self._cached_hashes = hashes
        return hashes

    def _save_hashes(self, new_hashes: set[str]) -> None:
        """Append new hashes to the dedup file.

        Args:
            new_hashes: New hashes to save
        """
        self._dedup_hashes_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self._dedup_hashes_file, "a") as f:
            for h in new_hashes:
                f.write(h + "\n")

        # Update cache
        if self._cached_hashes is not None:
            self._cached_hashes.update(new_hashes)

        self._compact_hashes_if_needed()

    def _compact_hashes_if_needed(self, threshold: int = 500_000) -> None:
        """Compact hash file by deduplicating when cache exceeds threshold."""
        if not self._dedup_hashes_file.exists():
            return
        if self._cached_hashes is not None and len(self._cached_hashes) <= threshold:
            return
        # Fallback: count lines if cache not populated
        if self._cached_hashes is None:
            with open(self._dedup_hashes_file) as _f:
                line_count = sum(1 for _ in _f)
            if line_count <= threshold:
                return
        unique = self._load_existing_hashes()
        tmp = self._dedup_hashes_file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            for h in unique:
                f.write(h + "\n")
        tmp.rename(self._dedup_hashes_file)
        logger.info(f"Compacted hash file to {len(unique)} unique lines")

    def save_listings(self, listings: list[TicketListing], deduplicate: bool = True) -> int:
        """Save listings to storage, partitioned by date.

        Args:
            listings: List of listings to save
            deduplicate: If True, filter out listings already stored

        Returns:
            Number of new listings saved
        """
        if not listings:
            return 0

        # Deduplicate if requested
        if deduplicate:
            existing_hashes = self._load_existing_hashes()
            new_listings = []
            new_hashes = set()

            for listing in listings:
                listing_hash = self.compute_listing_hash(listing)

                if listing_hash not in existing_hashes and listing_hash not in new_hashes:
                    new_listings.append(listing)
                    new_hashes.add(listing_hash)

            if len(new_listings) < len(listings):
                duplicates = len(listings) - len(new_listings)
                logger.debug(f"Filtered {duplicates} duplicate listings")

            listings = new_listings

            # Save new hashes
            if new_hashes:
                self._save_hashes(new_hashes)

        if not listings:
            return 0

        # Group listings by date (year/month/day)
        by_partition: dict[tuple[int, int, int], list[TicketListing]] = {}
        for listing in listings:
            key = (
                listing.timestamp.year,
                listing.timestamp.month,
                listing.timestamp.day,
            )
            if key not in by_partition:
                by_partition[key] = []
            by_partition[key].append(listing)

        total_saved = 0
        for (year, month, day), partition_listings in by_partition.items():
            path = (
                self._listings_dir
                / f"year={year}"
                / f"month={month:02d}"
                / f"day={day:02d}"
                / "listings.parquet"
            )
            append_parquet(partition_listings, path, TicketListing.parquet_schema())
            total_saved += len(partition_listings)

        return total_saved

    def clear_dedup_cache(self) -> None:
        """Clear the deduplication hash cache (useful for testing)."""
        if self._dedup_hashes_file.exists():
            self._dedup_hashes_file.unlink()
        self._cached_hashes = None

    def get_listings(
        self,
        event_id: str | None = None,
        section: str | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
    ) -> list[TicketListing]:
        """Get listings matching criteria.

        Args:
            event_id: Optional event ID filter
            section: Optional section filter
            min_price: Optional minimum price filter
            max_price: Optional maximum price filter

        Returns:
            List of matching listings
        """
        if not self._listings_dir.exists():
            return []

        all_listings: list[TicketListing] = []

        # Read all partition files
        for parquet_file in self._listings_dir.rglob("*.parquet"):
            pq_filters: list[tuple[str, str, Any]] = []
            if event_id:
                pq_filters.append(("event_id", "=", event_id))

            table = read_parquet(parquet_file, filters=pq_filters if pq_filters else None)

            if table is None:
                continue

            records = table_to_dicts(table)
            for record in records:
                listing = self._dict_to_listing(record)

                # Apply additional filters
                if section and listing.section != section:
                    continue
                if min_price and listing.listing_price < min_price:
                    continue
                if max_price and listing.listing_price > max_price:
                    continue

                all_listings.append(listing)

        # Sort by timestamp (newest first)
        all_listings.sort(key=lambda x: x.timestamp, reverse=True)

        return all_listings

    def get_price_history(
        self,
        event_id: str,
        section: str | None = None,
        row: str | None = None,
    ) -> list[TicketListing]:
        """Get price history for an event, optionally filtered by section/row.

        Args:
            event_id: Event ID to look up
            section: Optional section filter
            row: Optional row filter

        Returns:
            List of listings sorted by timestamp (oldest first)
        """
        listings = self.get_listings(event_id=event_id, section=section)

        if row:
            listings = [lst for lst in listings if lst.row == row]

        # Sort by timestamp (oldest first for history)
        listings.sort(key=lambda x: x.timestamp)

        return listings

    def get_latest_listings(
        self,
        event_id: str,
        limit: int = 100,
    ) -> list[TicketListing]:
        """Get the most recent listings for an event.

        Args:
            event_id: Event ID to look up
            limit: Maximum number of listings to return

        Returns:
            List of latest listings
        """
        listings = self.get_listings(event_id=event_id)
        return listings[:limit]

    def list_event_ids(self) -> list[str]:
        """Get all unique event IDs in the listings.

        Returns:
            List of unique event IDs
        """
        if not self._listings_dir.exists():
            return []

        event_ids: set[str] = set()

        for parquet_file in self._listings_dir.rglob("*.parquet"):
            table = read_parquet(parquet_file)
            if table is not None:
                ids = cast(list[str], table.column("event_id").to_pylist())
                event_ids.update(ids)

        return list(event_ids)

    def _dict_to_listing(self, data: dict[str, Any]) -> TicketListing:
        """Convert a dictionary to TicketListing."""
        # Handle timezone-naive datetimes from Parquet
        timestamp = data["timestamp"]
        if isinstance(timestamp, datetime) and timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)

        event_datetime = data["event_datetime"]
        if isinstance(event_datetime, datetime) and event_datetime.tzinfo is None:
            event_datetime = event_datetime.replace(tzinfo=UTC)

        return TicketListing(
            listing_id=data["listing_id"],
            event_id=data["event_id"],
            source=data.get("source", "stubhub"),
            timestamp=timestamp,
            event_name=data["event_name"],
            artist_or_team=data["artist_or_team"],
            venue_name=data["venue_name"],
            city=data["city"],
            event_datetime=event_datetime,
            event_type=data.get("event_type"),
            section=data["section"],
            row=data["row"],
            seat_from=data.get("seat_from"),
            seat_to=data.get("seat_to"),
            quantity=data["quantity"],
            face_value=data.get("face_value"),
            listing_price=data["listing_price"],
            total_price=data["total_price"],
            currency=data.get("currency", "USD"),
            days_to_event=data["days_to_event"],
        )
