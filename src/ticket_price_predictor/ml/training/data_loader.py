"""Data loading utilities for ML training."""

from pathlib import Path

import pandas as pd

from ticket_price_predictor.schemas.listings import TicketListing
from ticket_price_predictor.storage import ListingRepository, SnapshotRepository


class DataLoader:
    """Load and prepare data for ML training."""

    def __init__(self, data_dir: Path | str = Path("data")) -> None:
        """Initialize data loader.

        Args:
            data_dir: Path to data directory
        """
        self._data_dir = Path(data_dir)
        self._listing_repo = ListingRepository(self._data_dir)
        self._snapshot_repo = SnapshotRepository(self._data_dir)

    def load_listings(
        self,
        event_id: str | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
    ) -> pd.DataFrame:
        """Load listings as a DataFrame.

        Args:
            event_id: Filter by event ID
            min_price: Minimum price filter
            max_price: Maximum price filter

        Returns:
            DataFrame with listing data
        """
        listings = self._listing_repo.get_listings(
            event_id=event_id,
            min_price=min_price,
            max_price=max_price,
        )

        if not listings:
            return pd.DataFrame()

        return self._listings_to_dataframe(listings)

    def load_all_listings(self) -> pd.DataFrame:
        """Load all listings as a DataFrame.

        Returns:
            DataFrame with all listing data
        """
        listings = self._listing_repo.get_listings()
        if not listings:
            return pd.DataFrame()
        return self._listings_to_dataframe(listings)

    def _listings_to_dataframe(self, listings: list[TicketListing]) -> pd.DataFrame:
        """Convert list of TicketListing to DataFrame."""
        records = []
        for listing in listings:
            record = {
                "listing_id": listing.listing_id,
                "event_id": listing.event_id,
                "event_name": listing.event_name,
                "artist_or_team": listing.artist_or_team,
                "venue_name": listing.venue_name,
                "city": listing.city,
                "event_datetime": listing.event_datetime,
                "timestamp": listing.timestamp,
                "section": listing.section,
                "row": listing.row,
                "seat_from": listing.seat_from,
                "seat_to": listing.seat_to,
                "quantity": listing.quantity,
                "face_value": listing.face_value,
                "listing_price": listing.listing_price,
                "total_price": listing.total_price,
                "days_to_event": listing.days_to_event,
                "source": listing.source,
            }
            # Add computed fields if available
            if hasattr(listing, "markup_ratio"):
                record["markup_ratio"] = listing.markup_ratio
            records.append(record)

        df = pd.DataFrame(records)

        # Convert datetime columns
        if "event_datetime" in df.columns:
            df["event_datetime"] = pd.to_datetime(df["event_datetime"])

        return df

    def load_snapshots(self) -> pd.DataFrame:
        """Load all price snapshots as a DataFrame.

        Returns raw snapshot data — no joining or aggregation. The
        snapshot-to-listing join happens in the trainer after splitting
        to preserve the split-before-fit invariant.

        Returns:
            DataFrame with columns: event_id, seat_zone, timestamp,
            price_min, price_avg, price_max, inventory_remaining, days_to_event
        """
        snapshots = self._snapshot_repo.get_snapshots()
        if not snapshots:
            return pd.DataFrame()

        records = [
            {
                "event_id": s.event_id,
                "seat_zone": s.seat_zone.value,
                "timestamp": s.timestamp,
                "price_min": s.price_min,
                "price_avg": s.price_avg,
                "price_max": s.price_max,
                "inventory_remaining": s.inventory_remaining,
                "days_to_event": s.days_to_event,
            }
            for s in snapshots
        ]
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def get_summary(self) -> dict[str, int | float]:
        """Get summary statistics of available data.

        Returns:
            Dictionary with summary stats
        """
        df = self.load_all_listings()

        if df.empty:
            return {
                "n_listings": 0,
                "n_events": 0,
                "n_artists": 0,
                "price_min": 0,
                "price_max": 0,
                "price_mean": 0,
            }

        return {
            "n_listings": len(df),
            "n_events": df["event_id"].nunique(),
            "n_artists": df["artist_or_team"].nunique(),
            "price_min": df["listing_price"].min(),
            "price_max": df["listing_price"].max(),
            "price_mean": df["listing_price"].mean(),
        }
