"""Sale label construction from temporal listing and snapshot data.

Two labeling strategies are available:

1. SaleLabelBuilder: Disappearance-based labels
   Constructs binary "sold" labels by detecting listing_id disappearance across
   consecutive hourly scraping runs. A listing that vanishes from 2+ consecutive
   scrapes within a 24-hour window is labeled as sold (converted).

   Known limitation: disappearance conflates sold, delisted, and scraper errors.
   The scraper also captures partial inventory (median ~2 listings/scrape), so
   individual listing tracking has fundamental reliability limits.

2. InventoryDepletionLabeler: Aggregate inventory signal
   Labels listings based on whether the overall event inventory depleted
   significantly within a time window. Uses PriceSnapshot.inventory_remaining
   (zone-level aggregate), which is robust to the scraper's partial capture.
   All listings in a qualifying event that shows >30% depletion are labeled sold=1.

   Preferred strategy when snapshot data is available.

Both strategies are structurally analogous to CVR label construction in commerce:
  - listing appears → product is "exposed" (impression)
  - listing disappears / inventory depletes → product "converted" (purchase)
"""

from __future__ import annotations

import logging
import warnings
from datetime import UTC, timedelta

import pandas as pd

logger = logging.getLogger(__name__)


class SaleLabelBuilder:
    """Builds binary sale labels from temporal listing data.

    Labels are derived from listing_id disappearance across scraping timestamps.
    Calibrated for hourly scrape cadence (window_hours=24 ≈ 24 scrape cycles).
    """

    def verify_listing_id_stability(self, listings_df: pd.DataFrame) -> dict[str, float | int]:
        """Check whether listing_ids are stable across scraping runs.

        A stable listing_id means the same physical listing retains its ID
        across multiple scraping timestamps — required for disappearance-based
        label construction.

        Args:
            listings_df: DataFrame with at least listing_id, event_id, timestamp columns

        Returns:
            Dict with keys: total_listings, stable_ids, unstable_ids, stability_ratio
            stable_ids = listing_ids that appear in 2+ timestamps for the same event
        """
        required = {"listing_id", "event_id", "timestamp"}
        missing = required - set(listings_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if listings_df.empty:
            return {
                "total_listings": 0,
                "stable_ids": 0,
                "unstable_ids": 0,
                "stability_ratio": 0.0,
            }

        # Count how many distinct timestamps each listing_id appears in per event
        counts = (
            listings_df.groupby(["event_id", "listing_id"])["timestamp"]
            .nunique()
            .reset_index(name="timestamp_count")
        )

        stable = int((counts["timestamp_count"] >= 2).sum())
        unstable = int((counts["timestamp_count"] < 2).sum())
        total = stable + unstable
        ratio = stable / total if total > 0 else 0.0

        if ratio < 0.5:
            warnings.warn(
                f"Low listing_id stability ratio: {ratio:.2%}. "
                "Listing IDs may not persist across scrapes. "
                "Consider matching on (event_id, section, row, listing_price) instead.",
                UserWarning,
                stacklevel=2,
            )

        return {
            "total_listings": total,
            "stable_ids": stable,
            "unstable_ids": unstable,
            "stability_ratio": ratio,
        }

    def build_labels(
        self,
        listings_df: pd.DataFrame,
        window_hours: int = 24,
        min_absent_scrapes: int = 2,
        min_scrapes: int = 5,
    ) -> pd.DataFrame:
        """Build binary sold/not-sold labels from listing disappearance.

        For each listing observed at timestamp T, checks whether its listing_id
        is absent from min_absent_scrapes or more consecutive scrape timestamps
        within the window (T, T + window_hours].

        Labels:
            sold=1: listing absent from 2+ consecutive scrapes within window
            sold=0: listing still present within the window

        Excluded rows:
            - Listings in the final window_hours of data (no future to compare)
            - Post-event listings (event_datetime < observation timestamp)
            - Listings appearing in only one scrape (cannot determine outcome)
            - Events with fewer than min_scrapes scrape timestamps

        Timestamps are processed per-event to avoid cross-event contamination.
        The cutoff (max_timestamp - window_hours) is also computed per-event.

        Args:
            listings_df: DataFrame with listing_id, event_id, timestamp,
                         event_datetime columns
            window_hours: Hours to look ahead for disappearance. Default 24
                         (calibrated for hourly scrape cadence: ~24 cycles).
                         Adjust if scrape frequency changes.
            min_absent_scrapes: Minimum consecutive absent scrapes to label as
                                sold. Default 2 reduces false positives from
                                transient scraper failures.
            min_scrapes: Minimum number of distinct scrape timestamps an event
                        must have to qualify for label construction. Events
                        with fewer scrapes are excluded entirely. Default 5.

        Returns:
            listings_df with added columns:
                sold (int): 1 if listing sold, 0 if still active
                hours_until_disappearance (float): hours until first absence, NaN if still active
        """
        required = {"listing_id", "event_id", "timestamp", "event_datetime"}
        missing = required - set(listings_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if listings_df.empty:
            result = listings_df.copy()
            result["sold"] = pd.Series(dtype=int)
            result["hours_until_disappearance"] = pd.Series(dtype=float)
            return result

        df = listings_df.copy()

        # Ensure timestamps are timezone-aware
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(UTC)
        if df["event_datetime"].dt.tz is None:
            df["event_datetime"] = df["event_datetime"].dt.tz_localize(UTC)

        # Exclude post-event listings (scraper cleanup, not sales)
        df = df[df["timestamp"] < df["event_datetime"]].copy()

        if df.empty:
            result = df.copy()
            result["sold"] = pd.Series(dtype=int)
            result["hours_until_disappearance"] = pd.Series(dtype=float)
            return result

        # Process each event independently to avoid cross-event timestamp contamination
        result_parts: list[pd.DataFrame] = []

        for _event_id, event_group in df.groupby("event_id"):
            # Per-event timestamps (not global)
            event_timestamps = sorted(event_group["timestamp"].unique())

            # Density guard: skip events with too few scrapes
            if len(event_timestamps) < min_scrapes:
                continue

            # Per-event cutoff: exclude listings in the final window_hours of THIS event's data
            event_max_ts = event_group["timestamp"].max()
            event_cutoff = event_max_ts - timedelta(hours=window_hours)
            event_df = event_group[event_group["timestamp"] <= event_cutoff].copy()

            if event_df.empty:
                continue

            # Build a set of (listing_id, timestamp) that exist for this event
            event_existing: set[tuple[str, pd.Timestamp]] = set(
                zip(event_group["listing_id"], event_group["timestamp"], strict=False)
            )

            sold_labels: list[int] = []
            hours_disappeared: list[float | None] = []

            for _, row in event_df.iterrows():
                listing_id: str = row["listing_id"]
                t_obs: pd.Timestamp = row["timestamp"]
                window_end = t_obs + timedelta(hours=window_hours)

                # Get timestamps after t_obs within the window for this event
                future_ts = [t for t in event_timestamps if t_obs < t <= window_end]

                if len(future_ts) < min_absent_scrapes:
                    # Insufficient future scrapes — cannot determine outcome
                    sold_labels.append(0)
                    hours_disappeared.append(None)
                    continue

                # Find consecutive absences
                consecutive_absent = 0
                first_absence_ts: pd.Timestamp | None = None

                for ts in future_ts:
                    if (listing_id, ts) not in event_existing:
                        if first_absence_ts is None:
                            first_absence_ts = ts
                        consecutive_absent += 1
                        if consecutive_absent >= min_absent_scrapes:
                            break
                    else:
                        # Reset on reappearance (transient absence, not a sale)
                        consecutive_absent = 0
                        first_absence_ts = None

                if consecutive_absent >= min_absent_scrapes and first_absence_ts is not None:
                    sold_labels.append(1)
                    hours = (first_absence_ts - t_obs).total_seconds() / 3600
                    hours_disappeared.append(hours)
                else:
                    sold_labels.append(0)
                    hours_disappeared.append(None)

            event_df = event_df.copy()
            event_df["sold"] = sold_labels
            event_df["hours_until_disappearance"] = hours_disappeared
            result_parts.append(event_df)

        if not result_parts:
            empty = df.iloc[:0].copy()
            empty["sold"] = pd.Series(dtype=int)
            empty["hours_until_disappearance"] = pd.Series(dtype=float)
            return empty

        return pd.concat(result_parts, ignore_index=True)


class InventoryDepletionLabeler:
    """Labels listings using event-level inventory depletion from snapshot data.

    Robust to partial scraping because it uses aggregate inventory counts
    from PriceSnapshot rather than individual listing tracking.

    For each listing, computes whether the event's total inventory (sum across
    all seat zones) depleted by more than `depletion_threshold` within
    `window_hours` of the listing's observation time. Listings in events that
    deplete significantly are labeled sold=1; others sold=0.

    Note: Labels are at event-level granularity — all listings in the same
    event share the same depletion rate and label. This is intentional: with
    the current scraper's partial captures, event-level aggregate signals are
    more reliable than individual-listing signals.

    Note: Raw listings lack `seat_zone` at label time (it's computed during
    feature extraction), so aggregation must be event-level (not zone-level).
    """

    def __init__(
        self,
        window_hours: int = 48,
        depletion_threshold: float = 0.3,
        min_snapshots: int = 5,
    ) -> None:
        """Initialize labeler.

        Args:
            window_hours: Lookahead window for depletion measurement. Default 48h
                         (wider than listing window because snapshot cadence is
                         less frequent than listing scrapes).
            depletion_threshold: Fraction of inventory that must deplete to label
                                sold=1. Default 0.3 (30% depletion). Data analysis
                                shows this yields ~33% positive rate.
            min_snapshots: Minimum number of distinct snapshot timestamps an event
                          must have to qualify for label construction. Default 5.
        """
        self._window_hours = window_hours
        self._depletion_threshold = depletion_threshold
        self._min_snapshots = min_snapshots

    def build_labels(
        self,
        listings_df: pd.DataFrame,
        snapshots_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build binary sold/not-sold labels from event-level inventory depletion.

        For each listing observed at timestamp T in event E:
        1. Find total event inventory at T (latest snapshot at or before T)
        2. Find total event inventory at T + window_hours (latest snapshot before window end)
        3. depletion_rate = (inv_at_obs - inv_at_future) / inv_at_obs
        4. sold = 1 if depletion_rate > threshold, else 0

        Listings in events without sufficient snapshots are excluded.

        Args:
            listings_df: DataFrame with event_id, timestamp, event_datetime columns
            snapshots_df: DataFrame with event_id, timestamp, inventory_remaining columns

        Returns:
            Subset of listings_df (qualifying events only) with added columns:
                sold (int): 1 if inventory depleted, 0 otherwise
                _label_depletion_rate (float): event depletion rate used for labeling.
                    Prefixed with _label_ to signal it must NOT be used as a feature
                    — drop this column before feature extraction to prevent label leakage.
        """
        required_listings = {"event_id", "timestamp", "event_datetime"}
        missing_listings = required_listings - set(listings_df.columns)
        if missing_listings:
            raise ValueError(f"Missing required listing columns: {missing_listings}")

        required_snapshots = {"event_id", "timestamp", "inventory_remaining"}
        missing_snapshots = required_snapshots - set(snapshots_df.columns)
        if missing_snapshots:
            raise ValueError(f"Missing required snapshot columns: {missing_snapshots}")

        if listings_df.empty or snapshots_df.empty:
            result = listings_df.iloc[:0].copy()
            result["sold"] = pd.Series(dtype=int)
            result["_label_depletion_rate"] = pd.Series(dtype=float)
            return result

        listings = listings_df.copy()
        snapshots = snapshots_df.copy()

        # Ensure timestamps are timezone-aware
        for col in ["timestamp", "event_datetime"]:
            if col in listings.columns and listings[col].dt.tz is None:
                listings[col] = listings[col].dt.tz_localize(UTC)
        if snapshots["timestamp"].dt.tz is None:
            snapshots["timestamp"] = snapshots["timestamp"].dt.tz_localize(UTC)

        # Exclude post-event listings
        listings = listings[listings["timestamp"] < listings["event_datetime"]].copy()

        if listings.empty:
            result = listings.iloc[:0].copy()
            result["sold"] = pd.Series(dtype=int)
            result["_label_depletion_rate"] = pd.Series(dtype=float)
            return result

        # Step 0: Drop null inventory (Ticketmaster snapshots may have inventory_remaining=None)
        null_count = snapshots["inventory_remaining"].isna().sum()
        if null_count > 0:
            logger.info(
                "Dropping %d snapshot rows with null inventory_remaining before aggregation",
                null_count,
            )
            snapshots = snapshots.dropna(subset=["inventory_remaining"])

        # Aggregate snapshots to event-level: sum across all zones per event+timestamp
        event_snapshots = (
            snapshots.groupby(["event_id", "timestamp"])["inventory_remaining"]
            .sum()
            .reset_index(name="total_inventory")
        )

        # Filter to events with >= min_snapshots distinct timestamps
        event_snapshot_counts = event_snapshots.groupby("event_id")["timestamp"].nunique()
        qualifying_events = event_snapshot_counts[
            event_snapshot_counts >= self._min_snapshots
        ].index
        event_snapshots = event_snapshots[event_snapshots["event_id"].isin(qualifying_events)]

        # Filter listings to qualifying events only
        listings = listings[listings["event_id"].isin(qualifying_events)].copy()

        if listings.empty or event_snapshots.empty:
            result = listings.iloc[:0].copy()
            result["sold"] = pd.Series(dtype=int)
            result["_label_depletion_rate"] = pd.Series(dtype=float)
            return result

        # Sort both DataFrames by timestamp — merge_asof requires the on column
        # (timestamp) to be globally sorted. by="event_id" handles per-event grouping.
        listings = listings.sort_values("timestamp").reset_index(drop=True)
        event_snapshots = event_snapshots.sort_values("timestamp").reset_index(drop=True)

        # Step 1: Find inv_at_obs — latest snapshot at or before each listing's timestamp
        merged_obs = pd.merge_asof(
            listings,
            event_snapshots.rename(columns={"total_inventory": "inv_at_obs"}),
            by="event_id",
            on="timestamp",
            direction="backward",
        )

        # Step 2: Find inv_at_future — latest snapshot at or before (listing_timestamp + window)
        # Create a temporary copy with shifted timestamp for the future lookup only
        listings_shifted = listings[["event_id", "timestamp"]].copy()
        listings_shifted["timestamp_shifted"] = listings_shifted["timestamp"] + timedelta(
            hours=self._window_hours
        )
        listings_shifted = listings_shifted.sort_values("timestamp_shifted")

        event_snapshots_future = event_snapshots.rename(
            columns={"timestamp": "timestamp_shifted", "total_inventory": "inv_at_future"}
        )

        merged_future = pd.merge_asof(
            listings_shifted,
            event_snapshots_future,
            by="event_id",
            on="timestamp_shifted",
            direction="backward",
        )

        # Join future inventory back to obs DataFrame on (event_id, original timestamp).
        # Deduplicate on (event_id, timestamp) to prevent M:M join — multiple listings
        # sharing the same scrape timestamp get the same inv_at_future value.
        future_lookup = (
            merged_future[["event_id", "timestamp", "inv_at_future"]]
            .drop_duplicates(subset=["event_id", "timestamp"])
        )
        result = merged_obs.merge(future_lookup, on=["event_id", "timestamp"], how="left")

        # Drop listings where we couldn't find inventory at either time
        result = result.dropna(subset=["inv_at_obs", "inv_at_future"])

        if result.empty:
            result["sold"] = pd.Series(dtype=int)
            result["_label_depletion_rate"] = pd.Series(dtype=float)
            return result

        # Step 3: Compute depletion rate (guard against division by zero)
        result["_label_depletion_rate"] = 0.0
        has_inventory_mask = result["inv_at_obs"] > 0
        result.loc[has_inventory_mask, "_label_depletion_rate"] = (
            result.loc[has_inventory_mask, "inv_at_obs"]
            - result.loc[has_inventory_mask, "inv_at_future"]
        ) / result.loc[has_inventory_mask, "inv_at_obs"]

        # Step 4: Label
        result["sold"] = (result["_label_depletion_rate"] > self._depletion_threshold).astype(int)

        # Drop temporary columns used only for the snapshot join
        result = result.drop(columns=["inv_at_obs", "inv_at_future"], errors="ignore")

        return result
