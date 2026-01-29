"""Synthetic price trajectory generator for historical events.

Generates realistic price curves based on known ticket pricing patterns:
- Prices typically increase as event date approaches
- High-demand events (popular artists) have steeper curves
- Different seat zones have different base prices
- Some randomness to simulate market dynamics
"""

import random
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from ticket_price_predictor.api.setlistfm import HistoricalConcert
from ticket_price_predictor.schemas import EventMetadata, EventType, PriceSnapshot, SeatZone


@dataclass
class ArtistProfile:
    """Profile defining pricing characteristics for an artist."""

    name: str
    popularity: float  # 0.0 to 1.0 (1.0 = most popular)
    base_price_min: float  # Starting price for cheapest seats
    base_price_max: float  # Starting price for premium seats
    demand_multiplier: float = 1.0  # How much prices increase near event
    volatility: float = 0.1  # Random price fluctuation factor

    @classmethod
    def from_artist_name(cls, name: str) -> "ArtistProfile":
        """Create a profile based on artist name (uses heuristics)."""
        name_lower = name.lower()

        # K-pop artists - high demand, high prices
        kpop_artists = ["blackpink", "bts", "twice", "stray kids", "aespa", "newjeans"]
        if any(kpop in name_lower for kpop in kpop_artists):
            return cls(
                name=name,
                popularity=0.95,
                base_price_min=150.0,
                base_price_max=800.0,
                demand_multiplier=2.5,
                volatility=0.15,
            )

        # Major pop/rock artists
        major_artists = ["taylor swift", "beyonce", "coldplay", "ed sheeran", "eagles"]
        if any(artist in name_lower for artist in major_artists):
            return cls(
                name=name,
                popularity=0.9,
                base_price_min=100.0,
                base_price_max=500.0,
                demand_multiplier=2.0,
                volatility=0.12,
            )

        # Mid-tier artists
        return cls(
            name=name,
            popularity=0.5,
            base_price_min=40.0,
            base_price_max=200.0,
            demand_multiplier=1.5,
            volatility=0.1,
        )


@dataclass
class PriceTrajectory:
    """A price trajectory for an event over time."""

    event: EventMetadata
    snapshots: list[PriceSnapshot] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event": self.event.model_dump(),
            "snapshots": [s.model_dump() for s in self.snapshots],
        }


class SyntheticPriceGenerator:
    """Generates synthetic but realistic price trajectories.

    Based on research on ticket pricing patterns:
    - Prices start lower when tickets first go on sale
    - Gradual increase as event approaches
    - Steeper increase in final 2 weeks
    - High-demand events can see 2-3x price multiplier
    - Different zones follow similar patterns but at different price points
    """

    # Zone price ratios relative to max price
    ZONE_RATIOS: dict[SeatZone, float] = {
        SeatZone.FLOOR_VIP: 1.0,
        SeatZone.LOWER_TIER: 0.65,
        SeatZone.UPPER_TIER: 0.40,
        SeatZone.BALCONY: 0.25,
    }

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the generator.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

    def generate_trajectory(
        self,
        concert: HistoricalConcert,
        profile: ArtistProfile | None = None,
        snapshot_days: list[int] | None = None,
    ) -> PriceTrajectory:
        """Generate a price trajectory for a historical concert.

        Args:
            concert: Historical concert from setlist.fm
            profile: Artist pricing profile (auto-generated if None)
            snapshot_days: Days before event to generate snapshots
                          (default: [90, 60, 30, 14, 7, 3, 1, 0])

        Returns:
            PriceTrajectory with event metadata and price snapshots
        """
        if profile is None:
            profile = ArtistProfile.from_artist_name(concert.artist_name)

        if snapshot_days is None:
            snapshot_days = [90, 60, 30, 14, 7, 3, 1, 0]

        # Create event metadata
        event = EventMetadata(
            event_id=f"syn_{concert.event_id}",
            event_type=EventType.CONCERT,
            event_datetime=concert.event_date,
            artist_or_team=concert.artist_name,
            venue_id=f"syn_{concert.venue_name[:10]}".replace(" ", "_"),
            venue_name=concert.venue_name,
            city=concert.city,
            country=concert.country,
        )

        # Generate snapshots
        snapshots: list[PriceSnapshot] = []

        for days_before in sorted(snapshot_days, reverse=True):
            if days_before < 0:
                continue

            snapshot_time = concert.event_date - timedelta(days=days_before)

            # Generate price for each zone
            for zone in SeatZone:
                price = self._calculate_price(
                    profile=profile,
                    zone=zone,
                    days_to_event=days_before,
                )

                snapshot = PriceSnapshot(
                    event_id=event.event_id,
                    seat_zone=zone,
                    timestamp=snapshot_time,
                    price_min=price * 0.9,  # Slight range
                    price_avg=price,
                    price_max=price * 1.1,
                    days_to_event=days_before,
                )
                snapshots.append(snapshot)

        return PriceTrajectory(event=event, snapshots=snapshots)

    def _calculate_price(
        self,
        profile: ArtistProfile,
        zone: SeatZone,
        days_to_event: int,
    ) -> float:
        """Calculate price for a zone at a specific time.

        Price curve formula:
        - Base price determined by zone ratio
        - Time multiplier increases as event approaches
        - Random noise added for realism
        """
        # Base price for this zone
        zone_ratio = self.ZONE_RATIOS[zone]
        base_price = profile.base_price_min + (
            (profile.base_price_max - profile.base_price_min) * zone_ratio
        )

        # Time-based multiplier
        # Prices increase following a curve that accelerates near event
        if days_to_event > 60:
            time_multiplier = 1.0
        elif days_to_event > 30:
            # Gradual increase: 1.0 to 1.2
            progress = (60 - days_to_event) / 30
            time_multiplier = 1.0 + (0.2 * progress)
        elif days_to_event > 14:
            # Moderate increase: 1.2 to 1.5
            progress = (30 - days_to_event) / 16
            time_multiplier = 1.2 + (0.3 * progress)
        elif days_to_event > 7:
            # Faster increase: 1.5 to 1.8
            progress = (14 - days_to_event) / 7
            time_multiplier = 1.5 + (0.3 * progress)
        else:
            # Final week: 1.8 to demand_multiplier
            progress = (7 - days_to_event) / 7
            time_multiplier = 1.8 + ((profile.demand_multiplier - 1.8) * progress)

        # Apply demand multiplier based on popularity
        popularity_boost = 1.0 + (profile.popularity * 0.5)

        # Add random noise
        noise = 1.0 + random.uniform(-profile.volatility, profile.volatility)

        final_price = base_price * time_multiplier * popularity_boost * noise

        # Round to reasonable precision
        return round(final_price, 2)

    def generate_batch(
        self,
        concerts: list[HistoricalConcert],
        profile: ArtistProfile | None = None,
    ) -> list[PriceTrajectory]:
        """Generate trajectories for multiple concerts.

        Args:
            concerts: List of historical concerts
            profile: Shared artist profile (auto-generated per artist if None)

        Returns:
            List of PriceTrajectory objects
        """
        trajectories: list[PriceTrajectory] = []

        for concert in concerts:
            # Use provided profile or generate based on artist
            artist_profile = profile or ArtistProfile.from_artist_name(concert.artist_name)
            trajectory = self.generate_trajectory(concert, artist_profile)
            trajectories.append(trajectory)

        return trajectories
