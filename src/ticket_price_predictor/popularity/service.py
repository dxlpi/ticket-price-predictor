"""Main facade for popularity data aggregation."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from ticket_price_predictor.popularity.aggregator import (
    ArtistPopularity,
    PopularityAggregator,
    PopularityTier,
)
from ticket_price_predictor.popularity.bandsintown import BandsintownPopularity
from ticket_price_predictor.popularity.cache import PopularityCache
from ticket_price_predictor.popularity.songkick import SongkickPopularity
from ticket_price_predictor.popularity.spotify import SpotifyPopularity

logger = logging.getLogger(__name__)


class PopularityService:
    """Main service for fetching and aggregating artist popularity data."""

    def __init__(
        self,
        spotify_client_id: str | None = None,
        spotify_client_secret: str | None = None,
        songkick_api_key: str | None = None,
        bandsintown_app_id: str | None = None,
        cache_dir: Path | str | None = None,
        cache_ttl_hours: int = 24,
    ) -> None:
        """Initialize popularity service with API clients.

        Args:
            spotify_client_id: Spotify API client ID (or SPOTIFY_CLIENT_ID env var)
            spotify_client_secret: Spotify API client secret (or SPOTIFY_CLIENT_SECRET env var)
            songkick_api_key: Songkick API key (or SONGKICK_API_KEY env var)
            bandsintown_app_id: Bandsintown app ID (or BANDSINTOWN_APP_ID env var)
            cache_dir: Directory for caching results
            cache_ttl_hours: Cache time-to-live in hours
        """
        # Initialize clients with env var fallbacks
        spotify_id = spotify_client_id or os.getenv("SPOTIFY_CLIENT_ID")
        spotify_secret = spotify_client_secret or os.getenv("SPOTIFY_CLIENT_SECRET")

        self.spotify: SpotifyPopularity | None = None
        if spotify_id and spotify_secret:
            self.spotify = SpotifyPopularity(spotify_id, spotify_secret)

        songkick_key = songkick_api_key or os.getenv("SONGKICK_API_KEY")
        self.songkick: SongkickPopularity | None = None
        if songkick_key:
            self.songkick = SongkickPopularity(songkick_key)

        bandsintown_id = bandsintown_app_id or os.getenv("BANDSINTOWN_APP_ID")
        self.bandsintown: BandsintownPopularity | None = None
        if bandsintown_id:
            self.bandsintown = BandsintownPopularity(bandsintown_id)

        # Initialize aggregator
        self.aggregator = PopularityAggregator()

        # Initialize cache
        if cache_dir:
            self.cache = PopularityCache(cache_dir, cache_ttl_hours)
        else:
            default_cache = Path("data") / ".cache" / "popularity"
            self.cache = PopularityCache(default_cache, cache_ttl_hours)

        # Log available sources
        sources = []
        if self.spotify and self.spotify.available:
            sources.append("Spotify")
        if self.songkick and self.songkick.available:
            sources.append("Songkick")
        if self.bandsintown and self.bandsintown.available:
            sources.append("Bandsintown")

        if sources:
            logger.info(f"PopularityService initialized with sources: {', '.join(sources)}")
        else:
            logger.warning("PopularityService initialized with no API sources configured")

    def get_artist_popularity(
        self,
        artist_name: str,
        skip_cache: bool = False,
    ) -> ArtistPopularity:
        """Get popularity data for a single artist.

        Args:
            artist_name: Artist name to look up
            skip_cache: Skip cache lookup and fetch fresh data

        Returns:
            ArtistPopularity with aggregated score and tier
        """
        # Check cache first
        if not skip_cache:
            cached = self.cache.get(artist_name)
            if cached:
                logger.debug(f"Cache hit for {artist_name}")
                return ArtistPopularity(
                    name=str(cached["name"]),
                    popularity_score=float(cached["popularity_score"]),
                    tier=PopularityTier(str(cached["tier"])),
                    spotify_popularity=int(cached["spotify_popularity"])
                    if cached.get("spotify_popularity") is not None
                    else None,
                    spotify_followers=int(cached["spotify_followers"])
                    if cached.get("spotify_followers") is not None
                    else None,
                    songkick_trackers=int(cached["songkick_trackers"])
                    if cached.get("songkick_trackers") is not None
                    else None,
                    bandsintown_trackers=int(cached["bandsintown_trackers"])
                    if cached.get("bandsintown_trackers") is not None
                    else None,
                    sources_available=list(cached.get("sources_available", [])),
                )

        # Fetch from each source
        spotify_popularity = None
        spotify_followers = None
        songkick_trackers = None
        bandsintown_trackers = None

        if self.spotify and self.spotify.available:
            spotify_metrics = self.spotify.get_artist_metrics(artist_name)
            if spotify_metrics:
                spotify_popularity = spotify_metrics.popularity
                spotify_followers = spotify_metrics.followers

        if self.songkick and self.songkick.available:
            songkick_metrics = self.songkick.get_artist_metrics(artist_name)
            if songkick_metrics:
                songkick_trackers = songkick_metrics.tracker_count

        if self.bandsintown and self.bandsintown.available:
            bandsintown_metrics = self.bandsintown.get_artist_metrics(artist_name)
            if bandsintown_metrics:
                bandsintown_trackers = bandsintown_metrics.tracker_count

        # Calculate aggregated score
        popularity = self.aggregator.calculate_score(
            artist_name=artist_name,
            spotify_popularity=spotify_popularity,
            spotify_followers=spotify_followers,
            songkick_trackers=songkick_trackers,
            bandsintown_trackers=bandsintown_trackers,
        )

        # Cache result
        self.cache.set(artist_name, popularity)

        return popularity

    def rank_performers(
        self,
        performer_names: list[str],
        coverage_threshold: float = 0.80,
    ) -> list[ArtistPopularity]:
        """Rank performers by popularity and return top performers.

        Instead of hardcoded top N, returns performers whose cumulative
        popularity covers the specified threshold of total popularity.

        Args:
            performer_names: List of performer names to rank
            coverage_threshold: Cumulative score threshold (0.0-1.0)

        Returns:
            List of ArtistPopularity sorted by score (highest first)
        """
        # Get popularity for all performers
        popularities = []
        for name in performer_names:
            try:
                pop = self.get_artist_popularity(name)
                popularities.append(pop)
            except Exception as e:
                logger.warning(f"Failed to get popularity for {name}: {e}")
                # Create low-tier fallback
                popularities.append(
                    ArtistPopularity(
                        name=name,
                        popularity_score=0.0,
                        tier=PopularityTier.LOW,
                        sources_available=[],
                    )
                )

        # Sort by popularity score (highest first)
        popularities.sort(key=lambda p: p.popularity_score, reverse=True)

        # Calculate total score
        total_score = sum(p.popularity_score for p in popularities)

        if total_score == 0:
            # If no scores, return all performers
            return popularities

        # Select performers until cumulative score reaches threshold
        cumulative = 0.0
        selected = []

        for pop in popularities:
            selected.append(pop)
            cumulative += pop.popularity_score

            if cumulative / total_score >= coverage_threshold:
                break

        # Ensure we return at least 3 performers (minimum viable set)
        while len(selected) < min(3, len(popularities)):
            if len(selected) < len(popularities):
                selected.append(popularities[len(selected)])
            else:
                break

        logger.info(
            f"Selected {len(selected)}/{len(popularities)} performers "
            f"(coverage: {cumulative / total_score * 100:.1f}%)"
        )

        return selected

    def calculate_max_events(self, ranked_performers: list[ArtistPopularity]) -> int:
        """Calculate total max events based on tier allocations.

        Args:
            ranked_performers: List of ranked performers with tiers

        Returns:
            Total events to collect (sum of tier allocations)
        """
        return sum(p.tier_allocation for p in ranked_performers)
