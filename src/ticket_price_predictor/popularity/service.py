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
from ticket_price_predictor.popularity.cache import PopularityCache
from ticket_price_predictor.popularity.lastfm import LastfmPopularity
from ticket_price_predictor.popularity.youtube import YouTubePopularity

logger = logging.getLogger(__name__)


class PopularityService:
    """Main service for fetching and aggregating artist popularity data."""

    def __init__(
        self,
        lastfm_api_key: str | None = None,
        cache_dir: Path | str | None = None,
        cache_ttl_hours: int = 24,
    ) -> None:
        """Initialize popularity service with API clients.

        Args:
            lastfm_api_key: Last.fm API key (or LASTFM_API_KEY env var)
            cache_dir: Directory for caching results
            cache_ttl_hours: Cache time-to-live in hours
        """
        # Initialize YouTube Music client (no API key needed)
        self.youtube: YouTubePopularity | None = None
        try:
            yt = YouTubePopularity()
            if yt.available:
                self.youtube = yt
        except Exception as e:
            logger.warning(f"Failed to initialize YouTube Music client: {e}")

        # Initialize Last.fm client with env var fallback
        lastfm_key = lastfm_api_key or os.getenv("LASTFM_API_KEY")
        self.lastfm: LastfmPopularity | None = None
        if lastfm_key:
            self.lastfm = LastfmPopularity(lastfm_key)

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
        if self.youtube and self.youtube.available:
            sources.append("YouTube Music")
        if self.lastfm and self.lastfm.available:
            sources.append("Last.fm")

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
                    youtube_subscribers=int(cached["youtube_subscribers"])
                    if cached.get("youtube_subscribers") is not None
                    else None,
                    youtube_views=int(cached["youtube_views"])
                    if cached.get("youtube_views") is not None
                    else None,
                    lastfm_listeners=int(cached["lastfm_listeners"])
                    if cached.get("lastfm_listeners") is not None
                    else None,
                    lastfm_play_count=int(cached["lastfm_play_count"])
                    if cached.get("lastfm_play_count") is not None
                    else None,
                    sources_available=list(cached.get("sources_available", [])),
                )

        # Fetch from each source
        youtube_subscribers = None
        youtube_views = None
        lastfm_listeners = None
        lastfm_play_count = None

        if self.youtube and self.youtube.available:
            youtube_metrics = self.youtube.get_artist_metrics(artist_name)
            if youtube_metrics:
                youtube_subscribers = youtube_metrics.subscriber_count
                youtube_views = youtube_metrics.view_count

        if self.lastfm and self.lastfm.available:
            lastfm_metrics = self.lastfm.get_artist_metrics(artist_name)
            if lastfm_metrics:
                lastfm_listeners = lastfm_metrics.listener_count
                lastfm_play_count = lastfm_metrics.play_count

        # Calculate aggregated score
        popularity = self.aggregator.calculate_score(
            artist_name=artist_name,
            youtube_subscribers=youtube_subscribers,
            youtube_views=youtube_views,
            lastfm_listeners=lastfm_listeners,
            lastfm_play_count=lastfm_play_count,
        )

        # Cache result
        self.cache.set(artist_name, popularity)

        return popularity

    def get_artist_popularity_for_market(
        self,
        artist_name: str,
        market: str = "US",
        skip_cache: bool = False,
    ) -> ArtistPopularity:
        """Get popularity data for an artist in a specific market.

        Args:
            artist_name: Artist name to look up
            market: ISO 3166-1 alpha-2 country code
            skip_cache: Skip cache lookup and fetch fresh data

        Returns:
            ArtistPopularity with market-specific data where available
        """
        # Use market-specific cache key to avoid collision with global cache
        cache_key = f"{artist_name}__market_{market}"

        # Check cache first
        if not skip_cache:
            cached = self.cache.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for {artist_name} (market={market})")
                return ArtistPopularity(
                    name=str(cached["name"]),
                    popularity_score=float(cached["popularity_score"]),
                    tier=PopularityTier(str(cached["tier"])),
                    youtube_subscribers=int(cached["youtube_subscribers"])
                    if cached.get("youtube_subscribers") is not None
                    else None,
                    youtube_views=int(cached["youtube_views"])
                    if cached.get("youtube_views") is not None
                    else None,
                    lastfm_listeners=int(cached["lastfm_listeners"])
                    if cached.get("lastfm_listeners") is not None
                    else None,
                    lastfm_play_count=int(cached["lastfm_play_count"])
                    if cached.get("lastfm_play_count") is not None
                    else None,
                    sources_available=list(cached.get("sources_available", [])),
                )

        # Fetch from each source (neither YouTube nor Last.fm are market-specific)
        youtube_subscribers = None
        youtube_views = None
        lastfm_listeners = None
        lastfm_play_count = None

        if self.youtube and self.youtube.available:
            youtube_metrics = self.youtube.get_artist_metrics(artist_name)
            if youtube_metrics:
                youtube_subscribers = youtube_metrics.subscriber_count
                youtube_views = youtube_metrics.view_count

        if self.lastfm and self.lastfm.available:
            lastfm_metrics = self.lastfm.get_artist_metrics(artist_name)
            if lastfm_metrics:
                lastfm_listeners = lastfm_metrics.listener_count
                lastfm_play_count = lastfm_metrics.play_count

        # Calculate aggregated score
        popularity = self.aggregator.calculate_score(
            artist_name=artist_name,
            youtube_subscribers=youtube_subscribers,
            youtube_views=youtube_views,
            lastfm_listeners=lastfm_listeners,
            lastfm_play_count=lastfm_play_count,
        )

        # Cache result with market-specific key
        self.cache.set(cache_key, popularity)

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
