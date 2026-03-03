"""External popularity data aggregation from YouTube Music and Last.fm."""

from ticket_price_predictor.popularity.aggregator import ArtistPopularity, PopularityAggregator
from ticket_price_predictor.popularity.service import PopularityService

__all__ = [
    "ArtistPopularity",
    "PopularityAggregator",
    "PopularityService",
]
