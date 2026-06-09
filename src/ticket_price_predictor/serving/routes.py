"""HTTP route handlers for the serving layer.

Handler bodies are filled in by AC2/AC3/AC4; this module wires the routes,
dependencies, and response types so the app factory in `app.py` can mount the
router immediately.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from ticket_price_predictor.ml.inference.predictor import (
    ZONE_REPRESENTATIVE_SECTIONS,
    PricePredictor,
    UnknownEventError,
)
from ticket_price_predictor.serving.dependencies import (
    get_event_repo,
    get_known_events,
    get_predictor,
)
from ticket_price_predictor.serving.schemas import (
    EventDetail,
    EventSummary,
    PredictRequest,
    PredictResponse,
)
from ticket_price_predictor.storage.repository import EventFilters, EventRepository

router = APIRouter()

PredictorDep = Annotated[PricePredictor, Depends(get_predictor)]
EventRepoDep = Annotated[EventRepository, Depends(get_event_repo)]
KnownEventsDep = Annotated[set[str], Depends(get_known_events)]


@router.get("/events/search")
def search_events(
    repo: EventRepoDep,
    known_events: KnownEventsDep,
    q: str | None = None,
    city: str | None = None,
    from_: Annotated[datetime | None, Query(alias="from")] = None,
    to: datetime | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[EventSummary]:
    """Search known events, defaulting to future-dated only.

    `q` is a case-insensitive substring match across artist, venue, and city
    so a user-typed "las vegas" finds events without using the dedicated `city`
    param (which remains as a strict equality escape hatch). When `from` is
    omitted it defaults to `now(UTC)` so past events are hidden.
    """
    if from_ is None:
        from_ = datetime.now(UTC)
    elif from_.tzinfo is None:
        from_ = from_.replace(tzinfo=UTC)
    if to is not None and to.tzinfo is None:
        to = to.replace(tzinfo=UTC)

    filters = EventFilters(city=city, start_date=from_, end_date=to)
    events = repo.get_events(filters)
    events = [e for e in events if e.event_id in known_events]

    if q:
        needle = q.casefold()
        events = [
            e for e in events if needle in f"{e.artist_or_team} {e.venue_name} {e.city}".casefold()
        ]

    events.sort(key=lambda e: e.event_datetime)
    events = events[:limit]

    return [EventSummary.model_validate(e.model_dump()) for e in events]


@router.get("/events/{event_id}")
def get_event(
    event_id: str,
    repo: EventRepoDep,
    known_events: KnownEventsDep,
) -> EventDetail:
    """Return full detail for a known event, or 404 if absent.

    `known_events` should be a subset of the parquet, but we re-check the
    repository as defense in depth so a stale model artifact never returns a
    row that no longer exists on disk.
    """
    if event_id not in known_events:
        raise HTTPException(status_code=404, detail="event not found")
    event = repo.get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="event not found")
    return EventDetail.model_validate(event.model_dump())


@router.post("/predict")
def predict(
    req: PredictRequest,
    predictor: PredictorDep,
    repo: EventRepoDep,
    known_events: KnownEventsDep,
) -> PredictResponse:
    """Return a price prediction for a zone in a known event.

    Section vs. zone precedence (AC5):
      * When `section` is set it is passed straight through to the predictor;
        the predictor's `SeatZoneMapper` re-derives the zone from the section.
      * When only `seat_zone` is set the route maps it to a representative
        section via `ZONE_REPRESENTATIVE_SECTIONS` (the same mapping used by
        `PricePredictor.predict_for_zones`).
      * When BOTH are set, `section` wins — the `seat_zone` hint is ignored.

    Returns 404 when the event_id is unknown (either to `known_events` or to
    the repository), and 409 when the event has already occurred (strict `<`
    against `now(UTC)` — the search route's inclusive `>=` and this route's
    strict `<` form a clean partition at the instant `now()`).
    """
    if req.event_id not in known_events:
        raise HTTPException(status_code=404, detail="event not found")
    event = repo.get_event(req.event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="event not found")

    event_datetime = event.event_datetime
    if event_datetime.tzinfo is None:
        event_datetime = event_datetime.replace(tzinfo=UTC)

    ref_time = req.as_of_date or datetime.now(UTC)
    if ref_time.tzinfo is None:
        ref_time = ref_time.replace(tzinfo=UTC)
    if event_datetime < ref_time:
        raise HTTPException(status_code=409, detail="event has already occurred")

    days_to_event = max(0, (event_datetime - ref_time).days)

    if req.section:
        section = req.section
    else:
        # _zone_or_section validator guarantees seat_zone is set when section is empty.
        assert req.seat_zone is not None
        section = ZONE_REPRESENTATIVE_SECTIONS[req.seat_zone]

    try:
        pred = predictor.predict(
            event_id=req.event_id,
            artist_or_team=event.artist_or_team,
            venue_name=event.venue_name,
            city=event.city,
            event_datetime=event_datetime,
            section=section,
            row=req.row,
            days_to_event=days_to_event,
            event_type=event.event_type.value,
            quantity=req.quantity,
        )
    except UnknownEventError as exc:
        raise HTTPException(status_code=404, detail="event not found") from exc

    return PredictResponse.model_validate(pred.model_dump())
