"""Route-level tests for the serving layer (AC13).

Wires test doubles via FastAPI's idiomatic ``app.dependency_overrides`` rather
than any production-code branching. The stub predictor records every ``predict``
call so tests can assert on the exact kwargs the route built (notably the
section-vs-zone precedence and the server-computed ``days_to_event``).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ticket_price_predictor.ml.schemas import PricePrediction
from ticket_price_predictor.normalization.seat_zones import SeatZoneMapper
from ticket_price_predictor.schemas import EventMetadata, EventType
from ticket_price_predictor.serving.app import create_app
from ticket_price_predictor.serving.dependencies import (
    get_event_repo,
    get_known_events,
    get_predictor,
)
from ticket_price_predictor.storage.repository import EventRepository

FUTURE_KNOWN_ID = "evt_future_known"
PAST_KNOWN_ID = "evt_past_known"
FUTURE_UNKNOWN_ID = "evt_future_unknown"

FUTURE_DELTA = timedelta(days=60)
PAST_DELTA = timedelta(days=30)
UNKNOWN_DELTA = timedelta(days=90)


class _StubPredictor:
    """Minimal predictor stub that records calls and returns a fixed prediction."""

    def __init__(self, known_events: set[str]) -> None:
        self.known_events: set[str] = known_events
        self.last_call_kwargs: dict[str, Any] = {}
        self._mapper = SeatZoneMapper()

    def predict(self, **kwargs: Any) -> PricePrediction:
        self.last_call_kwargs = kwargs
        return PricePrediction(
            event_id=kwargs["event_id"],
            seat_zone=self._mapper.normalize_zone_name(kwargs["section"]).value,
            target_days_to_event=kwargs["days_to_event"],
            predicted_price=100.0,
            price_lower_bound=80.0,
            price_upper_bound=120.0,
            confidence_score=0.9,
            predicted_direction="STABLE",
            direction_probability=0.6,
            model_version="test",
            prediction_timestamp=datetime.now(UTC),
        )


def _build_event_repo_with_events(
    tmp_path: Path,
    *,
    future_known_dt: datetime,
    past_known_dt: datetime,
    future_unknown_dt: datetime,
) -> EventRepository:
    """Seed a real ``EventRepository`` with the three fixture events."""
    repo = EventRepository(tmp_path)
    repo.save_events(
        [
            EventMetadata(
                event_id=FUTURE_KNOWN_ID,
                event_type=EventType.CONCERT,
                event_datetime=future_known_dt,
                artist_or_team="Bruno Mars",
                venue_id="ven_001",
                venue_name="T-Mobile Arena",
                city="Las Vegas",
                country="US",
            ),
            EventMetadata(
                event_id=PAST_KNOWN_ID,
                event_type=EventType.CONCERT,
                event_datetime=past_known_dt,
                artist_or_team="BTS",
                venue_id="ven_002",
                venue_name="SoFi Stadium",
                city="Los Angeles",
                country="US",
            ),
            EventMetadata(
                event_id=FUTURE_UNKNOWN_ID,
                event_type=EventType.CONCERT,
                event_datetime=future_unknown_dt,
                artist_or_team="Taylor Swift",
                venue_id="ven_003",
                venue_name="MetLife Stadium",
                city="East Rutherford",
                country="US",
            ),
        ]
    )
    return repo


@pytest.fixture
def fixture_now() -> datetime:
    """Capture ``now(UTC)`` once per test for stable date arithmetic."""
    return datetime.now(UTC)


@pytest.fixture
def stub_predictor() -> _StubPredictor:
    return _StubPredictor(known_events={FUTURE_KNOWN_ID, PAST_KNOWN_ID})


@pytest.fixture
def client(
    tmp_path: Path, fixture_now: datetime, stub_predictor: _StubPredictor
) -> Iterator[TestClient]:
    repo = _build_event_repo_with_events(
        tmp_path,
        future_known_dt=fixture_now + FUTURE_DELTA,
        past_known_dt=fixture_now - PAST_DELTA,
        future_unknown_dt=fixture_now + UNKNOWN_DELTA,
    )
    app = create_app(static_dir=tmp_path)
    app.dependency_overrides[get_predictor] = lambda: stub_predictor
    app.dependency_overrides[get_event_repo] = lambda: repo
    app.dependency_overrides[get_known_events] = lambda: stub_predictor.known_events
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_search_returns_known_event_and_filters_out_non_known(client: TestClient) -> None:
    response = client.get("/api/events/search", params={"q": "bruno"})
    assert response.status_code == 200
    body = response.json()
    ids = [e["event_id"] for e in body]
    assert FUTURE_KNOWN_ID in ids
    assert FUTURE_UNKNOWN_ID not in ids


def test_search_with_no_matches_returns_empty_list(client: TestClient) -> None:
    response = client.get("/api/events/search", params={"q": "nonexistent_artist_xyz"})
    assert response.status_code == 200
    assert response.json() == []


def test_search_defaults_to_future_only(client: TestClient) -> None:
    response = client.get("/api/events/search")
    assert response.status_code == 200
    ids = [e["event_id"] for e in response.json()]
    assert PAST_KNOWN_ID not in ids
    assert FUTURE_KNOWN_ID in ids


def test_search_includes_past_event_when_from_is_set(client: TestClient) -> None:
    response = client.get("/api/events/search", params={"from": "2000-01-01"})
    assert response.status_code == 200
    ids = [e["event_id"] for e in response.json()]
    assert PAST_KNOWN_ID in ids


def test_search_q_lowercase_matches_city_case_insensitively(client: TestClient) -> None:
    response = client.get("/api/events/search", params={"q": "las vegas"})
    assert response.status_code == 200
    ids = [e["event_id"] for e in response.json()]
    assert FUTURE_KNOWN_ID in ids


def test_search_limit_out_of_range_returns_422(client: TestClient) -> None:
    response = client.get("/api/events/search", params={"limit": 999})
    assert response.status_code == 422


def test_event_detail_returns_200_for_known_event(client: TestClient) -> None:
    response = client.get(f"/api/events/{FUTURE_KNOWN_ID}")
    assert response.status_code == 200
    body = response.json()
    assert body["event_id"] == FUTURE_KNOWN_ID
    assert body["artist_or_team"] == "Bruno Mars"


def test_event_detail_returns_404_for_unknown_event(client: TestClient) -> None:
    response = client.get("/api/events/does_not_exist")
    assert response.status_code == 404
    assert response.json() == {"detail": "event not found"}


def test_predict_returns_200_with_seat_zone(
    client: TestClient, stub_predictor: _StubPredictor
) -> None:
    response = client.post(
        "/api/predict",
        json={"event_id": FUTURE_KNOWN_ID, "seat_zone": "lower_tier"},
    )
    assert response.status_code == 200
    # Zone-only path: route maps to representative section "Lower Level 100".
    assert stub_predictor.last_call_kwargs["section"] == "Lower Level 100"


def test_predict_returns_200_with_free_text_section(
    client: TestClient, stub_predictor: _StubPredictor
) -> None:
    response = client.post(
        "/api/predict",
        json={"event_id": FUTURE_KNOWN_ID, "section": "Lower Level 100"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["seat_zone"] == "lower_tier"
    assert stub_predictor.last_call_kwargs["section"] == "Lower Level 100"


def test_predict_section_wins_when_both_set(
    client: TestClient, stub_predictor: _StubPredictor
) -> None:
    response = client.post(
        "/api/predict",
        json={
            "event_id": FUTURE_KNOWN_ID,
            "seat_zone": "balcony",
            "section": "Floor VIP",
        },
    )
    assert response.status_code == 200
    # Section wins: predictor sees the verbatim section, not the zone's representative.
    assert stub_predictor.last_call_kwargs["section"] == "Floor VIP"


def test_predict_returns_422_when_neither_zone_nor_section(client: TestClient) -> None:
    response = client.post("/api/predict", json={"event_id": FUTURE_KNOWN_ID})
    assert response.status_code == 422


def test_predict_returns_404_for_unknown_event_id(
    client: TestClient, stub_predictor: _StubPredictor
) -> None:
    stub_predictor.last_call_kwargs = {}
    response = client.post(
        "/api/predict",
        json={"event_id": "does_not_exist", "seat_zone": "lower_tier"},
    )
    assert response.status_code == 404
    # Gating short-circuits before predictor is invoked.
    assert stub_predictor.last_call_kwargs == {}


def test_predict_returns_409_for_past_event(client: TestClient) -> None:
    response = client.post(
        "/api/predict",
        json={"event_id": PAST_KNOWN_ID, "seat_zone": "lower_tier"},
    )
    assert response.status_code == 409
    assert response.json() == {"detail": "event has already occurred"}


def test_predict_days_to_event_computed_from_event_datetime(
    client: TestClient, stub_predictor: _StubPredictor, fixture_now: datetime
) -> None:
    response = client.post(
        "/api/predict",
        json={"event_id": FUTURE_KNOWN_ID, "seat_zone": "lower_tier"},
    )
    assert response.status_code == 200
    expected = ((fixture_now + FUTURE_DELTA) - datetime.now(UTC)).days
    actual = stub_predictor.last_call_kwargs["days_to_event"]
    # Allow ±1 day tolerance to absorb a midnight clock-tick race.
    assert abs(actual - expected) <= 1


def test_predict_response_includes_prediction_timestamp(client: TestClient) -> None:
    response = client.post(
        "/api/predict",
        json={"event_id": FUTURE_KNOWN_ID, "seat_zone": "lower_tier"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "prediction_timestamp" in body
    assert isinstance(body["prediction_timestamp"], str)
