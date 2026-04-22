from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db import models as m
from app.db.base import Base
from app.ingestion.base import RawOdds
from app.ingestion.orchestrator import ingest_all
from app.ingestion.the_odds_api import (
    BASE_URL,
    TheOddsApiError,
    TheOddsApiRealSource,
    _map_event,
    _map_event_odds,
)

# ---------------------------------------------------------------------------
# Sample payloads
# ---------------------------------------------------------------------------


def _sample_event(*, event_id: str = "evt1") -> dict[str, Any]:
    return {
        "id": event_id,
        "sport_key": "soccer_epl",
        "sport_title": "English Premier League",
        "commence_time": "2025-01-15T15:00:00Z",
        "home_team": "Manchester City FC",
        "away_team": "Arsenal FC",
        "bookmakers": [
            {
                "key": "snai",
                "title": "SNAI",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Manchester City FC", "price": 1.85},
                            {"name": "Arsenal FC", "price": 4.2},
                            {"name": "Draw", "price": 3.6},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": 1.9, "point": 2.5},
                            {"name": "Under", "price": 1.95, "point": 2.5},
                        ],
                    },
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": "Manchester City FC", "price": 1.95, "point": -1.0},
                            {"name": "Arsenal FC", "price": 1.9, "point": 1.0},
                        ],
                    },
                ],
            },
            {
                "key": "bet365",
                "title": "Bet365",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Manchester City FC", "price": 1.83},
                            {"name": "Arsenal FC", "price": 4.3},
                            {"name": "Draw", "price": 3.7},
                        ],
                    }
                ],
            },
        ],
    }


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------


def test_map_event_strips_suffix_and_normalises_teams():
    raw = _map_event(_sample_event(), sport_key="soccer_epl")
    assert raw is not None
    assert raw.external_id == "odds-api:evt1"
    assert raw.sport_code == "football"
    assert raw.competition_code == "PL"
    assert raw.competition_name == "Premier League"
    assert raw.home_team == "Manchester City"
    assert raw.away_team == "Arsenal"
    assert raw.status == "scheduled"


def test_map_event_rejects_incomplete_payload():
    assert _map_event({"id": "x"}, sport_key="soccer_epl") is None
    assert (
        _map_event(
            {"id": "x", "home_team": "A", "away_team": "B"},
            sport_key="soccer_epl",
        )
        is None
    )


def test_map_event_odds_covers_1x2_totals_and_spreads():
    captured_at = datetime(2025, 1, 15, 14, 0, tzinfo=UTC)
    odds = _map_event_odds(_sample_event(), external_id="odds-api:evt1", captured_at=captured_at)
    # 1 event × (snai: 3 1X2 + 2 totals + 2 spreads) + (bet365: 3 1X2) = 10
    assert len(odds) == 10

    by_book_market: dict[tuple[str, str], list[RawOdds]] = {}
    for o in odds:
        by_book_market.setdefault((o.bookmaker, o.market), []).append(o)

    snai_1x2 = by_book_market[("snai", "1X2")]
    assert {o.selection for o in snai_1x2} == {"home", "away", "draw"}
    assert all(o.line is None for o in snai_1x2)

    snai_totals = by_book_market[("snai", "OU")]
    assert {(o.selection, o.line) for o in snai_totals} == {("over", 2.5), ("under", 2.5)}

    snai_ah = by_book_market[("snai", "AH")]
    assert {(o.selection, o.line) for o in snai_ah} == {("home", -1.0), ("away", 1.0)}

    # bet365 only exposes 1X2 in the sample.
    assert ("bet365", "1X2") in by_book_market
    assert ("bet365", "OU") not in by_book_market


def test_map_event_promotes_naive_commence_time_to_utc():
    """If The Odds API ever drops the trailing ``Z`` / tz offset, the adapter
    must still return a timezone-aware datetime so matches land in the same
    zone as Football-Data.org and don't duplicate on dedup."""
    evt = _sample_event()
    evt["commence_time"] = "2025-01-15T15:00:00"  # no Z, no offset
    raw = _map_event(evt, sport_key="soccer_epl")
    assert raw is not None
    assert raw.kickoff.tzinfo is not None
    assert raw.kickoff == datetime(2025, 1, 15, 15, 0, tzinfo=UTC)


def test_map_event_odds_skips_bogus_prices():
    evt = _sample_event()
    evt["bookmakers"][0]["markets"][0]["outcomes"][0]["price"] = 1.0
    odds = _map_event_odds(evt, external_id="x", captured_at=datetime.now(tz=UTC))
    # The bogus 1.00 outcome must be dropped; everything else survives.
    assert all(o.price > 1.01 for o in odds)


# ---------------------------------------------------------------------------
# Source end-to-end with mocked httpx
# ---------------------------------------------------------------------------


def _build_source(
    *,
    response_status: int = 200,
    payload: Any | None = None,
    sport_keys: tuple[str, ...] = ("soccer_epl",),
    raw_sink: Callable[..., None] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[TheOddsApiRealSource, list[httpx.Request]]:
    requests: list[httpx.Request] = []
    body = payload if payload is not None else [_sample_event()]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(response_status, json=body, headers=headers or {})

    client = httpx.Client(base_url=BASE_URL, transport=httpx.MockTransport(handler))
    src = TheOddsApiRealSource(
        "test-key",
        sport_keys=sport_keys,
        markets=("h2h", "totals", "spreads"),
        regions=("eu",),
        rate_limit_seconds=0.0,
        client=client,
        raw_sink=raw_sink,
        sleep_fn=lambda _s: None,
    )
    return src, requests


def test_fetch_yields_matches_and_odds_and_captures_quota_header():
    src, requests = _build_source(
        headers={"x-requests-remaining": "473", "x-requests-used": "27"},
    )
    result = src.fetch()
    src.close()
    assert len(result.matches) == 1
    assert len(result.odds) == 10
    assert result.matches[0].competition_code == "PL"
    assert result.meta["requests_remaining"] == 473
    # api key is never forwarded to raw sink / meta.
    assert "apiKey" not in result.meta
    # Outbound call hits the odds endpoint with the correct params.
    assert len(requests) == 1
    url = requests[0].url
    assert "/sports/soccer_epl/odds" in url.path
    assert url.params.get("regions") == "eu"
    assert "h2h" in url.params.get("markets", "")
    assert url.params.get("apiKey") == "test-key"


def test_raw_sink_receives_calls_without_api_key():
    calls: list[tuple[str, dict[str, Any], int]] = []

    def sink(endpoint: str, params: dict[str, Any], status: int, body: Any) -> None:
        calls.append((endpoint, dict(params), status))

    src, _ = _build_source(raw_sink=sink)
    src.fetch()
    src.close()
    assert len(calls) == 1
    endpoint, params, status = calls[0]
    assert "/sports/soccer_epl/odds" in endpoint
    assert "apiKey" not in params  # sensitive fields are stripped
    assert status == 200


def test_fetch_continues_after_per_sport_failure():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if "soccer_epl" in str(request.url):
            return httpx.Response(403, json={"message": "Forbidden"})
        return httpx.Response(200, json=[_sample_event(event_id="evt2")])

    client = httpx.Client(base_url=BASE_URL, transport=httpx.MockTransport(handler))
    src = TheOddsApiRealSource(
        "test-key",
        sport_keys=("soccer_epl", "soccer_italy_serie_a"),
        rate_limit_seconds=0.0,
        client=client,
        sleep_fn=lambda _s: None,
    )
    result = src.fetch()
    src.close()
    assert len(result.matches) == 1  # only soccer_italy_serie_a succeeded
    assert result.matches[0].external_id == "odds-api:evt2"
    assert "failures" in result.meta
    assert result.meta["failures"][0]["sport_key"] == "soccer_epl"


def test_401_is_not_retried():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"message": "Unauthorized"})

    client = httpx.Client(base_url=BASE_URL, transport=httpx.MockTransport(handler))
    src = TheOddsApiRealSource(
        "bad",
        sport_keys=("soccer_epl",),
        rate_limit_seconds=0.0,
        client=client,
        sleep_fn=lambda _s: None,
    )
    result = src.fetch()
    src.close()
    assert result.matches == []
    assert calls["n"] == 1
    assert "failures" in result.meta


def test_rate_limited_response_raises_retryable_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"message": "Too Many Requests"})

    client = httpx.Client(base_url=BASE_URL, transport=httpx.MockTransport(handler))
    src = TheOddsApiRealSource(
        "k",
        sport_keys=("soccer_epl",),
        rate_limit_seconds=0.0,
        client=client,
        sleep_fn=lambda _s: None,
    )
    # Retries exhaust -> surfaces as a FeatureError that the orchestrator wraps.
    with pytest.raises(TheOddsApiError):
        src._fetch_sport("soccer_epl")
    src.close()


def test_empty_response_body_does_not_produce_null_payload():
    """An empty HTTP body must surface as ``{}`` (not ``None``) so the raw_sink
    can insert into the NOT NULL ``ingestion_payloads.payload`` column without
    corrupting the session."""
    recorded: list[tuple[str, dict[str, Any], int, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"")

    def sink(endpoint: str, params: dict[str, Any], status: int, body: Any) -> None:
        recorded.append((endpoint, params, status, body))

    client = httpx.Client(base_url=BASE_URL, transport=httpx.MockTransport(handler))
    src = TheOddsApiRealSource(
        "k",
        sport_keys=("soccer_epl",),
        rate_limit_seconds=0.0,
        client=client,
        raw_sink=sink,
        sleep_fn=lambda _s: None,
    )
    result = src.fetch()
    src.close()

    assert result.matches == []
    assert recorded, "raw_sink should have been called"
    _, _, status, body = recorded[0]
    assert status == 200
    assert body == {}
    assert body is not None


def test_4xx_null_message_does_not_crash_fetch():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"message": None})

    client = httpx.Client(base_url=BASE_URL, transport=httpx.MockTransport(handler))
    src = TheOddsApiRealSource(
        "k",
        sport_keys=("soccer_epl", "soccer_italy_serie_a"),
        rate_limit_seconds=0.0,
        client=client,
        sleep_fn=lambda _s: None,
    )
    result = src.fetch()
    src.close()
    assert result.matches == []
    assert len(result.meta["failures"]) == 2


# ---------------------------------------------------------------------------
# Orchestrator persistence + closing-line helper
# ---------------------------------------------------------------------------


@pytest.fixture()
def sqlite_db() -> Session:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    session = TestSession()
    try:
        yield session
    finally:
        session.close()


def test_orchestrator_persists_matches_odds_and_marks_closing(sqlite_db: Session):
    # kickoff is in the past so the closing-line helper fires; odds were
    # captured earlier (before kickoff) as they would have been in reality.
    kickoff = datetime.now(tz=UTC) - timedelta(hours=1)
    captured_at = kickoff - timedelta(minutes=5)
    evt = _sample_event()
    evt["commence_time"] = kickoff.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[evt])

    client = httpx.Client(base_url=BASE_URL, transport=httpx.MockTransport(handler))
    src = TheOddsApiRealSource(
        "k",
        sport_keys=("soccer_epl",),
        rate_limit_seconds=0.0,
        client=client,
        now_fn=lambda: captured_at,
        sleep_fn=lambda _s: None,
    )

    counts = ingest_all(sqlite_db, sources=[src])
    src.close()

    assert counts["matches"] == 1
    assert counts["odds"] == 10
    matches = sqlite_db.query(m.Match).all()
    assert len(matches) == 1
    assert matches[0].home_team.name == "Manchester City"
    assert matches[0].away_team.name == "Arsenal"

    snaps = sqlite_db.query(m.OddsSnapshot).all()
    assert len(snaps) == 10
    closing = [s for s in snaps if s.is_closing]
    # One closing row per (book, market, selection, line) tuple for a single fetch,
    # i.e. exactly the 10 rows we just inserted.
    assert len(closing) == 10

    runs = sqlite_db.query(m.IngestionRun).all()
    assert len(runs) == 1
    assert runs[0].source == "the-odds-api"
    assert runs[0].ok is True


# ---------------------------------------------------------------------------
# Status monotonicity across overlapping sources
# ---------------------------------------------------------------------------


class _StubSource:
    """Minimal BaseSource-compatible stub that emits a single prepared result."""

    name = "stub"
    sports = ("football",)

    def __init__(self, name: str, result: Any) -> None:
        self.name = name
        self._result = result

    def is_enabled(self) -> bool:
        return True

    def fetch(self, *, season: str | None = None) -> Any:
        return self._result


def test_status_is_not_regressed_when_the_odds_api_runs_after_finished_source(
    sqlite_db: Session,
):
    """Regression test: The Odds API hardcodes ``status="scheduled"`` per-event.
    If the orchestrator persisted that naively, it would demote a match that
    Football-Data.org has already marked as ``finished`` (or ``live``) back to
    ``scheduled`` whenever both sources share a competition. Ingestion must
    only promote status forward along the lifecycle."""
    from app.ingestion.base import IngestionResult, RawMatch

    kickoff = datetime(2025, 1, 15, 15, 0, tzinfo=UTC)
    finished = RawMatch(
        external_id="fd:1",
        sport_code="football",
        competition_code="PL",
        competition_name="Premier League",
        home_team="Manchester City",
        away_team="Arsenal",
        kickoff=kickoff,
        status="finished",
        home_score=2,
        away_score=1,
    )
    scheduled = RawMatch(
        external_id="odds-api:evt1",
        sport_code="football",
        competition_code="PL",
        competition_name="Premier League",
        home_team="Manchester City",
        away_team="Arsenal",
        kickoff=kickoff,
        status="scheduled",
    )
    src1 = _StubSource("football-data.org", IngestionResult(matches=[finished]))
    src2 = _StubSource("the-odds-api", IngestionResult(matches=[scheduled]))

    ingest_all(sqlite_db, sources=[src1, src2])

    rows = sqlite_db.query(m.Match).all()
    assert len(rows) == 1, "sources must dedupe, not duplicate"
    assert rows[0].status == "finished", (
        "a 'scheduled' update from The Odds API must not demote an already "
        "'finished' match"
    )
    assert rows[0].home_score == 2
    assert rows[0].away_score == 1
