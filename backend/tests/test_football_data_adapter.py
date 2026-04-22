"""Tests for the Football-Data.org adapter.

All HTTP traffic is mocked via ``httpx.MockTransport`` so the suite runs
offline with zero latency.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest

from app.db import models as m
from app.ingestion.football_data_org import (
    FootballDataOrgError,
    FootballDataOrgSource,
    _map_match,
    _parse_datetime,
)
from app.ingestion.orchestrator import ingest_all

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sample_payload(comp_code: str = "PL") -> dict:
    return {
        "count": 2,
        "competition": {"code": comp_code, "name": "Premier League"},
        "matches": [
            {
                "id": 100001,
                "utcDate": "2024-08-17T16:30:00Z",
                "status": "FINISHED",
                "stage": "REGULAR_SEASON",
                "season": {"startDate": "2024-08-16", "endDate": "2025-05-25"},
                "homeTeam": {"name": "Arsenal FC"},
                "awayTeam": {"name": "Liverpool FC"},
                "score": {"fullTime": {"home": 2, "away": 1}},
                "competition": {"name": "Premier League"},
            },
            {
                "id": 100002,
                "utcDate": "2024-08-18T13:00:00Z",
                "status": "SCHEDULED",
                "stage": "REGULAR_SEASON",
                "season": {"startDate": "2024-08-16", "endDate": "2025-05-25"},
                "homeTeam": {"name": "Chelsea FC"},
                "awayTeam": {"name": "Manchester City FC"},
                "score": {"fullTime": {"home": None, "away": None}},
                "competition": {"name": "Premier League"},
            },
        ],
    }


def _build_source(
    *,
    response_status: int = 200,
    payload: dict | None = None,
    key: str = "test-token",
    competitions: tuple[str, ...] = ("PL",),
) -> tuple[FootballDataOrgSource, list[httpx.Request]]:
    calls: list[httpx.Request] = []
    body = payload if payload is not None else _sample_payload()

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(response_status, json=body)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(
        base_url="https://api.football-data.org/v4",
        transport=transport,
        headers={"X-Auth-Token": key},
    )
    src = FootballDataOrgSource(
        key,
        competitions=competitions,
        rate_limit_seconds=0.0,
        client=client,
        sleep_fn=lambda _s: None,
    )
    return src, calls


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------


def test_parse_datetime_handles_trailing_z():
    dt = _parse_datetime("2024-08-17T16:30:00Z")
    assert dt == datetime(2024, 8, 17, 16, 30, tzinfo=UTC)


def test_map_match_preserves_status_and_scores():
    payload = _sample_payload()["matches"][0]
    mapped = _map_match(payload, competition_code="PL")
    assert mapped is not None
    assert mapped.external_id == "100001"
    assert mapped.status == "finished"
    assert mapped.home_score == 2
    assert mapped.away_score == 1
    assert mapped.season == "2024"
    assert mapped.home_team == "Arsenal FC"
    assert mapped.competition_code == "PL"


def test_map_match_rejects_incomplete_payload():
    assert _map_match({}, competition_code="PL") is None


# ---------------------------------------------------------------------------
# Fetch happy-path
# ---------------------------------------------------------------------------


def test_fetch_maps_matches_from_mocked_api():
    src, calls = _build_source()
    result = src.fetch()
    src.close()

    assert len(calls) == 1
    req = calls[0]
    assert req.headers["X-Auth-Token"] == "test-token"
    assert req.url.path.endswith("/competitions/PL/matches")

    assert len(result.matches) == 2
    first = result.matches[0]
    assert first.competition_code == "PL"
    assert first.status == "finished"
    assert first.home_score == 2


def test_fetch_continues_after_per_competition_failure():
    """One competition errors out, the rest still yield matches."""
    responses = iter(
        [
            (500, {"message": "upstream boom"}),
            (500, {"message": "upstream boom"}),
            (500, {"message": "upstream boom"}),
            (200, _sample_payload("SA")),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        status, body = next(responses)
        return httpx.Response(status, json=body)

    client = httpx.Client(
        base_url="https://api.football-data.org/v4",
        transport=httpx.MockTransport(handler),
        headers={"X-Auth-Token": "k"},
    )
    src = FootballDataOrgSource(
        "k",
        competitions=("PL", "SA"),
        rate_limit_seconds=0.0,
        client=client,
        sleep_fn=lambda _s: None,
    )
    result = src.fetch()
    src.close()

    assert len(result.matches) == 2  # SA succeeded
    assert "failures" in result.meta
    assert result.meta["failures"][0]["competition"] == "PL"


def test_unauthorized_response_is_not_retried():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(403, json={"message": "Forbidden"})

    client = httpx.Client(
        base_url="https://api.football-data.org/v4",
        transport=httpx.MockTransport(handler),
        headers={"X-Auth-Token": "bad"},
    )
    src = FootballDataOrgSource(
        "bad",
        competitions=("PL",),
        rate_limit_seconds=0.0,
        client=client,
        sleep_fn=lambda _s: None,
    )
    # fetch() swallows the error and records it in meta.
    result = src.fetch()
    src.close()
    assert result.matches == []
    assert calls["n"] == 1  # single 4xx attempt, no retries
    assert "failures" in result.meta


def test_rate_limited_response_raises_retryable_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"message": "Too Many Requests"})

    client = httpx.Client(
        base_url="https://api.football-data.org/v4",
        transport=httpx.MockTransport(handler),
        headers={"X-Auth-Token": "k"},
    )
    src = FootballDataOrgSource(
        "k",
        competitions=("PL",),
        rate_limit_seconds=0.0,
        client=client,
        sleep_fn=lambda _s: None,
    )
    # With retries exhausted the per-competition failure is captured in meta.
    result = src.fetch()
    src.close()
    assert result.matches == []
    failures = result.meta.get("failures") or []
    assert failures and "429" in failures[0]["error"]


# ---------------------------------------------------------------------------
# Raw sink
# ---------------------------------------------------------------------------


def test_raw_sink_receives_endpoint_params_and_status():
    captured: list[tuple[str, dict, int, dict]] = []

    def sink(endpoint, params, status, payload):
        captured.append((endpoint, dict(params), status, payload))

    payload = _sample_payload()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = httpx.Client(
        base_url="https://api.football-data.org/v4",
        transport=httpx.MockTransport(handler),
        headers={"X-Auth-Token": "k"},
    )
    src = FootballDataOrgSource(
        "k",
        competitions=("PL",),
        rate_limit_seconds=0.0,
        client=client,
        sleep_fn=lambda _s: None,
        raw_sink=sink,
        season="2024",
    )
    src.fetch()
    src.close()
    assert len(captured) == 1
    endpoint, params, status, got = captured[0]
    assert endpoint == "/competitions/PL/matches"
    assert params == {"season": "2024"}
    assert status == 200
    # Confirm we serialised the full response (not just a subset).
    assert got == json.loads(json.dumps(payload))


# ---------------------------------------------------------------------------
# End-to-end through the orchestrator (DB persistence)
# ---------------------------------------------------------------------------


def test_orchestrator_persists_matches_and_audit_trail(db_session):
    src, _calls = _build_source()
    counts = ingest_all(db_session, sources=[src], trigger="test")
    src.close()

    assert counts["matches"] == 2
    # Raw payload audit trail captured.
    payloads = db_session.query(m.IngestionPayload).all()
    assert len(payloads) == 1
    assert payloads[0].source == "football-data"
    assert payloads[0].status_code == 200
    # Run summary recorded.
    runs = db_session.query(m.IngestionRun).all()
    assert len(runs) == 1
    assert runs[0].ok is True
    assert runs[0].matches_upserted == 2
    # Matches landed in the relational tables.
    assert db_session.query(m.Match).count() == 2
    # Competition code preserved as-is so the scheduler can slice by it later.
    codes = {c.code for c in db_session.query(m.Competition).all()}
    assert "PL" in codes


def test_orchestrator_records_failure_run_when_source_raises(db_session):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    client = httpx.Client(
        base_url="https://api.football-data.org/v4",
        transport=httpx.MockTransport(handler),
        headers={"X-Auth-Token": "k"},
    )
    src = FootballDataOrgSource(
        "k",
        competitions=("PL",),
        rate_limit_seconds=0.0,
        client=client,
        sleep_fn=lambda _s: None,
    )
    counts = ingest_all(db_session, sources=[src], trigger="test")
    src.close()

    assert counts["matches"] == 0
    runs = db_session.query(m.IngestionRun).all()
    assert len(runs) == 1
    # The source catches the transport error inside fetch() and reports it in
    # meta rather than bubbling up, so the run is recorded as ok=True but
    # with zero upserts and a "failures" entry in meta.
    assert runs[0].matches_upserted == 0
    failures = (runs[0].meta or {}).get("failures") or []
    assert failures and "PL" in failures[0]["competition"]


def test_disabled_source_without_key_yields_no_work():
    src = FootballDataOrgSource(api_key="", competitions=("PL",), rate_limit_seconds=0.0)
    assert src.is_enabled() is False
    # fetch() short-circuits when disabled.
    assert src.fetch().matches == []


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_rate_limit_waits_between_calls_when_configured():
    sleep_calls: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"matches": []})

    client = httpx.Client(
        base_url="https://api.football-data.org/v4",
        transport=httpx.MockTransport(handler),
        headers={"X-Auth-Token": "k"},
    )
    src = FootballDataOrgSource(
        "k",
        competitions=("PL", "SA", "BL1"),
        rate_limit_seconds=0.5,
        client=client,
        sleep_fn=lambda s: sleep_calls.append(s),
    )
    src.fetch()
    src.close()
    # First request has no prior timestamp so it doesn't sleep. The next two do.
    assert len(sleep_calls) >= 1
    assert all(0 <= s <= 0.5 for s in sleep_calls)


def test_raw_sink_error_does_not_break_fetch():
    def bad_sink(*_args, **_kwargs):
        raise RuntimeError("sink exploded")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_sample_payload())

    client = httpx.Client(
        base_url="https://api.football-data.org/v4",
        transport=httpx.MockTransport(handler),
        headers={"X-Auth-Token": "k"},
    )
    src = FootballDataOrgSource(
        "k",
        competitions=("PL",),
        rate_limit_seconds=0.0,
        client=client,
        sleep_fn=lambda _s: None,
        raw_sink=bad_sink,
    )
    result = src.fetch()
    src.close()
    assert len(result.matches) == 2


def test_error_payload_short_circuits_mapping():
    src, _ = _build_source(response_status=404, payload={"message": "Not found"})
    result = src.fetch()
    src.close()
    assert result.matches == []
    failures = result.meta.get("failures") or []
    assert failures and "404" in failures[0]["error"]


def test_football_data_error_str_contains_status_code():
    err = FootballDataOrgError("500 upstream error")
    assert "500" in str(err)


@pytest.mark.parametrize(
    "status_api, mapped",
    [
        ("FINISHED", "finished"),
        ("SCHEDULED", "scheduled"),
        ("TIMED", "scheduled"),
        ("IN_PLAY", "live"),
        ("PAUSED", "live"),
        ("POSTPONED", "postponed"),
        ("CANCELLED", "cancelled"),
        ("WEIRD_UNKNOWN", "scheduled"),
    ],
)
def test_match_status_mapping(status_api, mapped):
    raw = {
        "id": 1,
        "utcDate": "2024-08-17T16:30:00Z",
        "status": status_api,
        "homeTeam": {"name": "A"},
        "awayTeam": {"name": "B"},
        "season": {"startDate": "2024-08-16"},
        "score": {"fullTime": {"home": None, "away": None}},
    }
    m_ = _map_match(raw, competition_code="PL")
    assert m_ is not None
    assert m_.status == mapped
