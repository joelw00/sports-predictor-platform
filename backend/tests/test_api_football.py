"""Unit tests for the api-sports.io (API-Football) adapter."""

from __future__ import annotations

import httpx
import pytest

from app.ingestion.api_football import ApiFootballSource


def _stub_client(*, handler) -> httpx.Client:
    transport = httpx.MockTransport(handler)
    return httpx.Client(
        base_url="https://v3.football.api-sports.io",
        transport=transport,
        headers={"x-apisports-key": "test"},
    )


def test_disabled_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    assert ApiFootballSource(api_key=None).is_enabled() is False


def test_fetch_parses_fixtures_into_raw_matches() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/fixtures":
            return httpx.Response(
                200,
                json={
                    "response": [
                        {
                            "fixture": {
                                "id": 101,
                                "date": "2024-08-17T14:00:00+00:00",
                                "status": {"short": "FT"},
                            },
                            "teams": {
                                "home": {"name": "Arsenal"},
                                "away": {"name": "Chelsea"},
                            },
                            "goals": {"home": 2, "away": 1},
                        },
                        {
                            "fixture": {
                                "id": 102,
                                "date": "2024-08-18T13:30:00+00:00",
                                "status": {"short": "NS"},
                            },
                            "teams": {
                                "home": {"name": "Liverpool"},
                                "away": {"name": "Man City"},
                            },
                            "goals": {"home": None, "away": None},
                        },
                    ],
                    "errors": [],
                },
            )
        return httpx.Response(404)

    client = _stub_client(handler=handler)
    src = ApiFootballSource(
        api_key="test",
        leagues={39: ("eng-premier-league", "Premier League")},
        rate_limit_seconds=0.0,
        client=client,
    )
    result = src.fetch(season="2024")
    assert len(result.matches) == 2
    finished = next(m for m in result.matches if m.home_team == "Arsenal")
    scheduled = next(m for m in result.matches if m.home_team == "Liverpool")
    assert finished.status == "finished"
    assert finished.home_score == 2 and finished.away_score == 1
    assert scheduled.status == "scheduled"
    assert scheduled.home_score is None


def test_fetch_fixture_statistics_maps_corner_and_card_rows() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/fixtures/statistics":
            return httpx.Response(
                200,
                json={
                    "response": [
                        {
                            "team": {"name": "Arsenal"},
                            "statistics": [
                                {"type": "Total Shots", "value": 14},
                                {"type": "Shots on Goal", "value": 6},
                                {"type": "Corner Kicks", "value": 7},
                                {"type": "Yellow Cards", "value": 2},
                                {"type": "Red Cards", "value": 0},
                                {"type": "Fouls", "value": 11},
                                {"type": "Ball Possession", "value": "58%"},
                            ],
                        },
                        {
                            "team": {"name": "Chelsea"},
                            "statistics": [
                                {"type": "Total Shots", "value": 9},
                                {"type": "Corner Kicks", "value": 4},
                                {"type": "Yellow Cards", "value": 3},
                                {"type": "Red Cards", "value": 1},
                            ],
                        },
                    ],
                    "errors": [],
                },
            )
        return httpx.Response(404)

    client = _stub_client(handler=handler)
    src = ApiFootballSource(
        api_key="test",
        leagues={},
        rate_limit_seconds=0.0,
        client=client,
    )
    rows = src.fetch_fixture_statistics(101, home_external_id="match-1")
    assert len(rows) == 2
    home = next(s for s in rows if s.team_name == "Arsenal")
    away = next(s for s in rows if s.team_name == "Chelsea")
    assert home.corners == 7 and home.yellow_cards == 2 and home.red_cards == 0
    assert home.possession == 58.0
    assert away.corners == 4 and away.yellow_cards == 3 and away.red_cards == 1


def test_api_errors_are_raised() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"response": [], "errors": {"token": "invalid key"}}
        )

    client = _stub_client(handler=handler)
    src = ApiFootballSource(
        api_key="test",
        leagues={},
        rate_limit_seconds=0.0,
        client=client,
    )
    with pytest.raises(httpx.HTTPStatusError):
        src._get("/fixtures", params={"league": 1, "season": 2024})


def test_fetch_tolerates_per_league_failures() -> None:
    calls: dict[str, int] = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        # First league → HTTP 500, second league → success.
        if calls["n"] <= 3:  # retry x3
            return httpx.Response(500)
        return httpx.Response(
            200,
            json={
                "response": [
                    {
                        "fixture": {
                            "id": 1,
                            "date": "2024-08-17T14:00:00+00:00",
                            "status": {"short": "FT"},
                        },
                        "teams": {
                            "home": {"name": "Arsenal"},
                            "away": {"name": "Chelsea"},
                        },
                        "goals": {"home": 1, "away": 0},
                    }
                ],
                "errors": [],
            },
        )

    client = _stub_client(handler=handler)
    src = ApiFootballSource(
        api_key="test",
        leagues={
            39: ("eng-premier-league", "Premier League"),
            135: ("ita-serie-a", "Serie A"),
        },
        rate_limit_seconds=0.0,
        client=client,
    )
    result = src.fetch(season="2024")
    # Premier League failed after retries but Serie A succeeded.
    assert len(result.matches) == 1
    assert result.matches[0].competition_code == "ita-serie-a"


def test_throttle_noop_when_rate_limit_zero() -> None:
    src = ApiFootballSource(api_key="test", rate_limit_seconds=0.0)
    # Should not raise or sleep.
    src._throttle()
    src._throttle()
