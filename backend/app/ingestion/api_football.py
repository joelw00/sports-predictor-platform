"""api-sports.io (API-Football) adapter.

Uses the direct https://v3.football.api-sports.io endpoint (free plan: 100
requests/day, no credit card). Documented at https://www.api-football.com.

The adapter is intentionally thin — we only fetch fixtures for a small set of
leagues + their per-fixture statistics so we can populate corners/cards on top
of whatever Football-Data.org already gave us. The daily 100-request budget is
easy to blow through if we're not careful, so the caller is expected to pass
tight ``league_ids``/``season`` windows rather than letting us walk every
competition.

Auth header: ``x-apisports-key: <API_FOOTBALL_KEY>``.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

import httpx
from tenacity import (
    RetryError,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.ingestion.base import BaseSource, IngestionResult, RawMatch, RawStat
from app.ingestion.team_names import normalise_team_name
from app.logging import get_logger

log = get_logger(__name__)

BASE_URL = "https://v3.football.api-sports.io"

# league_id → (our competition code, human name). The ids below are the
# standard api-sports.io league ids for the top-5 European competitions.
DEFAULT_LEAGUES: dict[int, tuple[str, str]] = {
    39: ("eng-premier-league", "Premier League"),
    140: ("esp-la-liga", "La Liga"),
    135: ("ita-serie-a", "Serie A"),
    78: ("ger-bundesliga", "Bundesliga"),
    61: ("fra-ligue-1", "Ligue 1"),
}

# Free plan: 30 req/min, 100 req/day. Keep a small floor between calls.
DEFAULT_RATE_LIMIT_SECONDS = 2.0


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, httpx.RequestError)


class ApiFootballSource(BaseSource):
    """Minimal API-Football adapter for fixtures + statistics."""

    name = "api-football"
    sports = ("football",)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        leagues: dict[int, tuple[str, str]] | None = None,
        rate_limit_seconds: float = DEFAULT_RATE_LIMIT_SECONDS,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("API_FOOTBALL_KEY")
        self.leagues = leagues or DEFAULT_LEAGUES
        self.rate_limit_seconds = rate_limit_seconds
        self._client = client
        self._last_request_at: float | None = None

    # ------------------------------------------------------------------
    # BaseSource interface
    # ------------------------------------------------------------------

    def is_enabled(self) -> bool:
        return bool(self.api_key)

    def fetch(self, *, season: str | None = None) -> IngestionResult:
        if not self.is_enabled():
            return IngestionResult(source=self.name, meta={"disabled": True})
        try:
            season_int = int(season) if season else datetime.now(tz=UTC).year
        except (TypeError, ValueError):
            season_int = datetime.now(tz=UTC).year

        matches: list[RawMatch] = []
        stats: list[RawStat] = []
        for league_id, (comp_code, comp_name) in self.leagues.items():
            try:
                payload = self._get(
                    "/fixtures",
                    params={"league": league_id, "season": season_int},
                )
            except (httpx.HTTPError, RetryError) as exc:
                log.warning(
                    "apifootball.fixtures_failed",
                    league=league_id,
                    error=str(exc),
                )
                continue
            partial = self._parse_fixtures(payload, comp_code, comp_name, str(season_int))
            matches.extend(partial.matches)
            # For stats we'd need N extra API calls (one per finished fixture);
            # the free daily budget can't sustain that in one shot. Callers can
            # enrich incrementally via ``fetch_fixture_statistics``.
        return IngestionResult(
            matches=matches,
            stats=stats,
            source=self.name,
            meta={"leagues": list(self.leagues.keys()), "season": season_int},
        )

    # ------------------------------------------------------------------
    # Per-fixture statistics
    # ------------------------------------------------------------------

    def fetch_fixture_statistics(
        self, fixture_id: int, *, home_external_id: str
    ) -> list[RawStat]:
        """Fetch corners / cards / shots for a single fixture.

        ``home_external_id`` is the ``match_external_id`` we want the resulting
        :class:`RawStat` rows to link back to.
        """
        if not self.is_enabled():
            return []
        try:
            payload = self._get("/fixtures/statistics", params={"fixture": fixture_id})
        except (httpx.HTTPError, RetryError) as exc:
            log.warning(
                "apifootball.stats_failed", fixture=fixture_id, error=str(exc)
            )
            return []
        stats: list[RawStat] = []
        for team_block in payload.get("response", []) or []:
            team_name = normalise_team_name(
                (team_block.get("team") or {}).get("name", "")
            )
            by_key: dict[str, Any] = {
                (row.get("type") or "").strip().lower(): row.get("value")
                for row in team_block.get("statistics", []) or []
            }
            stats.append(
                RawStat(
                    match_external_id=home_external_id,
                    team_name=team_name,
                    shots=_to_int(by_key.get("total shots")),
                    shots_on_target=_to_int(by_key.get("shots on goal")),
                    corners=_to_int(by_key.get("corner kicks")),
                    yellow_cards=_to_int(by_key.get("yellow cards")),
                    red_cards=_to_int(by_key.get("red cards")),
                    fouls=_to_int(by_key.get("fouls")),
                    possession=_to_float(
                        str(by_key.get("ball possession") or "").rstrip("%")
                    ),
                    passes=_to_int(by_key.get("total passes")),
                    pass_accuracy=_to_float(
                        str(by_key.get("passes %") or "").rstrip("%")
                    ),
                )
            )
        return stats

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=1.0, min=1.0, max=20.0),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._throttle()
        client = self._client or httpx.Client(
            base_url=BASE_URL,
            timeout=15.0,
            headers={"x-apisports-key": self.api_key or ""},
        )
        own_client = self._client is None
        try:
            resp = client.get(path, params=params)
            resp.raise_for_status()
            payload = resp.json()
            errors = payload.get("errors")
            if errors and isinstance(errors, dict) and errors:
                # api-sports.io returns HTTP 200 with "errors" on quota/auth issues.
                raise httpx.HTTPStatusError(
                    f"api-football errors: {errors}",
                    request=resp.request,
                    response=resp,
                )
            return payload if isinstance(payload, dict) else {}
        finally:
            if own_client:
                client.close()

    def _throttle(self) -> None:
        if self.rate_limit_seconds <= 0:
            return
        now = time.monotonic()
        if self._last_request_at is not None:
            elapsed = now - self._last_request_at
            gap = self.rate_limit_seconds - elapsed
            if gap > 0:
                time.sleep(gap)
        self._last_request_at = time.monotonic()

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_fixtures(
        payload: dict[str, Any],
        competition_code: str,
        competition_name: str,
        season_tag: str,
    ) -> IngestionResult:
        matches: list[RawMatch] = []
        for fixture in payload.get("response", []) or []:
            fx = fixture.get("fixture") or {}
            teams = fixture.get("teams") or {}
            goals = fixture.get("goals") or {}
            home = normalise_team_name((teams.get("home") or {}).get("name", ""))
            away = normalise_team_name((teams.get("away") or {}).get("name", ""))
            if not home or not away:
                continue
            ts = fx.get("date") or fx.get("timestamp")
            kickoff = _parse_datetime(ts)
            if kickoff is None:
                continue
            status_short = (fx.get("status") or {}).get("short", "")
            status = _map_status(status_short)
            fixture_id = fx.get("id")
            external_id = (
                f"apisports:{competition_code}:{kickoff.strftime('%Y%m%d')}:{fixture_id}"
                if fixture_id
                else f"apisports:{competition_code}:{kickoff.strftime('%Y%m%d')}:{home}-{away}"
            )
            matches.append(
                RawMatch(
                    external_id=external_id,
                    sport_code="football",
                    competition_code=competition_code,
                    competition_name=competition_name,
                    home_team=home,
                    away_team=away,
                    kickoff=kickoff,
                    status=status,
                    home_score=_to_int(goals.get("home")),
                    away_score=_to_int(goals.get("away")),
                    season=season_tag,
                )
            )
        return IngestionResult(matches=matches, source="api-football")


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.astimezone(UTC)
        except ValueError:
            return None
    return None


def _map_status(api_status: str) -> str:
    # api-sports.io short codes: https://www.api-football.com/documentation-v3#section/Fixtures/Status
    if api_status in ("TBD", "NS"):
        return "scheduled"
    if api_status in ("1H", "HT", "2H", "ET", "P", "BT", "LIVE", "INT"):
        return "live"
    if api_status in ("FT", "AET", "PEN"):
        return "finished"
    if api_status == "PST":
        return "postponed"
    if api_status in ("CANC", "ABD", "AWD", "WO", "SUSP"):
        return "cancelled"
    return "scheduled"


def _retry_wrapper(fn: Iterable[Any]) -> Iterable[Any]:  # pragma: no cover
    # placeholder helper — kept so callers can hook their own retry policy later
    return fn


__all__ = ["ApiFootballSource", "DEFAULT_LEAGUES"]
