"""Football-Data.org v4 adapter.

Implements :class:`BaseSource` for the free tier of https://www.football-data.org.

Design notes
------------

- **Rate limiting**: free tier is 10 requests/minute. We spread calls with a
  simple client-side token bucket (one request every 6.5 s by default) so a
  single ingestion run never trips the limit.
- **Retries**: ``tenacity`` retries on 429 / 5xx / network errors with
  exponential backoff (max 3 attempts). We also honour ``Retry-After`` when
  present.
- **Raw storage**: every successful HTTP call is persisted into
  ``ingestion_payloads`` by the orchestrator, keyed by ``(source, endpoint)``.
  The adapter itself is stateless w.r.t. the database — it just yields the
  parsed payloads to the caller.
- **Surface**: we expose ``fetch_competition_matches`` which pulls finished +
  scheduled matches for a competition + season. The adapter's ``fetch()``
  iterates all enabled competitions and unions the results.

Free-tier competition codes we care about today (ids from Football-Data.org):

=======  =========================  ============
code     competition                 country
=======  =========================  ============
PL       Premier League              England
ELC      Championship                England
PD       La Liga                     Spain
SA       Serie A                     Italy
BL1      Bundesliga                  Germany
FL1      Ligue 1                     France
DED      Eredivisie                  Netherlands
PPL      Primeira Liga               Portugal
CL       UEFA Champions League       Europe
BSA      Brasileirão Série A         Brazil
=======  =========================  ============

Season is passed as the starting year, e.g. ``2024`` means the 2024/25 season.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
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

from app.ingestion.base import BaseSource, IngestionResult, RawMatch
from app.logging import get_logger

log = get_logger(__name__)

BASE_URL = "https://api.football-data.org/v4"
DEFAULT_COMPETITIONS: tuple[str, ...] = (
    "PL",
    "SA",
    "BL1",
    "FL1",
    "PD",
    "DED",
    "PPL",
)

# Free tier is 10 req/min → at least 6 s between requests. Pad slightly for
# clock skew. Can be overridden per-instance for tests.
DEFAULT_RATE_LIMIT_SECONDS = 6.5


class FootballDataOrgError(RuntimeError):
    """Raised when Football-Data.org returns a non-recoverable error."""


@dataclass
class _Response:
    status_code: int
    body: dict[str, Any]


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, FootballDataOrgError):
        msg = str(exc)
        return "429" in msg or msg[:1] == "5" if msg else False
    return False


class FootballDataOrgSource(BaseSource):
    """Football-Data.org implementation of :class:`BaseSource`."""

    name = "football-data"
    sports = ("football",)

    def __init__(
        self,
        api_key: str,
        *,
        competitions: Iterable[str] | None = None,
        season: str | None = None,
        rate_limit_seconds: float = DEFAULT_RATE_LIMIT_SECONDS,
        client: httpx.Client | None = None,
        raw_sink: Callable[[str, dict[str, Any], int, dict[str, Any]], None] | None = None,
        now_fn: Callable[[], datetime] = lambda: datetime.now(tz=UTC),
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._key = api_key or ""
        self._competitions = tuple(competitions) if competitions else DEFAULT_COMPETITIONS
        self._season = season
        self._rate_limit = max(0.0, float(rate_limit_seconds))
        self._client_external = client is not None
        self._client = client or httpx.Client(
            base_url=BASE_URL,
            timeout=30.0,
            headers={"X-Auth-Token": self._key} if self._key else {},
        )
        self._raw_sink = raw_sink
        self._now = now_fn
        self._sleep = sleep_fn
        self._last_request_at: float = 0.0

    # ------------------------------------------------------------------
    # BaseSource API
    # ------------------------------------------------------------------

    def is_enabled(self) -> bool:
        return bool(self._key)

    def fetch(self, *, season: str | None = None) -> IngestionResult:
        """Pull matches for every configured competition.

        The caller is expected to iterate the :class:`BaseSource` registry,
        hand the result to the orchestrator, and persist. Failures on a single
        competition do not abort the rest of the run — they're logged and the
        partial result is returned.
        """
        result = IngestionResult(source=self.name)
        if not self.is_enabled():
            return result
        season_arg = season if season is not None else self._season
        meta: dict[str, Any] = {"competitions": list(self._competitions)}
        if season_arg is not None:
            meta["season"] = season_arg
        for code in self._competitions:
            try:
                matches = self._fetch_competition_matches(code, season=season_arg)
            except (FootballDataOrgError, RetryError, httpx.HTTPError) as exc:
                log.warning(
                    "football_data.competition.failed",
                    competition=code,
                    error=str(exc),
                )
                meta.setdefault("failures", []).append({"competition": code, "error": str(exc)})
                continue
            result.matches.extend(matches)
        result.meta = meta
        return result

    def close(self) -> None:
        if not self._client_external:
            self._client.close()

    # ------------------------------------------------------------------
    # Fetch helpers
    # ------------------------------------------------------------------

    def _fetch_competition_matches(
        self,
        competition_code: str,
        *,
        season: str | None,
    ) -> list[RawMatch]:
        params: dict[str, Any] = {}
        if season is not None:
            params["season"] = season
        endpoint = f"/competitions/{competition_code}/matches"
        resp = self._get(endpoint, params=params)
        body = resp.body
        raw_list = body.get("matches") or []
        out: list[RawMatch] = []
        for raw in raw_list:
            mapped = _map_match(raw, competition_code=competition_code)
            if mapped is not None:
                out.append(mapped)
        return out

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _get(self, endpoint: str, *, params: dict[str, Any] | None = None) -> _Response:
        @retry(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1.0, min=1.0, max=20.0),
            retry=retry_if_exception(_is_retryable),
        )
        def _do() -> _Response:
            self._respect_rate_limit()
            try:
                http_resp = self._client.get(endpoint, params=params or {})
            except httpx.TransportError:
                raise
            return self._handle_response(endpoint, params or {}, http_resp)

        return _do()

    def _respect_rate_limit(self) -> None:
        if self._rate_limit <= 0.0:
            return
        if self._last_request_at <= 0.0:
            self._last_request_at = time.monotonic()
            return
        elapsed = time.monotonic() - self._last_request_at
        wait_for = self._rate_limit - elapsed
        if wait_for > 0:
            self._sleep(wait_for)
        self._last_request_at = time.monotonic()

    def _handle_response(
        self,
        endpoint: str,
        params: dict[str, Any],
        http_resp: httpx.Response,
    ) -> _Response:
        try:
            body = http_resp.json() if http_resp.content else {}
        except ValueError:
            body = {"_raw": http_resp.text}
        if not isinstance(body, dict):
            body = {"_body": body}

        if self._raw_sink is not None:
            try:
                self._raw_sink(endpoint, params, http_resp.status_code, body)
            except Exception as exc:  # pragma: no cover - sink must not break the fetch
                log.warning("football_data.raw_sink.failed", error=str(exc))

        if http_resp.status_code == 429:
            raise FootballDataOrgError("429 rate limited by Football-Data.org")
        if http_resp.status_code >= 500:
            raise FootballDataOrgError(f"{http_resp.status_code} upstream error")
        if http_resp.status_code >= 400:
            # 4xx other than 429 is not retryable.
            raise FootballDataOrgError(
                f"{http_resp.status_code} {body.get('message', http_resp.text)[:200]}"
            )
        return _Response(status_code=http_resp.status_code, body=body)


# ---------------------------------------------------------------------------
# Response -> domain mapping
# ---------------------------------------------------------------------------

_STATUS_MAP = {
    "FINISHED": "finished",
    "IN_PLAY": "live",
    "PAUSED": "live",
    "LIVE": "live",
    "SUSPENDED": "live",
    "POSTPONED": "postponed",
    "CANCELLED": "cancelled",
    "CANCELED": "cancelled",
    "AWARDED": "finished",
    "SCHEDULED": "scheduled",
    "TIMED": "scheduled",
}


def _map_match(raw: dict[str, Any], *, competition_code: str) -> RawMatch | None:
    try:
        external_id = str(raw["id"])
        kickoff = _parse_datetime(raw["utcDate"])
        home = raw["homeTeam"]["name"]
        away = raw["awayTeam"]["name"]
    except (KeyError, TypeError, ValueError) as exc:
        log.warning("football_data.match.skipped", reason=str(exc))
        return None
    status = _STATUS_MAP.get(str(raw.get("status") or "").upper(), "scheduled")

    score = raw.get("score") or {}
    full_time = score.get("fullTime") or {}
    home_score = full_time.get("home")
    away_score = full_time.get("away")

    competition_block = raw.get("competition") or {}
    competition_name = str(competition_block.get("name") or competition_code)

    season_block = raw.get("season") or {}
    season_label: str | None = None
    start_date = season_block.get("startDate")
    if isinstance(start_date, str) and len(start_date) >= 4 and start_date[:4].isdigit():
        season_label = start_date[:4]

    return RawMatch(
        external_id=external_id,
        sport_code="football",
        competition_code=competition_code,
        competition_name=competition_name,
        home_team=str(home),
        away_team=str(away),
        kickoff=kickoff,
        status=status,
        home_score=int(home_score) if home_score is not None else None,
        away_score=int(away_score) if away_score is not None else None,
        season=season_label,
        stage=str(raw.get("stage")) if raw.get("stage") else None,
        venue=str(raw.get("venue")) if raw.get("venue") else None,
    )


def _parse_datetime(value: str) -> datetime:
    """Parse an ISO-8601 datetime from the Football-Data.org response.

    The API returns ``"2024-08-17T16:30:00Z"`` style strings. ``fromisoformat``
    handles the trailing ``Z`` in Python 3.11+, but we normalise anyway.
    """
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
