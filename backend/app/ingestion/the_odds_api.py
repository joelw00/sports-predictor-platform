"""The Odds API v4 adapter.

Implements :class:`BaseSource` for https://the-odds-api.com. The adapter
yields matches and odds snapshots across a configurable set of sport keys
(football leagues today; ping pong handled separately in PR 6).

Design notes
------------

- **Sport keys** follow The Odds API taxonomy (``soccer_epl``,
  ``soccer_italy_serie_a``, ``soccer_germany_bundesliga``, ``soccer_spain_la_liga``,
  ``soccer_france_ligue_one``, ``soccer_uefa_champs_league`` …).
- **Markets**: we request ``h2h`` (1X2), ``totals`` (over/under) and
  ``spreads`` (asian handicap) in one call — this is billed as one
  request across all markets per sport key.
- **Regions**: configurable via ``ODDS_API_REGIONS`` (default ``eu,uk``).
  EU includes SNAI on Italian football.
- **Rate limits**: The Odds API returns a per-response header
  ``x-requests-remaining``. We honour ``Retry-After`` on 429 and retry up
  to 3 times with exponential backoff; we log the remaining quota so
  operators see when they're close to exhausting the monthly budget.
- **Raw storage**: identical hook pattern as
  :class:`app.ingestion.football_data_org.FootballDataOrgSource` —
  every HTTP response is forwarded to a ``raw_sink`` callable that the
  orchestrator wires into ``ingestion_payloads``.
- **Team normalisation**: The Odds API returns canonical full names. We
  apply :func:`normalise_team_name` so odds line up with teams already
  ingested from Football-Data.org (e.g. ``"Manchester City FC"`` →
  ``"Manchester City"``).
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

from app.ingestion.base import BaseSource, IngestionResult, RawMatch, RawOdds
from app.ingestion.team_names import normalise_team_name
from app.logging import get_logger

log = get_logger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"

DEFAULT_SPORT_KEYS: tuple[str, ...] = (
    "soccer_epl",
    "soccer_italy_serie_a",
    "soccer_germany_bundesliga",
    "soccer_spain_la_liga",
    "soccer_france_ligue_one",
    "soccer_uefa_champs_league",
)
DEFAULT_MARKETS: tuple[str, ...] = ("h2h", "totals", "spreads")
DEFAULT_REGIONS: tuple[str, ...] = ("eu", "uk")

# 500 requests/month on the free tier -> be gentle. 1 s spacing is plenty.
DEFAULT_RATE_LIMIT_SECONDS = 1.0


#: Map from The Odds API sport_key -> (sport_code, competition_code, competition_name)
_SPORT_KEY_META: dict[str, tuple[str, str, str]] = {
    "soccer_epl": ("football", "PL", "Premier League"),
    "soccer_italy_serie_a": ("football", "SA", "Serie A"),
    "soccer_germany_bundesliga": ("football", "BL1", "Bundesliga"),
    "soccer_spain_la_liga": ("football", "PD", "La Liga"),
    "soccer_france_ligue_one": ("football", "FL1", "Ligue 1"),
    "soccer_uefa_champs_league": ("football", "CL", "UEFA Champions League"),
    "soccer_netherlands_eredivisie": ("football", "DED", "Eredivisie"),
    "soccer_portugal_primeira_liga": ("football", "PPL", "Primeira Liga"),
    "soccer_brazil_campeonato": ("football", "BSA", "Brasileirão Série A"),
    "soccer_uefa_europa_league": ("football", "EL", "UEFA Europa League"),
}


class TheOddsApiError(RuntimeError):
    """Raised when The Odds API returns a non-recoverable error."""


@dataclass
class _Response:
    status_code: int
    body: Any
    headers: dict[str, str]


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, TheOddsApiError):
        msg = str(exc)
        if "429" in msg:
            return True
        if msg[:1] == "5" and len(msg) >= 3 and msg[:3].isdigit():
            return True
    return False


class TheOddsApiRealSource(BaseSource):
    """Real Odds API adapter returning matches + odds snapshots."""

    name = "the-odds-api"
    sports = ("football",)

    def __init__(
        self,
        api_key: str,
        *,
        sport_keys: Iterable[str] | None = None,
        markets: Iterable[str] | None = None,
        regions: Iterable[str] | None = None,
        rate_limit_seconds: float = DEFAULT_RATE_LIMIT_SECONDS,
        client: httpx.Client | None = None,
        raw_sink: Callable[[str, dict[str, Any], int, Any], None] | None = None,
        now_fn: Callable[[], datetime] = lambda: datetime.now(tz=UTC),
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._key = api_key or ""
        self._sport_keys = tuple(sport_keys) if sport_keys else DEFAULT_SPORT_KEYS
        self._markets = tuple(markets) if markets else DEFAULT_MARKETS
        self._regions = tuple(regions) if regions else DEFAULT_REGIONS
        self._rate_limit = max(0.0, float(rate_limit_seconds))
        self._client_external = client is not None
        self._client = client or httpx.Client(base_url=BASE_URL, timeout=30.0)
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
        result = IngestionResult(source=self.name)
        if not self.is_enabled():
            return result
        meta: dict[str, Any] = {
            "sport_keys": list(self._sport_keys),
            "markets": list(self._markets),
            "regions": list(self._regions),
        }
        failures: list[dict[str, str]] = []
        remaining_quota: int | None = None
        for sport_key in self._sport_keys:
            try:
                matches, odds, remaining = self._fetch_sport(sport_key)
            except (TheOddsApiError, RetryError, httpx.HTTPError) as exc:
                log.warning(
                    "the_odds_api.sport.failed",
                    sport_key=sport_key,
                    error=str(exc),
                )
                failures.append({"sport_key": sport_key, "error": str(exc)})
                continue
            result.matches.extend(matches)
            result.odds.extend(odds)
            if remaining is not None:
                remaining_quota = remaining
        if failures:
            meta["failures"] = failures
        if remaining_quota is not None:
            meta["requests_remaining"] = remaining_quota
        result.meta = meta
        return result

    def close(self) -> None:
        if not self._client_external:
            self._client.close()

    # ------------------------------------------------------------------
    # Fetch helpers
    # ------------------------------------------------------------------

    def _fetch_sport(
        self, sport_key: str
    ) -> tuple[list[RawMatch], list[RawOdds], int | None]:
        endpoint = f"/sports/{sport_key}/odds"
        params = {
            "apiKey": self._key,
            "regions": ",".join(self._regions),
            "markets": ",".join(self._markets),
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
        resp = self._get(endpoint, params=params)
        events = resp.body if isinstance(resp.body, list) else []
        matches: list[RawMatch] = []
        odds: list[RawOdds] = []
        captured_at = self._now()
        for evt in events:
            if not isinstance(evt, dict):
                continue
            raw_match = _map_event(evt, sport_key=sport_key)
            if raw_match is None:
                continue
            matches.append(raw_match)
            odds.extend(_map_event_odds(evt, external_id=raw_match.external_id, captured_at=captured_at))
        remaining = _parse_int_header(resp.headers, "x-requests-remaining")
        return matches, odds, remaining

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _get(self, endpoint: str, *, params: dict[str, Any]) -> _Response:
        @retry(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1.0, min=1.0, max=20.0),
            retry=retry_if_exception(_is_retryable),
        )
        def _do() -> _Response:
            self._respect_rate_limit()
            http_resp = self._client.get(endpoint, params=params)
            return self._handle_response(endpoint, params, http_resp)

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
        headers = {k.lower(): v for k, v in http_resp.headers.items()}
        body: Any
        # Empty body -> {} (not None) so the raw_sink can persist it into the
        # NOT NULL ``ingestion_payloads.payload`` column without corrupting the
        # session. Matches Football-Data.org adapter behaviour.
        try:
            body = http_resp.json() if http_resp.content else {}
        except ValueError:
            body = {"_raw": http_resp.text}

        # Don't persist the API key.
        sanitised = {k: v for k, v in params.items() if k != "apiKey"}
        if self._raw_sink is not None:
            try:
                self._raw_sink(endpoint, sanitised, http_resp.status_code, body)
            except Exception as exc:  # pragma: no cover - sink must not break the fetch
                log.warning("the_odds_api.raw_sink.failed", error=str(exc))

        if http_resp.status_code == 429:
            raise TheOddsApiError("429 rate limited by The Odds API")
        if http_resp.status_code >= 500:
            raise TheOddsApiError(f"{http_resp.status_code} upstream error")
        if http_resp.status_code >= 400:
            detail_src = body if isinstance(body, dict) else {}
            detail = detail_src.get("message") or http_resp.text or ""
            raise TheOddsApiError(f"{http_resp.status_code} {detail[:200]}")
        return _Response(status_code=http_resp.status_code, body=body, headers=headers)


# ---------------------------------------------------------------------------
# Response -> domain mapping
# ---------------------------------------------------------------------------


def _map_event(evt: dict[str, Any], *, sport_key: str) -> RawMatch | None:
    ext_id = str(evt.get("id") or "").strip()
    home = evt.get("home_team")
    away = evt.get("away_team")
    commence = evt.get("commence_time")
    if not ext_id or not home or not away or not commence:
        return None
    try:
        kickoff = _parse_datetime(str(commence))
    except ValueError:
        return None
    sport_code, comp_code, comp_name = _SPORT_KEY_META.get(
        sport_key, ("football", sport_key.upper(), sport_key)
    )
    return RawMatch(
        external_id=f"odds-api:{ext_id}",
        sport_code=sport_code,
        competition_code=comp_code,
        competition_name=comp_name,
        home_team=normalise_team_name(str(home)),
        away_team=normalise_team_name(str(away)),
        kickoff=kickoff,
        status="scheduled",
    )


def _map_event_odds(
    evt: dict[str, Any], *, external_id: str, captured_at: datetime
) -> list[RawOdds]:
    out: list[RawOdds] = []
    home = normalise_team_name(str(evt.get("home_team") or ""))
    away = normalise_team_name(str(evt.get("away_team") or ""))
    for book in evt.get("bookmakers") or []:
        if not isinstance(book, dict):
            continue
        book_key = str(book.get("key") or "").strip() or "unknown"
        for mkt in book.get("markets") or []:
            if not isinstance(mkt, dict):
                continue
            market_key = str(mkt.get("key") or "").strip()
            for outcome in mkt.get("outcomes") or []:
                if not isinstance(outcome, dict):
                    continue
                mapped = _map_outcome(
                    market_key=market_key,
                    outcome=outcome,
                    home=home,
                    away=away,
                )
                if mapped is None:
                    continue
                selection, line = mapped
                raw_price = outcome.get("price")
                if raw_price is None:
                    continue
                try:
                    price = float(raw_price)
                except (TypeError, ValueError):
                    continue
                if price <= 1.01:
                    # Clearly bogus / placeholder price.
                    continue
                out.append(
                    RawOdds(
                        match_external_id=external_id,
                        bookmaker=book_key,
                        market=_market_code(market_key),
                        selection=selection,
                        price=price,
                        captured_at=captured_at,
                        line=line,
                        is_closing=False,
                        is_live=False,
                    )
                )
    return out


def _market_code(key: str) -> str:
    return {"h2h": "1X2", "totals": "OU", "spreads": "AH"}.get(key, key)


def _map_outcome(
    *,
    market_key: str,
    outcome: dict[str, Any],
    home: str,
    away: str,
) -> tuple[str, float | None] | None:
    name = str(outcome.get("name") or "").strip()
    if not name:
        return None
    if market_key == "h2h":
        norm = normalise_team_name(name)
        if norm == home:
            return "home", None
        if norm == away:
            return "away", None
        if name.lower() in {"draw", "tie"}:
            return "draw", None
        return None
    if market_key == "totals":
        raw_point = outcome.get("point")
        if raw_point is None:
            return None
        try:
            line = float(raw_point)
        except (TypeError, ValueError):
            return None
        if name.lower() == "over":
            return "over", line
        if name.lower() == "under":
            return "under", line
        return None
    if market_key == "spreads":
        raw_point = outcome.get("point")
        if raw_point is None:
            return None
        try:
            line = float(raw_point)
        except (TypeError, ValueError):
            return None
        norm = normalise_team_name(name)
        if norm == home:
            return "home", line
        if norm == away:
            return "away", line
        return None
    return None


def _parse_datetime(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    # Guarantee tz-aware datetimes so matches land in the same timezone as the
    # Football-Data.org adapter and don't create duplicate kickoff rows.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _parse_int_header(headers: dict[str, str], name: str) -> int | None:
    raw = headers.get(name.lower())
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


__all__ = [
    "DEFAULT_MARKETS",
    "DEFAULT_REGIONS",
    "DEFAULT_SPORT_KEYS",
    "TheOddsApiError",
    "TheOddsApiRealSource",
]
