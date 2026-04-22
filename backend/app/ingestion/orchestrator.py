"""Orchestrates ingestion: picks active adapters, fetches raw data, upserts into the DB."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from app.db import models as m
from app.ingestion.base import BaseSource, IngestionResult, RawOdds, RawStat
from app.ingestion.registry import get_active_sources
from app.logging import get_logger
from app.odds.history import mark_closing_odds

log = get_logger(__name__)


# Monotonic lifecycle order used when multiple sources disagree on the status
# of the same match. Unknown statuses default to 0 so they can never demote a
# known status.
_STATUS_PRIORITY: dict[str, int] = {
    "scheduled": 0,
    "postponed": 1,
    "live": 2,
    "cancelled": 3,
    "finished": 4,
}


def ingest_all(
    db: Session,
    *,
    sources: Iterable[BaseSource] | None = None,
    trigger: str = "manual",
) -> dict[str, int]:
    """Run every active ingestion source and persist results.

    Each source invocation is recorded in ``ingestion_runs`` with a status
    summary. Sources that expose a ``raw_sink`` hook also persist every HTTP
    payload into ``ingestion_payloads`` for full audit / replay.
    """
    counts: dict[str, int] = {"matches": 0, "stats": 0, "odds": 0}
    active = list(sources) if sources is not None else get_active_sources()
    for source in active:
        started_at = datetime.now(tz=UTC)
        run = m.IngestionRun(
            source=source.name,
            trigger=trigger,
            started_at=started_at,
            ok=True,
        )
        db.add(run)
        db.flush()

        # Wire the raw sink through to the source only if it supports one and
        # one wasn't already supplied by the caller (e.g. in tests).
        if hasattr(source, "_raw_sink") and source._raw_sink is None:
            source._raw_sink = _make_raw_sink(db, source.name)

        try:
            log.info("ingest.source.start", source=source.name)
            result = source.fetch()
        except NotImplementedError as exc:
            log.warning("ingest.source.not_implemented", source=source.name, reason=str(exc))
            run.ok = False
            run.error = f"not_implemented: {exc}"[:1024]
            run.finished_at = datetime.now(tz=UTC)
            continue
        except Exception as exc:  # noqa: BLE001
            log.exception("ingest.source.failed", source=source.name, error=str(exc))
            run.ok = False
            run.error = str(exc)[:1024]
            run.finished_at = datetime.now(tz=UTC)
            continue

        c = _persist(db, result)
        for k, v in c.items():
            counts[k] = counts.get(k, 0) + v

        run.matches_upserted = c["matches"]
        run.stats_upserted = c["stats"]
        run.odds_upserted = c["odds"]
        run.finished_at = datetime.now(tz=UTC)
        run.meta = dict(result.meta)
        log.info("ingest.source.done", source=source.name, **c)
    # Mark closing lines on any match whose kickoff has elapsed. Safe to call
    # when no new odds were ingested — it is a no-op in that case.
    try:
        mark_closing_odds(db)
    except Exception as exc:  # noqa: BLE001 - closing mark must not fail ingestion
        log.exception("ingest.closing_mark.failed", error=str(exc))
    db.commit()
    return counts


def _make_raw_sink(db: Session, source_name: str):
    def sink(
        endpoint: str,
        params: dict[str, Any],
        status_code: int,
        payload: dict[str, Any],
    ) -> None:
        db.add(
            m.IngestionPayload(
                source=source_name,
                endpoint=endpoint[:256],
                params=dict(params),
                status_code=status_code,
                payload=payload,
                fetched_at=datetime.now(tz=UTC),
                ok=200 <= status_code < 400,
            )
        )
        db.flush()

    return sink


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _get_or_create_sport(db: Session, code: str) -> m.Sport:
    sport = db.query(m.Sport).filter_by(code=code).one_or_none()
    if sport is None:
        sport = m.Sport(code=code, name=_sport_display(code))
        db.add(sport)
        db.flush()
    return sport


def _sport_display(code: str) -> str:
    return {"football": "Football", "table_tennis": "Table Tennis"}.get(code, code.title())


def _get_or_create_competition(
    db: Session, sport: m.Sport, code: str, name: str, country: str | None
) -> m.Competition:
    comp = db.query(m.Competition).filter_by(sport_id=sport.id, code=code).one_or_none()
    if comp is None:
        comp = m.Competition(sport_id=sport.id, code=code, name=name, country=country)
        db.add(comp)
        db.flush()
    return comp


def _get_or_create_team(db: Session, competition: m.Competition | None, name: str) -> m.Team:
    comp_id = competition.id if competition else None
    team = (
        db.query(m.Team).filter(m.Team.name == name, m.Team.competition_id == comp_id).one_or_none()
    )
    if team is None:
        team = m.Team(name=name, competition_id=comp_id)
        db.add(team)
        db.flush()
    return team


def _persist(db: Session, result: IngestionResult) -> dict[str, int]:
    cnt = {"matches": 0, "stats": 0, "odds": 0}

    # Index teams by (competition_code, team_name) so stats/odds can look up match ids.
    match_by_ext: dict[str, m.Match] = {}
    teams_by_match: dict[str, dict[str, m.Team]] = {}

    for raw in result.matches:
        sport = _get_or_create_sport(db, raw.sport_code)
        comp = _get_or_create_competition(
            db,
            sport,
            raw.competition_code,
            raw.competition_name,
            getattr(raw, "country", None),
        )
        home = _get_or_create_team(db, comp, raw.home_team)
        away = _get_or_create_team(db, comp, raw.away_team)

        match = (
            db.query(m.Match)
            .filter(
                m.Match.sport_id == sport.id,
                m.Match.home_team_id == home.id,
                m.Match.away_team_id == away.id,
                m.Match.kickoff == raw.kickoff,
            )
            .one_or_none()
        )
        if match is None:
            match = m.Match(
                sport_id=sport.id,
                competition_id=comp.id,
                home_team_id=home.id,
                away_team_id=away.id,
                kickoff=raw.kickoff,
                status=raw.status,
                home_score=raw.home_score,
                away_score=raw.away_score,
                season=raw.season,
                stage=raw.stage,
                venue=raw.venue,
            )
            db.add(match)
            db.flush()
            cnt["matches"] += 1
        else:
            # Only promote the status forward in the match lifecycle. Without
            # this guard a source that always reports "scheduled" (e.g. The
            # Odds API, which doesn't carry a real lifecycle field) would
            # regress a match that another source has already flagged as
            # "live" or "finished".
            if _STATUS_PRIORITY.get(raw.status, 0) >= _STATUS_PRIORITY.get(
                match.status, 0
            ):
                match.status = raw.status
            if raw.home_score is not None:
                match.home_score = raw.home_score
            if raw.away_score is not None:
                match.away_score = raw.away_score
        match_by_ext[raw.external_id] = match
        teams_by_match[raw.external_id] = {raw.home_team: home, raw.away_team: away}

    # Stats
    for stat in result.stats:
        match = match_by_ext.get(stat.match_external_id)
        if match is None:
            continue
        team_map = teams_by_match.get(stat.match_external_id, {})
        team = team_map.get(stat.team_name)
        if team is None:
            continue
        existing_stat = (
            db.query(m.MatchStat).filter_by(match_id=match.id, team_id=team.id).one_or_none()
        )
        if existing_stat is None:
            db.add(_build_stat(match.id, team.id, stat))
            cnt["stats"] += 1

    # Odds
    for odds in result.odds:
        match = match_by_ext.get(odds.match_external_id)
        if match is None:
            continue
        existing_odds = (
            db.query(m.OddsSnapshot)
            .filter_by(
                match_id=match.id,
                bookmaker=odds.bookmaker,
                market=odds.market,
                selection=odds.selection,
                line=odds.line,
                captured_at=odds.captured_at,
            )
            .one_or_none()
        )
        if existing_odds is None:
            db.add(_build_odds(match.id, odds))
            cnt["odds"] += 1

    db.flush()
    return cnt


def _build_stat(match_id: int, team_id: int, s: RawStat) -> m.MatchStat:
    return m.MatchStat(
        match_id=match_id,
        team_id=team_id,
        shots=s.shots,
        shots_on_target=s.shots_on_target,
        corners=s.corners,
        yellow_cards=s.yellow_cards,
        red_cards=s.red_cards,
        fouls=s.fouls,
        possession=s.possession,
        xg=s.xg,
        xga=s.xga,
        passes=s.passes,
        pass_accuracy=s.pass_accuracy,
    )


def _build_odds(match_id: int, o: RawOdds) -> m.OddsSnapshot:
    return m.OddsSnapshot(
        match_id=match_id,
        bookmaker=o.bookmaker,
        market=o.market,
        selection=o.selection,
        line=o.line,
        price=o.price,
        captured_at=o.captured_at,
        is_closing=o.is_closing,
        is_live=o.is_live,
    )


__all__ = ["ingest_all"]
