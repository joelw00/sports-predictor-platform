from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.db import models as m
from app.schemas import (
    Event,
    EventList,
    MatchDetail,
    OddsQuote,
    PredictionOut,
    TeamOut,
    ValueBetOut,
)

router = APIRouter(tags=["events"])


def _day_bounds(d: date) -> tuple[datetime, datetime]:
    start = datetime.combine(d, time.min, tzinfo=UTC)
    return start, start + timedelta(days=1)


@router.get("/events", response_model=EventList)
def list_events(
    sport: str | None = None,
    day: date | None = None,
    competition: str | None = None,
    status: str | None = None,
    db: Session = Depends(get_db),
) -> EventList:
    q = db.query(m.Match).join(m.Sport, m.Match.sport_id == m.Sport.id)
    if sport:
        q = q.filter(m.Sport.code == sport)
    if status:
        q = q.filter(m.Match.status == status)
    if competition:
        q = q.join(m.Competition, m.Match.competition_id == m.Competition.id).filter(
            m.Competition.code == competition
        )
    if day:
        start, end = _day_bounds(day)
        q = q.filter(m.Match.kickoff >= start, m.Match.kickoff < end)
    else:
        today = date.today()
        start = datetime.combine(today, time.min, tzinfo=UTC)
        end = start + timedelta(days=3)
        q = q.filter(m.Match.kickoff >= start - timedelta(days=1), m.Match.kickoff < end)

    matches = q.order_by(m.Match.kickoff.asc()).all()
    items: list[Event] = []
    for match in matches:
        items.append(_to_event(db, match))
    return EventList(items=items, total=len(items))


@router.get("/events/{match_id}", response_model=MatchDetail)
def get_event(match_id: int, db: Session = Depends(get_db)) -> MatchDetail:
    match = db.get(m.Match, match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="match not found")
    home = db.get(m.Team, match.home_team_id)
    away = db.get(m.Team, match.away_team_id)
    preds = (
        db.query(m.Prediction)
        .filter(m.Prediction.match_id == match.id)
        .order_by(m.Prediction.market, m.Prediction.selection)
        .all()
    )
    odds = (
        db.query(m.OddsSnapshot)
        .filter(m.OddsSnapshot.match_id == match.id)
        .order_by(m.OddsSnapshot.captured_at.desc())
        .limit(100)
        .all()
    )
    value_bets = (
        db.query(m.ValueBet)
        .filter(m.ValueBet.match_id == match.id)
        .order_by(m.ValueBet.edge.desc())
        .all()
    )
    comp = db.get(m.Competition, match.competition_id) if match.competition_id else None
    sport = db.get(m.Sport, match.sport_id)

    return MatchDetail(
        id=match.id,
        sport=sport.code if sport else "unknown",
        competition=comp.name if comp else None,
        home_team=TeamOut.model_validate(home),
        away_team=TeamOut.model_validate(away),
        kickoff=match.kickoff,
        status=match.status,
        home_score=match.home_score,
        away_score=match.away_score,
        predictions=[PredictionOut.model_validate(p) for p in preds],
        odds=[OddsQuote.model_validate(o) for o in odds],
        value_bets=[_hydrate_value_bet(db, vb) for vb in value_bets],
        form={},
    )


def _to_event(db: Session, match: m.Match) -> Event:
    home = db.get(m.Team, match.home_team_id)
    away = db.get(m.Team, match.away_team_id)
    sport = db.get(m.Sport, match.sport_id)
    comp = db.get(m.Competition, match.competition_id) if match.competition_id else None

    # Top 1x2 prediction
    top_pred = (
        db.query(m.Prediction)
        .filter(m.Prediction.match_id == match.id, m.Prediction.market == "1x2")
        .order_by(m.Prediction.probability.desc())
        .first()
    )
    best_value = (
        db.query(m.ValueBet)
        .filter(m.ValueBet.match_id == match.id)
        .order_by(m.ValueBet.edge.desc())
        .first()
    )
    return Event(
        id=match.id,
        sport=sport.code if sport else "unknown",
        competition=comp.name if comp else None,
        home_team=home.name if home else "?",
        away_team=away.name if away else "?",
        kickoff=match.kickoff,
        status=match.status,
        home_score=match.home_score,
        away_score=match.away_score,
        top_prediction=PredictionOut.model_validate(top_pred) if top_pred else None,
        best_value=_hydrate_value_bet(db, best_value) if best_value else None,
    )


def _hydrate_value_bet(db: Session, vb: m.ValueBet) -> ValueBetOut:
    match = db.get(m.Match, vb.match_id)
    out = ValueBetOut.model_validate(vb)
    if match is not None:
        home = db.get(m.Team, match.home_team_id)
        away = db.get(m.Team, match.away_team_id)
        sport = db.get(m.Sport, match.sport_id)
        comp = db.get(m.Competition, match.competition_id) if match.competition_id else None
        out.home_team = home.name if home else None
        out.away_team = away.name if away else None
        out.competition = comp.name if comp else None
        out.kickoff = match.kickoff
        out.sport = sport.code if sport else None
    return out
