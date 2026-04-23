"""Football feature engineering.

Produces a `MatchFeatures` dataclass for every past or upcoming match. Features
lag the target by ≥ 1 match to avoid leakage and include Elo ratings, rolling
form, goal rates, xG rates, head-to-head aggregates and rest days.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import models as m


@dataclass
class MatchFeatures:
    match_id: int
    home_team: str
    away_team: str
    competition: str | None
    kickoff: datetime
    home_elo: float
    away_elo: float
    elo_diff: float
    home_form: float
    away_form: float
    home_goals_scored_avg: float
    home_goals_conceded_avg: float
    away_goals_scored_avg: float
    away_goals_conceded_avg: float
    home_xg_for_avg: float
    home_xg_against_avg: float
    away_xg_for_avg: float
    away_xg_against_avg: float
    h2h_home_win_rate: float
    h2h_draw_rate: float
    h2h_goals_avg: float
    home_rest_days: float
    away_rest_days: float
    home_shots_avg: float
    away_shots_avg: float
    home_corners_for_avg: float
    home_corners_against_avg: float
    away_corners_for_avg: float
    away_corners_against_avg: float
    home_cards_avg: float
    away_cards_avg: float

    def as_dict(self) -> dict[str, float | int | str | datetime | None]:
        return asdict(self)


@dataclass
class _TeamState:
    elo: float = 1500.0
    last_match_at: datetime | None = None
    results: deque = field(default_factory=lambda: deque(maxlen=10))  # 1/0.5/0
    goals_for: deque = field(default_factory=lambda: deque(maxlen=10))
    goals_against: deque = field(default_factory=lambda: deque(maxlen=10))
    xg_for: deque = field(default_factory=lambda: deque(maxlen=10))
    xg_against: deque = field(default_factory=lambda: deque(maxlen=10))
    shots: deque = field(default_factory=lambda: deque(maxlen=10))
    corners_for: deque = field(default_factory=lambda: deque(maxlen=10))
    corners_against: deque = field(default_factory=lambda: deque(maxlen=10))
    cards: deque = field(default_factory=lambda: deque(maxlen=10))


@dataclass
class _H2HState:
    home_wins: int = 0
    draws: int = 0
    away_wins: int = 0
    goals: int = 0
    n: int = 0


class FootballFeatureBuilder:
    """Incremental builder — process matches in chronological order."""

    K_FACTOR = 20.0
    HOME_ADVANTAGE = 60.0

    def __init__(self) -> None:
        self._teams: dict[int, _TeamState] = defaultdict(_TeamState)
        self._h2h: dict[tuple[int, int], _H2HState] = defaultdict(_H2HState)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_for_matches(self, db: Session, matches: Iterable[m.Match]) -> list[MatchFeatures]:
        chronological = sorted(matches, key=lambda x: x.kickoff)
        features: list[MatchFeatures] = []
        for match in chronological:
            feats = self._snapshot(db, match)
            features.append(feats)
            if match.status == "finished" and match.home_score is not None:
                self._update_with_result(db, match)
        return features

    def snapshot(self, db: Session, match: m.Match) -> MatchFeatures:
        return self._snapshot(db, match)

    def load_history(self, db: Session, sport_code: str = "football") -> None:
        """Warm the state with all finished matches, ordered by kickoff."""
        stmt = (
            select(m.Match)
            .join(m.Sport, m.Match.sport_id == m.Sport.id)
            .where(m.Sport.code == sport_code, m.Match.status == "finished")
            .order_by(m.Match.kickoff.asc())
        )
        for match in db.scalars(stmt):
            self._update_with_result(db, match)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _snapshot(self, db: Session, match: m.Match) -> MatchFeatures:
        home = self._teams[match.home_team_id]
        away = self._teams[match.away_team_id]
        h2h = self._h2h[self._h2h_key(match.home_team_id, match.away_team_id)]

        home_team = db.get(m.Team, match.home_team_id)
        away_team = db.get(m.Team, match.away_team_id)
        comp = db.get(m.Competition, match.competition_id) if match.competition_id else None

        return MatchFeatures(
            match_id=match.id,
            home_team=home_team.name if home_team else "?",
            away_team=away_team.name if away_team else "?",
            competition=comp.name if comp else None,
            kickoff=match.kickoff,
            home_elo=home.elo,
            away_elo=away.elo,
            elo_diff=home.elo - away.elo,
            home_form=_mean(home.results, default=0.5),
            away_form=_mean(away.results, default=0.5),
            home_goals_scored_avg=_mean(home.goals_for, default=1.3),
            home_goals_conceded_avg=_mean(home.goals_against, default=1.3),
            away_goals_scored_avg=_mean(away.goals_for, default=1.1),
            away_goals_conceded_avg=_mean(away.goals_against, default=1.5),
            home_xg_for_avg=_mean(home.xg_for, default=1.3),
            home_xg_against_avg=_mean(home.xg_against, default=1.3),
            away_xg_for_avg=_mean(away.xg_for, default=1.1),
            away_xg_against_avg=_mean(away.xg_against, default=1.5),
            h2h_home_win_rate=(h2h.home_wins / h2h.n) if h2h.n else 0.45,
            h2h_draw_rate=(h2h.draws / h2h.n) if h2h.n else 0.25,
            h2h_goals_avg=(h2h.goals / h2h.n) if h2h.n else 2.6,
            home_rest_days=_rest_days(home.last_match_at, match.kickoff),
            away_rest_days=_rest_days(away.last_match_at, match.kickoff),
            home_shots_avg=_mean(home.shots, default=12.0),
            away_shots_avg=_mean(away.shots, default=10.0),
            home_corners_for_avg=_mean(home.corners_for, default=5.0),
            home_corners_against_avg=_mean(home.corners_against, default=4.5),
            away_corners_for_avg=_mean(away.corners_for, default=4.5),
            away_corners_against_avg=_mean(away.corners_against, default=5.0),
            home_cards_avg=_mean(home.cards, default=2.2),
            away_cards_avg=_mean(away.cards, default=2.3),
        )

    def _update_with_result(self, db: Session, match: m.Match) -> None:
        if match.home_score is None or match.away_score is None:
            return
        home = self._teams[match.home_team_id]
        away = self._teams[match.away_team_id]

        # Elo update
        exp_home = 1.0 / (1.0 + 10 ** (-(home.elo + self.HOME_ADVANTAGE - away.elo) / 400.0))
        if match.home_score > match.away_score:
            s_home = 1.0
        elif match.home_score == match.away_score:
            s_home = 0.5
        else:
            s_home = 0.0
        home.elo += self.K_FACTOR * (s_home - exp_home)
        away.elo += self.K_FACTOR * ((1 - s_home) - (1 - exp_home))

        home.results.append(s_home)
        away.results.append(1 - s_home)
        home.goals_for.append(match.home_score)
        home.goals_against.append(match.away_score)
        away.goals_for.append(match.away_score)
        away.goals_against.append(match.home_score)
        home.last_match_at = match.kickoff
        away.last_match_at = match.kickoff

        stats = db.query(m.MatchStat).filter_by(match_id=match.id).all()
        stat_by_team: dict[int, m.MatchStat] = {s.team_id: s for s in stats}
        home_stat = stat_by_team.get(match.home_team_id)
        away_stat = stat_by_team.get(match.away_team_id)
        if home_stat is not None:
            if home_stat.xg is not None:
                home.xg_for.append(home_stat.xg)
            if home_stat.xga is not None:
                home.xg_against.append(home_stat.xga)
            if home_stat.shots is not None:
                home.shots.append(home_stat.shots)
            if home_stat.corners is not None:
                home.corners_for.append(home_stat.corners)
            hy = home_stat.yellow_cards or 0
            hr = home_stat.red_cards or 0
            if home_stat.yellow_cards is not None or home_stat.red_cards is not None:
                home.cards.append(hy + hr)
        if away_stat is not None:
            if away_stat.xg is not None:
                away.xg_for.append(away_stat.xg)
            if away_stat.xga is not None:
                away.xg_against.append(away_stat.xga)
            if away_stat.shots is not None:
                away.shots.append(away_stat.shots)
            if away_stat.corners is not None:
                away.corners_for.append(away_stat.corners)
            ay = away_stat.yellow_cards or 0
            ar = away_stat.red_cards or 0
            if away_stat.yellow_cards is not None or away_stat.red_cards is not None:
                away.cards.append(ay + ar)
        # corners_against is independent of the team's own corners — record
        # it whenever the opponent's stat is available.
        if away_stat is not None and away_stat.corners is not None:
            home.corners_against.append(away_stat.corners)
        if home_stat is not None and home_stat.corners is not None:
            away.corners_against.append(home_stat.corners)

        key = self._h2h_key(match.home_team_id, match.away_team_id)
        h2h = self._h2h[key]
        h2h.n += 1
        h2h.goals += match.home_score + match.away_score
        if match.home_score > match.away_score:
            h2h.home_wins += 1
        elif match.home_score == match.away_score:
            h2h.draws += 1
        else:
            h2h.away_wins += 1

    @staticmethod
    def _h2h_key(a: int, b: int) -> tuple[int, int]:
        return (min(a, b), max(a, b))


def _mean(dq: Iterable[float], *, default: float) -> float:
    values = list(dq)
    if not values:
        return default
    return float(sum(values) / len(values))


def _rest_days(prev: datetime | None, now: datetime) -> float:
    if prev is None:
        return 7.0
    return max(0.0, (now - prev).total_seconds() / 86400.0)
