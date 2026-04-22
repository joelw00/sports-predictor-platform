from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class RawMatch:
    external_id: str
    sport_code: str
    competition_code: str
    competition_name: str
    home_team: str
    away_team: str
    kickoff: datetime
    status: str = "scheduled"
    home_score: int | None = None
    away_score: int | None = None
    season: str | None = None
    stage: str | None = None
    venue: str | None = None


@dataclass
class RawStat:
    match_external_id: str
    team_name: str
    shots: int | None = None
    shots_on_target: int | None = None
    corners: int | None = None
    yellow_cards: int | None = None
    red_cards: int | None = None
    fouls: int | None = None
    possession: float | None = None
    xg: float | None = None
    xga: float | None = None
    passes: int | None = None
    pass_accuracy: float | None = None


@dataclass
class RawOdds:
    match_external_id: str
    bookmaker: str
    market: str
    selection: str
    price: float
    captured_at: datetime
    line: float | None = None
    is_closing: bool = False
    is_live: bool = False


@dataclass
class IngestionResult:
    matches: list[RawMatch] = field(default_factory=list)
    stats: list[RawStat] = field(default_factory=list)
    odds: list[RawOdds] = field(default_factory=list)
    source: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)


class BaseSource(ABC):
    """Abstract adapter that every data source must implement."""

    #: human-readable name (e.g. "api-football")
    name: str = "base"

    #: sports this source can provide
    sports: tuple[str, ...] = ("football",)

    @abstractmethod
    def is_enabled(self) -> bool: ...

    @abstractmethod
    def fetch(self, *, season: str | None = None) -> IngestionResult:
        """Return a snapshot of matches / stats / odds for the requested window."""
