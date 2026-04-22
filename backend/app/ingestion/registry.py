from __future__ import annotations

from app.config import get_settings
from app.ingestion.base import BaseSource
from app.ingestion.demo import DemoSource
from app.ingestion.stubs import (
    ApiFootballSource,
    FootballDataSource,
    SnaiSource,
    SofaScoreSource,
    TheOddsApiSource,
    UnderstatSource,
)


def get_active_sources() -> list[BaseSource]:
    settings = get_settings()
    candidates: list[BaseSource] = [
        ApiFootballSource(settings.api_football_key),
        FootballDataSource(settings.football_data_key),
        SofaScoreSource(settings.sofascore_enabled),
        UnderstatSource(settings.understat_enabled),
        TheOddsApiSource(settings.the_odds_api_key),
        SnaiSource(settings.snai_enabled),
    ]
    active = [s for s in candidates if s.is_enabled()]
    # Demo source is always available as a fallback.
    active.append(DemoSource())
    return active
