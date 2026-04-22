from __future__ import annotations

from app.config import get_settings
from app.ingestion.base import BaseSource
from app.ingestion.demo import DemoSource
from app.ingestion.football_data_org import FootballDataOrgSource
from app.ingestion.stubs import (
    ApiFootballSource,
    SnaiSource,
    SofaScoreSource,
    TheOddsApiSource,
    UnderstatSource,
)


def get_active_sources() -> list[BaseSource]:
    settings = get_settings()
    if settings.force_demo_mode:
        return [DemoSource()]
    candidates: list[BaseSource] = [
        ApiFootballSource(settings.api_football_key),
        FootballDataOrgSource(
            settings.football_data_key,
            competitions=settings.football_data_competition_list,
            season=settings.football_data_season or None,
        ),
        SofaScoreSource(settings.sofascore_enabled),
        UnderstatSource(settings.understat_enabled),
        TheOddsApiSource(settings.the_odds_api_key),
        SnaiSource(settings.snai_enabled),
    ]
    active = [s for s in candidates if s.is_enabled()]
    if not active:
        # Demo source is the fallback when no real credential is configured.
        active.append(DemoSource())
    return active
