from __future__ import annotations

from app.config import get_settings
from app.ingestion.base import BaseSource
from app.ingestion.demo import DemoSource
from app.ingestion.football_data_org import FootballDataOrgSource
from app.ingestion.stubs import (
    ApiFootballSource,
    SnaiSource,
    SofaScoreSource,
    UnderstatSource,
)
from app.ingestion.the_odds_api import TheOddsApiRealSource


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
        TheOddsApiRealSource(
            settings.the_odds_api_key,
            sport_keys=settings.the_odds_api_sport_key_list or None,
            regions=settings.the_odds_api_region_list or None,
            markets=settings.the_odds_api_market_list or None,
        ),
        SnaiSource(settings.snai_enabled),
    ]
    active = [s for s in candidates if s.is_enabled()]
    if not active:
        # Demo source is the fallback when no real credential is configured.
        active.append(DemoSource())
    return active
