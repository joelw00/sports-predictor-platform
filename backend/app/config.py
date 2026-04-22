from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration, populated from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")
    secret_key: str = Field(default="change-me-in-production")

    database_url: str = Field(
        default="postgresql+psycopg://sports:sports@localhost:5432/sports",
    )
    redis_url: str = Field(default="redis://localhost:6379/0")

    # Data sources
    api_football_key: str = Field(default="")
    football_data_key: str = Field(default="")
    football_data_competitions: str = Field(
        default="PL,SA,BL1,FL1,PD,DED,PPL",
        description="Comma-separated Football-Data.org competition codes to ingest.",
    )
    football_data_season: str = Field(
        default="",
        description="Starting year of the season to ingest (e.g. '2024'). Empty = current.",
    )
    sofascore_enabled: bool = Field(default=False)
    understat_enabled: bool = Field(default=False)
    the_odds_api_key: str = Field(default="")
    the_odds_api_sport_keys: str = Field(
        default=(
            "soccer_epl,soccer_italy_serie_a,soccer_germany_bundesliga,"
            "soccer_spain_la_liga,soccer_france_ligue_one,soccer_uefa_champs_league"
        ),
        description=(
            "Comma-separated The Odds API sport keys to ingest. "
            "See https://the-odds-api.com/sports-odds-data/sports-apis.html"
        ),
    )
    the_odds_api_regions: str = Field(
        default="eu,uk",
        description="Comma-separated The Odds API regions (eu, uk, us, au).",
    )
    the_odds_api_markets: str = Field(
        default="h2h,totals,spreads",
        description="Comma-separated The Odds API markets (h2h, totals, spreads).",
    )
    snai_enabled: bool = Field(default=False)

    # Force demo data even when credentials are present (useful for CI / tests).
    force_demo_mode: bool = Field(default=False)

    # Scheduler
    scheduler_enabled: bool = Field(default=False)
    scheduler_cron_hour: int = Field(default=3)
    scheduler_cron_minute: int = Field(default=0)

    # Retraining
    retrain_enabled: bool = Field(
        default=False,
        description="When true the scheduler also runs a weekly retrain job.",
    )
    retrain_cron_day_of_week: str = Field(
        default="mon",
        description="APScheduler day-of-week token for the weekly retrain job.",
    )
    retrain_cron_hour: int = Field(default=4)
    retrain_cron_minute: int = Field(default=0)

    # Value bet engine
    value_bet_min_edge: float = Field(default=0.03)
    value_bet_min_confidence: float = Field(default=0.55)

    # Storage
    artifacts_dir: str = Field(default="models/artifacts")

    # CORS
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
        ]
    )

    @property
    def is_demo_mode(self) -> bool:
        """True when no real data source credentials are configured, or when the
        operator forced demo mode."""
        if self.force_demo_mode:
            return True
        return not any(
            [
                self.api_football_key,
                self.football_data_key,
                self.sofascore_enabled,
                self.understat_enabled,
                self.the_odds_api_key,
                self.snai_enabled,
            ]
        )

    @property
    def football_data_competition_list(self) -> tuple[str, ...]:
        codes = [c.strip().upper() for c in self.football_data_competitions.split(",")]
        return tuple(c for c in codes if c)

    @property
    def the_odds_api_sport_key_list(self) -> tuple[str, ...]:
        keys = [k.strip() for k in self.the_odds_api_sport_keys.split(",")]
        return tuple(k for k in keys if k)

    @property
    def the_odds_api_region_list(self) -> tuple[str, ...]:
        regions = [r.strip().lower() for r in self.the_odds_api_regions.split(",")]
        return tuple(r for r in regions if r)

    @property
    def the_odds_api_market_list(self) -> tuple[str, ...]:
        markets = [mkt.strip().lower() for mkt in self.the_odds_api_markets.split(",")]
        return tuple(mkt for mkt in markets if mkt)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
