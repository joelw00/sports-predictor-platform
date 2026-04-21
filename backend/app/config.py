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
    sofascore_enabled: bool = Field(default=False)
    understat_enabled: bool = Field(default=False)
    the_odds_api_key: str = Field(default="")
    snai_enabled: bool = Field(default=False)

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
        """True when no real data source credentials are configured."""
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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
