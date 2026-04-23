"""SQLAlchemy ORM models covering reference data, events, odds, predictions and value bets."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

# ---------------------------------------------------------------------------
# Reference tables
# ---------------------------------------------------------------------------


class Sport(Base, TimestampMixin):
    __tablename__ = "sports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(64), nullable=False)

    competitions: Mapped[list[Competition]] = relationship(back_populates="sport")


class Competition(Base, TimestampMixin):
    __tablename__ = "competitions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sport_id: Mapped[int] = mapped_column(ForeignKey("sports.id"), nullable=False)
    code: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    country: Mapped[str | None] = mapped_column(String(64))
    tier: Mapped[int | None] = mapped_column(Integer)

    sport: Mapped[Sport] = relationship(back_populates="competitions")
    teams: Mapped[list[Team]] = relationship(back_populates="competition")

    __table_args__ = (UniqueConstraint("sport_id", "code", name="uq_competition_sport_code"),)


class Team(Base, TimestampMixin):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    competition_id: Mapped[int | None] = mapped_column(ForeignKey("competitions.id"))
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    short_name: Mapped[str | None] = mapped_column(String(32))
    country: Mapped[str | None] = mapped_column(String(64))

    competition: Mapped[Competition | None] = relationship(back_populates="teams")


class Player(Base, TimestampMixin):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"))
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    position: Mapped[str | None] = mapped_column(String(16))
    rating: Mapped[float | None] = mapped_column(Float)


# ---------------------------------------------------------------------------
# Events (matches)
# ---------------------------------------------------------------------------


class Match(Base, TimestampMixin):
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sport_id: Mapped[int] = mapped_column(ForeignKey("sports.id"), nullable=False)
    competition_id: Mapped[int | None] = mapped_column(ForeignKey("competitions.id"))
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    kickoff: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="scheduled", nullable=False)
    home_score: Mapped[int | None] = mapped_column(Integer)
    away_score: Mapped[int | None] = mapped_column(Integer)
    season: Mapped[str | None] = mapped_column(String(16))
    stage: Mapped[str | None] = mapped_column(String(64))
    venue: Mapped[str | None] = mapped_column(String(128))

    home_team: Mapped[Team] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped[Team] = relationship(foreign_keys=[away_team_id])
    sport: Mapped[Sport] = relationship()
    competition: Mapped[Competition | None] = relationship()
    stats: Mapped[list[MatchStat]] = relationship(back_populates="match", cascade="all,delete")
    odds: Mapped[list[OddsSnapshot]] = relationship(back_populates="match", cascade="all,delete")
    predictions: Mapped[list[Prediction]] = relationship(
        back_populates="match", cascade="all,delete"
    )


class MatchStat(Base, TimestampMixin):
    __tablename__ = "match_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    shots: Mapped[int | None] = mapped_column(Integer)
    shots_on_target: Mapped[int | None] = mapped_column(Integer)
    corners: Mapped[int | None] = mapped_column(Integer)
    yellow_cards: Mapped[int | None] = mapped_column(Integer)
    red_cards: Mapped[int | None] = mapped_column(Integer)
    fouls: Mapped[int | None] = mapped_column(Integer)
    possession: Mapped[float | None] = mapped_column(Float)
    xg: Mapped[float | None] = mapped_column(Float)
    xga: Mapped[float | None] = mapped_column(Float)
    passes: Mapped[int | None] = mapped_column(Integer)
    pass_accuracy: Mapped[float | None] = mapped_column(Float)

    match: Mapped[Match] = relationship(back_populates="stats")
    team: Mapped[Team] = relationship()


# ---------------------------------------------------------------------------
# Odds
# ---------------------------------------------------------------------------


class OddsSnapshot(Base, TimestampMixin):
    __tablename__ = "odds_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False)
    bookmaker: Mapped[str] = mapped_column(String(64), nullable=False)
    market: Mapped[str] = mapped_column(String(32), nullable=False)
    selection: Mapped[str] = mapped_column(String(64), nullable=False)
    line: Mapped[float | None] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_closing: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_live: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    match: Mapped[Match] = relationship(back_populates="odds")


# ---------------------------------------------------------------------------
# ML / predictions / value bets
# ---------------------------------------------------------------------------


class ModelRegistry(Base, TimestampMixin):
    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sport_code: Mapped[str] = mapped_column(String(32), nullable=False)
    market: Mapped[str] = mapped_column(String(32), nullable=False)
    family: Mapped[str] = mapped_column(String(32), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    artifact_path: Mapped[str] = mapped_column(String(512), nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    __table_args__ = (
        UniqueConstraint("sport_code", "market", "version", name="uq_model_unique_version"),
    )


class Prediction(Base, TimestampMixin):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False)
    market: Mapped[str] = mapped_column(String(32), nullable=False)
    selection: Mapped[str] = mapped_column(String(64), nullable=False)
    line: Mapped[float | None] = mapped_column(Float)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    drivers: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    match: Mapped[Match] = relationship(back_populates="predictions")

    __table_args__ = (
        UniqueConstraint(
            "match_id", "market", "selection", "line", name="uq_pred_match_market_selection"
        ),
    )


class ValueBet(Base, TimestampMixin):
    __tablename__ = "value_bets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False)
    market: Mapped[str] = mapped_column(String(32), nullable=False)
    selection: Mapped[str] = mapped_column(String(64), nullable=False)
    line: Mapped[float | None] = mapped_column(Float)
    bookmaker: Mapped[str] = mapped_column(String(64), nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    p_model: Mapped[float] = mapped_column(Float, nullable=False)
    p_fair: Mapped[float] = mapped_column(Float, nullable=False)
    edge: Mapped[float] = mapped_column(Float, nullable=False)
    expected_value: Mapped[float] = mapped_column(Float, nullable=False)
    kelly_fraction: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    rationale: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    match: Mapped[Match] = relationship()


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------


class BacktestRun(Base, TimestampMixin):
    __tablename__ = "backtest_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    label: Mapped[str] = mapped_column(String(128), nullable=False)
    sport_code: Mapped[str] = mapped_column(String(32), nullable=False)
    market: Mapped[str] = mapped_column(String(32), nullable=False)
    start_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    strategy: Mapped[str] = mapped_column(String(32), default="flat", nullable=False)
    min_edge: Mapped[float] = mapped_column(Float, default=0.03, nullable=False)
    stake: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    n_bets: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    n_wins: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_staked: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    total_return: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    roi: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    yield_pct: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    max_drawdown: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    profit_factor: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    avg_clv: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    clv_win_rate: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    n_clv_tracked: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    equity_curve: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    breakdown: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)


# ---------------------------------------------------------------------------
# Raw ingestion payloads (audit trail for external API responses)
# ---------------------------------------------------------------------------


class IngestionPayload(Base, TimestampMixin):
    """Unmodified JSON response stored per (source, endpoint) call.

    Serves as an audit trail / reprocessing buffer so we can rebuild the
    ``matches`` / ``match_stats`` / ``odds_snapshots`` tables without hitting
    the external API again.
    """

    __tablename__ = "ingestion_payloads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    endpoint: Mapped[str] = mapped_column(String(256), nullable=False)
    params: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    status_code: Mapped[int | None] = mapped_column(Integer)
    payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ok: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    error: Mapped[str | None] = mapped_column(String(512))


class IngestionRun(Base, TimestampMixin):
    """Summary row per ingestion invocation (scheduled or manual).

    Useful for monitoring: how many rows were touched, which sources succeeded,
    what the payload IDs are that backed this run.
    """

    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    trigger: Mapped[str] = mapped_column(String(32), default="manual", nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    matches_upserted: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    stats_upserted: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    odds_upserted: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    ok: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    error: Mapped[str | None] = mapped_column(String(1024))
    meta: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)


# ---------------------------------------------------------------------------
# Monitoring snapshots (live calibration, drift, alerts)
# ---------------------------------------------------------------------------


class RiskPolicy(Base, TimestampMixin):
    """Bankroll / risk policy applied to the value-bet engine and backtester.

    There is always a single active ``"default"`` policy. UI lets the operator
    tweak bankroll, fractional-Kelly cap, per-bet and per-day exposure limits,
    max concurrent positions, and a drawdown-based stop-loss.
    """

    __tablename__ = "risk_policies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, default="default")
    bankroll: Mapped[float] = mapped_column(Float, nullable=False, default=1000.0)
    kelly_fraction: Mapped[float] = mapped_column(Float, nullable=False, default=0.25)
    max_stake_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.02)
    max_daily_exposure_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.10)
    max_concurrent_positions: Mapped[int] = mapped_column(Integer, nullable=False, default=10)
    stop_loss_drawdown_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.20)
    min_edge: Mapped[float] = mapped_column(Float, nullable=False, default=0.03)
    min_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.55)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class MonitoringSnapshot(Base, TimestampMixin):
    """Periodic monitoring pass. One row per run per (sport, market).

    Captures the live vs training Brier gap, feature drift (PSI + KS), and any
    alerts fired (low data / high variance / stale model / calibration drift).
    Daily cadence is enough for a football model — we mostly want to detect
    when the upstream feed changes shape or the model degrades.
    """

    __tablename__ = "monitoring_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sport_code: Mapped[str] = mapped_column(String(32), nullable=False)
    market: Mapped[str] = mapped_column(String(32), nullable=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(64))
    n_recent_finished: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    n_predictions_evaluated: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    brier_live: Mapped[float | None] = mapped_column(Float)
    log_loss_live: Mapped[float | None] = mapped_column(Float)
    accuracy_live: Mapped[float | None] = mapped_column(Float)
    brier_training: Mapped[float | None] = mapped_column(Float)
    max_psi: Mapped[float | None] = mapped_column(Float)
    drift: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    alerts: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    meta: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
