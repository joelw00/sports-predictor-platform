from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class SportOut(ORMModel):
    id: int
    code: str
    name: str


class TeamOut(ORMModel):
    id: int
    name: str
    short_name: str | None = None
    country: str | None = None


class CompetitionOut(ORMModel):
    id: int
    code: str
    name: str
    country: str | None = None
    tier: int | None = None


class OddsQuote(ORMModel):
    bookmaker: str
    market: str
    selection: str
    line: float | None = None
    price: float
    captured_at: datetime
    is_closing: bool = False
    is_live: bool = False


class PredictionOut(ORMModel):
    market: str
    selection: str
    line: float | None = None
    probability: float
    confidence: float
    model_version: str
    drivers: dict = Field(default_factory=dict)


class ValueBetOut(ORMModel):
    match_id: int
    market: str
    selection: str
    line: float | None = None
    bookmaker: str
    price: float
    p_model: float
    p_fair: float
    edge: float
    expected_value: float
    kelly_fraction: float
    confidence: float
    rationale: dict = Field(default_factory=dict)
    # match context (hydrated by API):
    home_team: str | None = None
    away_team: str | None = None
    competition: str | None = None
    kickoff: datetime | None = None
    sport: str | None = None


class Event(ORMModel):
    id: int
    sport: str
    competition: str | None = None
    home_team: str
    away_team: str
    kickoff: datetime
    status: str
    home_score: int | None = None
    away_score: int | None = None
    top_prediction: PredictionOut | None = None
    best_value: ValueBetOut | None = None


class EventList(BaseModel):
    items: list[Event]
    total: int


class MatchDetail(ORMModel):
    id: int
    sport: str
    competition: str | None = None
    home_team: TeamOut
    away_team: TeamOut
    kickoff: datetime
    status: str
    home_score: int | None = None
    away_score: int | None = None
    predictions: list[PredictionOut]
    odds: list[OddsQuote]
    value_bets: list[ValueBetOut]
    form: dict = Field(default_factory=dict)


class FeatureContributionOut(BaseModel):
    feature: str
    value: float
    shap_value: float


class OutcomeExplanationOut(BaseModel):
    outcome: str
    base_probability: float
    model_probability: float
    top_positive: list[FeatureContributionOut]
    top_negative: list[FeatureContributionOut]


class MatchExplanationOut(BaseModel):
    match_id: int
    home_team: str
    away_team: str
    model_version: str
    outcomes: list[OutcomeExplanationOut]


class RiskPolicyOut(ORMModel):
    id: int
    name: str
    bankroll: float
    kelly_fraction: float
    max_stake_pct: float
    max_daily_exposure_pct: float
    max_concurrent_positions: int
    stop_loss_drawdown_pct: float
    min_edge: float
    min_confidence: float
    enabled: bool


class RiskPolicyUpdate(BaseModel):
    bankroll: float = Field(gt=0)
    kelly_fraction: float = Field(ge=0.0, le=1.0)
    max_stake_pct: float = Field(ge=0.0, le=1.0)
    max_daily_exposure_pct: float = Field(ge=0.0, le=1.0)
    max_concurrent_positions: int = Field(ge=0)
    stop_loss_drawdown_pct: float = Field(ge=0.0, le=1.0)
    min_edge: float = Field(ge=0.0)
    min_confidence: float = Field(ge=0.0, le=1.0)
    enabled: bool = True


class RiskDecisionOut(BaseModel):
    match_id: int
    market: str
    selection: str
    line: float | None = None
    bookmaker: str
    edge: float
    confidence: float
    kelly_fraction: float
    recommended_stake: float
    accepted: bool
    reasons: list[str]


class BacktestResult(ORMModel):
    id: int
    label: str
    sport_code: str
    market: str
    start_date: datetime
    end_date: datetime
    strategy: str
    min_edge: float
    stake: float
    n_bets: int
    n_wins: int
    total_staked: float
    total_return: float
    roi: float
    yield_pct: float
    max_drawdown: float
    profit_factor: float
    avg_clv: float = 0.0
    clv_win_rate: float = 0.0
    n_clv_tracked: int = 0
    equity_curve: list
    breakdown: dict
