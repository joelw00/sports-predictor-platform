from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.backtesting.engine import Backtester, EntryStrategy
from app.backtesting.walk_forward import WalkForwardBacktester
from app.db import get_db
from app.db import models as m
from app.ml.predictor import FootballPredictor
from app.schemas import BacktestResult
from app.value_bet.engine import ValueBetEngine

router = APIRouter(tags=["backtests"])


class BacktestRequest(BaseModel):
    label: str = "adhoc"
    sport: str = "football"
    market: str = "1x2"
    strategy: str = Field(default="flat", pattern="^(flat|kelly)$")
    stake: float = 1.0
    min_edge: float = 0.03
    start_date: datetime | None = None
    end_date: datetime | None = None
    mode: str = Field(
        default="walk_forward",
        pattern="^(walk_forward|pretrained)$",
        description=(
            "walk_forward = honest out-of-sample evaluation (retrains per fold). "
            "pretrained = legacy replay with the already-trained predictor (leaky; "
            "kept only for diagnostic comparison)."
        ),
    )
    entry_odds_strategy: str = Field(
        default="closing",
        pattern="^(closing|opening)$",
        description=(
            "closing = bet at the closing line (legacy; CLV always 0). "
            "opening = bet at the earliest pre-kickoff snapshot and report "
            "CLV against the closing line."
        ),
    )
    n_folds: int = 6
    min_train_folds: int = 2


@router.get("/backtests", response_model=list[BacktestResult])
def list_backtests(db: Session = Depends(get_db)) -> list[BacktestResult]:
    rows = db.query(m.BacktestRun).order_by(m.BacktestRun.created_at.desc()).limit(50).all()
    return [BacktestResult.model_validate(r) for r in rows]


@router.get("/backtests/{backtest_id}", response_model=BacktestResult)
def get_backtest(backtest_id: int, db: Session = Depends(get_db)) -> BacktestResult:
    row = db.get(m.BacktestRun, backtest_id)
    if row is None:
        raise HTTPException(status_code=404, detail="backtest not found")
    return BacktestResult.model_validate(row)


@router.post("/backtests/run", response_model=BacktestResult)
def run_backtest(
    request: Annotated[BacktestRequest, Body()],
    db: Session = Depends(get_db),
) -> BacktestResult:
    engine = ValueBetEngine(min_edge=request.min_edge)
    entry_strategy: EntryStrategy = (
        "opening" if request.entry_odds_strategy == "opening" else "closing"
    )
    if request.mode == "walk_forward":
        bt_wf = WalkForwardBacktester(
            n_folds=request.n_folds,
            min_train_folds=request.min_train_folds,
            engine=engine,
            stake=request.stake,
            strategy=request.strategy,
            entry_odds_strategy=entry_strategy,
        )
        result = bt_wf.run(
            db,
            sport_code=request.sport,
            market=request.market,
            start_date=request.start_date,
            end_date=request.end_date,
            label=request.label,
        )
    else:
        try:
            predictor = FootballPredictor.load(FootballPredictor.default_artifact_path())
        except Exception:
            predictor = FootballPredictor()
        bt = Backtester(
            predictor,
            engine=engine,
            stake=request.stake,
            strategy=request.strategy,
            entry_odds_strategy=entry_strategy,
        )
        result = bt.run(
            db,
            sport_code=request.sport,
            market=request.market,
            start_date=request.start_date,
            end_date=request.end_date,
            label=request.label,
        )
    row = m.BacktestRun(
        label=result.label,
        sport_code=result.sport_code,
        market=result.market,
        start_date=result.start_date,
        end_date=result.end_date,
        strategy=result.strategy,
        min_edge=result.min_edge,
        stake=result.stake,
        n_bets=result.n_bets,
        n_wins=result.n_wins,
        total_staked=result.total_staked,
        total_return=result.total_return,
        roi=result.roi,
        yield_pct=result.yield_pct,
        max_drawdown=result.max_drawdown,
        profit_factor=result.profit_factor,
        avg_clv=result.avg_clv,
        clv_win_rate=result.clv_win_rate,
        n_clv_tracked=result.n_clv_tracked,
        equity_curve=result.equity_curve,
        breakdown=result.breakdown,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return BacktestResult.model_validate(row)
