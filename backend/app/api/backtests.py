from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.backtesting.engine import Backtester
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
    try:
        predictor = FootballPredictor.load(FootballPredictor.default_artifact_path())
    except Exception:
        predictor = FootballPredictor()
    engine = ValueBetEngine(min_edge=request.min_edge)
    bt = Backtester(predictor, engine=engine, stake=request.stake, strategy=request.strategy)
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
        equity_curve=result.equity_curve,
        breakdown=result.breakdown,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return BacktestResult.model_validate(row)
