from __future__ import annotations

from app.backtesting.engine import Backtester
from app.db import models as m
from app.ingestion.demo import DemoSource
from app.ingestion.orchestrator import ingest_all
from app.ml.predictor import FootballPredictor
from app.ml.trainer import FootballTrainer
from app.value_bet.engine import ValueBetEngine


def test_backtest_runs_on_demo_data(db_session):
    ingest_all(db_session, sources=[DemoSource(seed=11)])
    predictor, _report = FootballTrainer(version="t").train(db_session)
    bt = Backtester(predictor, engine=ValueBetEngine(min_edge=0.0, min_confidence=0.0))
    result = bt.run(db_session, sport_code="football", market="1x2", label="unit")
    assert result.n_bets >= 0
    assert result.start_date <= result.end_date
    # Equity curve length equals number of bets taken.
    assert len(result.equity_curve) == result.n_bets


def test_predictor_saves_and_loads(tmp_path, db_session):
    ingest_all(db_session, sources=[DemoSource(seed=13)])
    predictor, _ = FootballTrainer(version="t").train(db_session)
    path = tmp_path / "pred.joblib"
    predictor.save(path)
    loaded = FootballPredictor.load(path)
    assert loaded.version == "t"
    # Run one prediction on a match to confirm the loaded predictor still works.
    match = db_session.query(m.Match).filter_by(status="scheduled").first()
    if match is None:
        return
    from app.features.football import FootballFeatureBuilder

    builder = FootballFeatureBuilder()
    feats = builder.snapshot(db_session, match)
    bundle = loaded.predict(feats)
    probs = [p.probability for p in bundle.markets if p.market == "1x2"]
    assert abs(sum(probs) - 1.0) < 1e-3
