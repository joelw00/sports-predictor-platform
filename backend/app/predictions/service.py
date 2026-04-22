"""Prediction service: orchestrates feature building, model inference and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import models as m
from app.features.football import FootballFeatureBuilder
from app.logging import get_logger
from app.ml.predictor import FootballPredictor
from app.value_bet.engine import ModelProbability, OddsQuote, ValueBetEngine

log = get_logger(__name__)


@dataclass
class ServiceStats:
    predictions_upserted: int = 0
    value_bets_upserted: int = 0


class PredictionService:
    def __init__(
        self,
        predictor: FootballPredictor | None = None,
        engine: ValueBetEngine | None = None,
    ) -> None:
        self.predictor = predictor or self._load_default_predictor()
        self.engine = engine or ValueBetEngine()

    @staticmethod
    def _load_default_predictor() -> FootballPredictor:
        path = FootballPredictor.default_artifact_path()
        if Path(path).exists():
            return FootballPredictor.load(path)
        return FootballPredictor()

    # ------------------------------------------------------------------
    # Run on a batch of upcoming matches
    # ------------------------------------------------------------------

    def refresh(self, db: Session, *, sport_code: str = "football") -> ServiceStats:
        stats = ServiceStats()
        builder = FootballFeatureBuilder()
        # Warm state from history.
        builder.load_history(db, sport_code=sport_code)

        stmt = (
            select(m.Match)
            .join(m.Sport, m.Match.sport_id == m.Sport.id)
            .where(m.Sport.code == sport_code, m.Match.status != "finished")
            .order_by(m.Match.kickoff.asc())
        )
        matches = list(db.scalars(stmt))
        for match in matches:
            feats = builder.snapshot(db, match)
            bundle = self.predictor.predict(feats)

            for market_prob in bundle.markets:
                row = (
                    db.query(m.Prediction)
                    .filter_by(
                        match_id=match.id,
                        market=market_prob.market,
                        selection=market_prob.selection,
                        line=market_prob.line,
                    )
                    .one_or_none()
                )
                if row is None:
                    row = m.Prediction(
                        match_id=match.id,
                        market=market_prob.market,
                        selection=market_prob.selection,
                        line=market_prob.line,
                    )
                    db.add(row)
                row.probability = float(market_prob.probability)
                row.confidence = float(bundle.confidence)
                row.model_version = bundle.model_version
                row.drivers = bundle.drivers
                stats.predictions_upserted += 1

            # Value bets
            odds_rows = db.query(m.OddsSnapshot).filter(m.OddsSnapshot.match_id == match.id).all()
            probs = [
                ModelProbability(p.market, p.selection, p.line, p.probability)
                for p in bundle.markets
            ]
            quotes = [
                OddsQuote(o.bookmaker, o.market, o.selection, o.line, o.price) for o in odds_rows
            ]
            raw_bets = self.engine.evaluate(
                match_id=match.id,
                model_probs=probs,
                odds=quotes,
                confidence=bundle.confidence,
            )
            filtered = self.engine.filter_and_rank(raw_bets)
            # Clear previous value bets for this match and upsert the filtered ones.
            db.query(m.ValueBet).filter(m.ValueBet.match_id == match.id).delete()
            for vb in filtered:
                db.add(
                    m.ValueBet(
                        match_id=vb.match_id,
                        market=vb.market,
                        selection=vb.selection,
                        line=vb.line,
                        bookmaker=vb.bookmaker,
                        price=vb.price,
                        p_model=vb.p_model,
                        p_fair=vb.p_fair,
                        edge=vb.edge,
                        expected_value=vb.expected_value,
                        kelly_fraction=vb.kelly_fraction,
                        confidence=vb.confidence,
                        rationale=vb.rationale,
                    )
                )
                stats.value_bets_upserted += 1

        db.commit()
        log.info(
            "predictions.refresh.done",
            predictions=stats.predictions_upserted,
            value_bets=stats.value_bets_upserted,
            matches=len(matches),
        )
        return stats


def utcnow() -> datetime:
    return datetime.now(tz=UTC)
