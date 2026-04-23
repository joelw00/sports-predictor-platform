"""Explainability endpoints — SHAP contributions for the 1X2 GBM model."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.db import models as m
from app.features.football import FootballFeatureBuilder
from app.ml.explain import explain_gbm
from app.predictions.service import PredictionService
from app.schemas import (
    FeatureContributionOut,
    MatchExplanationOut,
    OutcomeExplanationOut,
)

router = APIRouter(tags=["explain"])


@router.get("/events/{match_id}/explain", response_model=MatchExplanationOut)
def explain_match(
    match_id: int,
    top_k: int = 5,
    db: Session = Depends(get_db),
) -> MatchExplanationOut:
    match = db.get(m.Match, match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="match not found")

    home = db.get(m.Team, match.home_team_id)
    away = db.get(m.Team, match.away_team_id)
    if home is None or away is None:
        raise HTTPException(status_code=400, detail="match has no teams")

    sport = db.get(m.Sport, match.sport_id)
    sport_code = sport.code if sport else "football"

    builder = FootballFeatureBuilder()
    builder.load_history(db, sport_code=sport_code)
    feats = builder.snapshot(db, match)

    predictor = PredictionService._load_active_predictor()
    outcomes_raw = explain_gbm(predictor.gbm, feats, top_k=top_k)

    outcomes: list[OutcomeExplanationOut] = []
    for o in outcomes_raw:
        outcomes.append(
            OutcomeExplanationOut(
                outcome=o.outcome,
                base_probability=o.base_probability,
                model_probability=o.model_probability,
                top_positive=[
                    FeatureContributionOut(
                        feature=c.feature, value=c.value, shap_value=c.shap_value
                    )
                    for c in o.top_positive
                ],
                top_negative=[
                    FeatureContributionOut(
                        feature=c.feature, value=c.value, shap_value=c.shap_value
                    )
                    for c in o.top_negative
                ],
            )
        )

    return MatchExplanationOut(
        match_id=match.id,
        home_team=home.name,
        away_team=away.name,
        model_version=predictor.version,
        outcomes=outcomes,
    )
