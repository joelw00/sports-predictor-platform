"""SHAP-based explanation for the 1X2 GBM classifier.

Produces per-feature attributions explaining why the model predicts the
observed probability split for ``home``/``draw``/``away``. The Poisson
sub-model is already white-box (team strengths + home advantage), so this
module focuses on the gradient-boosted component.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from app.features.football import MatchFeatures
from app.ml.gbm import FEATURE_COLUMNS, Gbm1X2Model

Outcome = Literal["home", "draw", "away"]


@dataclass
class FeatureContribution:
    feature: str
    value: float
    shap_value: float  # positive = pushes probability of `outcome` up


@dataclass
class OutcomeExplanation:
    outcome: Outcome
    base_probability: float
    model_probability: float
    top_positive: list[FeatureContribution]
    top_negative: list[FeatureContribution]


@dataclass
class MatchExplanation:
    match_id: int
    home_team: str
    away_team: str
    model_version: str
    outcomes: list[OutcomeExplanation]


_OUTCOME_ORDER: tuple[Outcome, ...] = ("home", "draw", "away")


def _features_to_row(f: MatchFeatures) -> dict[str, float]:
    return {
        "home_elo": f.home_elo,
        "away_elo": f.away_elo,
        "elo_diff": f.elo_diff,
        "home_form": f.home_form,
        "away_form": f.away_form,
        "home_goals_scored_avg": f.home_goals_scored_avg,
        "home_goals_conceded_avg": f.home_goals_conceded_avg,
        "away_goals_scored_avg": f.away_goals_scored_avg,
        "away_goals_conceded_avg": f.away_goals_conceded_avg,
        "home_xg_for_avg": f.home_xg_for_avg,
        "home_xg_against_avg": f.home_xg_against_avg,
        "away_xg_for_avg": f.away_xg_for_avg,
        "away_xg_against_avg": f.away_xg_against_avg,
        "h2h_home_win_rate": f.h2h_home_win_rate,
        "h2h_draw_rate": f.h2h_draw_rate,
        "h2h_goals_avg": f.h2h_goals_avg,
        "home_rest_days": f.home_rest_days,
        "away_rest_days": f.away_rest_days,
        "home_shots_avg": f.home_shots_avg,
        "away_shots_avg": f.away_shots_avg,
    }


def _shap_values(gbm: Gbm1X2Model, X: pd.DataFrame) -> np.ndarray:
    """Return SHAP values with shape ``(n_samples, n_features, n_classes)``.

    Uses LightGBM's native ``predict(..., pred_contrib=True)`` which is fast and
    exact for tree ensembles (equivalent to TreeSHAP).
    """
    assert gbm._booster is not None
    raw = np.asarray(gbm._booster.predict(X[FEATURE_COLUMNS], pred_contrib=True))
    n_features = len(FEATURE_COLUMNS)
    # LightGBM returns shape (n_samples, n_classes * (n_features + 1)) for
    # multi-class models: for each class there are n_features contributions
    # followed by one expected-value column.
    n_classes = raw.shape[1] // (n_features + 1)
    contribs = np.zeros((raw.shape[0], n_features, n_classes), dtype=float)
    for c in range(n_classes):
        start = c * (n_features + 1)
        contribs[:, :, c] = raw[:, start : start + n_features]
    return contribs


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def explain_gbm(
    gbm: Gbm1X2Model,
    feats: MatchFeatures,
    *,
    top_k: int = 5,
) -> list[OutcomeExplanation]:
    """Explain the GBM 1X2 output for a single match.

    Returns one :class:`OutcomeExplanation` per outcome (home / draw / away) with
    the top ``top_k`` positive and negative feature contributions.

    When the underlying LightGBM model is not fitted, returns a neutral
    explanation with empty contributor lists — callers can gracefully degrade.
    """
    if not gbm.fitted or gbm._booster is None:
        base = np.array([0.45, 0.25, 0.30])
        return [
            OutcomeExplanation(
                outcome=o,
                base_probability=float(base[i]),
                model_probability=float(base[i]),
                top_positive=[],
                top_negative=[],
            )
            for i, o in enumerate(_OUTCOME_ORDER)
        ]

    row = _features_to_row(feats)
    X = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    contribs = _shap_values(gbm, X)[0]  # shape (n_features, n_classes)

    probs = np.asarray(gbm.predict_proba(X))[0]

    # Reconstruct the base probability from the raw expected-value column.
    n_features = len(FEATURE_COLUMNS)
    raw = np.asarray(gbm._booster.predict(X[FEATURE_COLUMNS], pred_contrib=True))[0]
    n_classes = raw.shape[0] // (n_features + 1)
    base_logits = np.array([raw[(c + 1) * (n_features + 1) - 1] for c in range(n_classes)])
    base_probs = _softmax(base_logits)

    result: list[OutcomeExplanation] = []
    for i, outcome in enumerate(_OUTCOME_ORDER):
        entries = [
            FeatureContribution(
                feature=name,
                value=float(row[name]),
                shap_value=float(contribs[j, i]),
            )
            for j, name in enumerate(FEATURE_COLUMNS)
        ]
        positives = sorted([e for e in entries if e.shap_value > 0], key=lambda e: -e.shap_value)
        negatives = sorted([e for e in entries if e.shap_value < 0], key=lambda e: e.shap_value)
        result.append(
            OutcomeExplanation(
                outcome=outcome,
                base_probability=float(base_probs[i]),
                model_probability=float(probs[i]),
                top_positive=positives[:top_k],
                top_negative=negatives[:top_k],
            )
        )
    return result
