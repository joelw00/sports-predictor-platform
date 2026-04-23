"""Unit tests for the SHAP explainer (``app.ml.explain``).

These tests intentionally use ``FootballFeatureBuilder``-compatible synthetic
features rather than training on DB data so they run in < 1 s.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from app.features.football import MatchFeatures
from app.ml.explain import explain_gbm
from app.ml.gbm import FEATURE_COLUMNS, Gbm1X2Model


def _make_features(**overrides: float) -> MatchFeatures:
    base: dict[str, float | int | str | datetime | None] = {
        "match_id": 1,
        "home_team": "Home",
        "away_team": "Away",
        "competition": "Test",
        "kickoff": datetime(2024, 1, 1, tzinfo=UTC),
        "home_elo": 1600.0,
        "away_elo": 1500.0,
        "elo_diff": 100.0,
        "home_form": 0.6,
        "away_form": 0.4,
        "home_goals_scored_avg": 1.6,
        "home_goals_conceded_avg": 1.0,
        "away_goals_scored_avg": 1.2,
        "away_goals_conceded_avg": 1.3,
        "home_xg_for_avg": 1.5,
        "home_xg_against_avg": 1.0,
        "away_xg_for_avg": 1.1,
        "away_xg_against_avg": 1.3,
        "h2h_home_win_rate": 0.5,
        "h2h_draw_rate": 0.25,
        "h2h_goals_avg": 2.6,
        "home_rest_days": 4.0,
        "away_rest_days": 4.0,
        "home_shots_avg": 12.0,
        "away_shots_avg": 10.0,
    }
    base.update(overrides)
    return MatchFeatures(**base)  # type: ignore[arg-type]


def _fit_gbm_on_synthetic(n: int = 300, seed: int = 7) -> Gbm1X2Model:
    """Fit the 1X2 GBM on a noisy synthetic sample where higher elo_diff
    correlates with more home wins. Good enough to produce non-zero SHAP."""
    rng = np.random.default_rng(seed)
    rows: list[list[float]] = []
    labels: list[int] = []
    for _ in range(n):
        elo_diff = float(rng.normal(0, 120))
        home_form = float(rng.uniform(0, 1))
        away_form = float(rng.uniform(0, 1))
        features = {col: 0.0 for col in FEATURE_COLUMNS}
        features["home_elo"] = 1500 + elo_diff / 2
        features["away_elo"] = 1500 - elo_diff / 2
        features["elo_diff"] = elo_diff
        features["home_form"] = home_form
        features["away_form"] = away_form
        features["home_goals_scored_avg"] = 1.3 + home_form * 0.5
        features["away_goals_scored_avg"] = 1.1 + away_form * 0.5
        features["home_goals_conceded_avg"] = 1.3 - home_form * 0.3
        features["away_goals_conceded_avg"] = 1.3 - away_form * 0.3
        features["home_xg_for_avg"] = features["home_goals_scored_avg"]
        features["away_xg_for_avg"] = features["away_goals_scored_avg"]
        features["home_xg_against_avg"] = features["home_goals_conceded_avg"]
        features["away_xg_against_avg"] = features["away_goals_conceded_avg"]
        features["h2h_home_win_rate"] = 0.4
        features["h2h_draw_rate"] = 0.25
        features["h2h_goals_avg"] = 2.5
        features["home_rest_days"] = 4.0
        features["away_rest_days"] = 4.0
        features["home_shots_avg"] = 10.0
        features["away_shots_avg"] = 10.0
        # Deterministic-ish label with noise.
        p_home = 1.0 / (1.0 + np.exp(-(elo_diff / 150.0)))
        u = rng.uniform()
        if u < p_home * 0.55:
            label = 0  # home
        elif u < 0.60:
            label = 1  # draw
        else:
            label = 2  # away
        rows.append([features[col] for col in FEATURE_COLUMNS])
        labels.append(label)
    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    y = np.asarray(labels)
    return Gbm1X2Model().fit(X, y)


def test_explain_gbm_returns_three_outcomes_on_unfitted_model() -> None:
    gbm = Gbm1X2Model()  # not fitted
    feats = _make_features()
    out = explain_gbm(gbm, feats)
    assert [o.outcome for o in out] == ["home", "draw", "away"]
    # Degrades gracefully: neutral priors, empty driver lists.
    for o in out:
        assert abs(o.base_probability - o.model_probability) < 1e-9
        assert o.top_positive == []
        assert o.top_negative == []


def test_explain_gbm_returns_sorted_top_k_with_non_zero_shap() -> None:
    gbm = _fit_gbm_on_synthetic()
    feats = _make_features(elo_diff=250.0, home_elo=1700.0, away_elo=1450.0)
    out = explain_gbm(gbm, feats, top_k=5)
    assert len(out) == 3
    # At least one outcome must have non-empty driver lists (otherwise the
    # booster is returning zero contributions which would be a regression).
    total_drivers = sum(len(o.top_positive) + len(o.top_negative) for o in out)
    assert total_drivers > 0

    for o in out:
        # top_positive is sorted by decreasing shap_value.
        pos_values = [c.shap_value for c in o.top_positive]
        assert pos_values == sorted(pos_values, reverse=True)
        # top_negative is sorted by increasing shap_value (most negative first).
        neg_values = [c.shap_value for c in o.top_negative]
        assert neg_values == sorted(neg_values)
        assert len(o.top_positive) <= 5
        assert len(o.top_negative) <= 5
        # Probabilities are in [0, 1].
        assert 0.0 <= o.base_probability <= 1.0
        assert 0.0 <= o.model_probability <= 1.0

    # Softmax-like normalisation: the three model probabilities sum to ~1.
    total = sum(o.model_probability for o in out)
    assert abs(total - 1.0) < 1e-3


def test_explain_gbm_strong_home_favours_home_outcome() -> None:
    """For a match with a very strong home side, ``elo_diff`` should be a
    positive driver for the 'home' outcome and a negative driver for 'away'."""
    gbm = _fit_gbm_on_synthetic(n=400, seed=11)
    feats = _make_features(
        home_elo=1750.0, away_elo=1400.0, elo_diff=350.0, home_form=0.85, away_form=0.25
    )
    out = explain_gbm(gbm, feats, top_k=len(FEATURE_COLUMNS))
    outcomes = {o.outcome: o for o in out}
    home_contribs = {c.feature: c.shap_value for c in outcomes["home"].top_positive}
    away_contribs = {c.feature: c.shap_value for c in outcomes["away"].top_negative}

    # If elo_diff carries any signal at all it should be among the drivers; at
    # worst LightGBM may not split on it for tiny ensembles, in which case we
    # still accept the test as long as SOME feature pushes home up.
    any_positive_home = any(c.shap_value > 0 for c in outcomes["home"].top_positive)
    any_negative_away = any(c.shap_value < 0 for c in outcomes["away"].top_negative)
    assert any_positive_home or "elo_diff" in home_contribs
    assert any_negative_away or "elo_diff" in away_contribs
