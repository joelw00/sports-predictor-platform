"""Unit tests for the Gaussian corners/cards totals baseline."""

from __future__ import annotations

from datetime import UTC, datetime

from app.features.football import MatchFeatures
from app.ml.corners_cards import DEFAULT_CARD_LINES, DEFAULT_CORNER_LINES, CornersCardsModel


def _feats(**overrides: float) -> MatchFeatures:
    base: dict[str, object] = {
        "match_id": 1,
        "home_team": "Home",
        "away_team": "Away",
        "competition": "Test",
        "kickoff": datetime(2024, 1, 1, tzinfo=UTC),
        "home_elo": 1500.0,
        "away_elo": 1500.0,
        "elo_diff": 0.0,
        "home_form": 0.5,
        "away_form": 0.5,
        "home_goals_scored_avg": 1.3,
        "home_goals_conceded_avg": 1.3,
        "away_goals_scored_avg": 1.1,
        "away_goals_conceded_avg": 1.5,
        "home_xg_for_avg": 1.3,
        "home_xg_against_avg": 1.3,
        "away_xg_for_avg": 1.1,
        "away_xg_against_avg": 1.5,
        "h2h_home_win_rate": 0.45,
        "h2h_draw_rate": 0.25,
        "h2h_goals_avg": 2.6,
        "home_rest_days": 7.0,
        "away_rest_days": 7.0,
        "home_shots_avg": 12.0,
        "away_shots_avg": 10.0,
        "home_corners_for_avg": 5.5,
        "home_corners_against_avg": 4.5,
        "away_corners_for_avg": 4.5,
        "away_corners_against_avg": 5.5,
        "home_cards_avg": 2.3,
        "away_cards_avg": 2.2,
    }
    base.update(overrides)
    return MatchFeatures(**base)  # type: ignore[arg-type]


def test_unfitted_model_uses_sensible_defaults_for_corners() -> None:
    model = CornersCardsModel()
    feats = _feats()
    # Unfitted → fit.n_samples == 0 but we still get sensible outputs.
    mu, sigma = model.predict_corners_mu_sigma(feats)
    assert 8.0 < mu < 12.0
    assert 2.5 < sigma < 4.0
    # P(over) should decrease monotonically as the line increases.
    probs = [model.prob_corners_over(feats, line) for line in DEFAULT_CORNER_LINES]
    assert all(probs[i] > probs[i + 1] for i in range(len(probs) - 1))
    # All within [0, 1].
    for p in probs:
        assert 0.0 <= p <= 1.0


def test_unfitted_model_uses_sensible_defaults_for_cards() -> None:
    model = CornersCardsModel()
    feats = _feats()
    mu, sigma = model.predict_cards_mu_sigma(feats)
    assert 3.5 < mu < 5.5
    assert 1.0 < sigma < 2.5
    probs = [model.prob_cards_over(feats, line) for line in DEFAULT_CARD_LINES]
    assert all(probs[i] > probs[i + 1] for i in range(len(probs) - 1))


def test_fit_learns_mu_bias_from_residuals() -> None:
    """If the actual totals systematically exceed the feature-based μ, the
    learned bias should shift predictions upward (and vice versa)."""
    feats = [_feats() for _ in range(12)]
    # Feature-based μ for corners ≈ 10.0. If real totals average 13.0, the
    # bias should be ≈ +3.
    corners_totals: list[int | None] = [12, 14, 13, 13, 12, 14, 13, 13, 13, 13, 13, 13]
    # Features imply 4.5 → real totals average ~6.0 ⇒ positive bias.
    cards_totals: list[int | None] = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
    model = CornersCardsModel().fit(feats, corners_totals, cards_totals)
    assert model.fitted is True
    assert 2.0 < model.corners_fit.mu_bias < 4.0
    assert 1.0 < model.cards_fit.mu_bias < 2.0
    # The corners σ estimate should be greater than zero (data has variance).
    assert model.corners_fit.sigma > 0.0


def test_missing_totals_are_skipped() -> None:
    feats = [_feats() for _ in range(4)]
    model = CornersCardsModel().fit(feats, [None, 10, None, 11], [None, None, None, None])
    # Cards had zero usable samples → fallback σ kicks in.
    assert model.cards_fit.sigma > 0.0
    assert model.corners_fit.n == 2


def test_state_roundtrip_preserves_parameters() -> None:
    feats = [_feats() for _ in range(6)]
    model = CornersCardsModel().fit(feats, [11, 12, 10, 11, 12, 10], [4, 5, 4, 5, 4, 5])
    state = model.state()
    restored = CornersCardsModel.from_state(state)
    assert restored.corners_fit.mu_bias == model.corners_fit.mu_bias
    assert restored.cards_fit.sigma == model.cards_fit.sigma
    assert restored.fitted == model.fitted
    # Predictions should match exactly.
    for line in DEFAULT_CORNER_LINES:
        assert (
            restored.prob_corners_over(feats[0], line)
            == model.prob_corners_over(feats[0], line)
        )


def test_over_under_probabilities_sum_to_one() -> None:
    model = CornersCardsModel()
    feats = _feats()
    for line in DEFAULT_CORNER_LINES:
        p_over = model.prob_corners_over(feats, line)
        # Under = 1 - over in the inference layer.
        assert 0.0 <= p_over <= 1.0
        assert abs((p_over + (1.0 - p_over)) - 1.0) < 1e-9


def test_strong_favourite_has_higher_corners_mu_than_weak_team() -> None:
    strong = _feats(
        home_corners_for_avg=8.0,
        away_corners_against_avg=7.0,
    )
    weak = _feats(
        home_corners_for_avg=3.0,
        away_corners_against_avg=3.0,
    )
    model = CornersCardsModel()
    strong_mu, _ = model.predict_corners_mu_sigma(strong)
    weak_mu, _ = model.predict_corners_mu_sigma(weak)
    assert strong_mu > weak_mu
