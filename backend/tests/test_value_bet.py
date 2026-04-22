from __future__ import annotations

from app.value_bet.engine import ModelProbability, OddsQuote, ValueBetEngine


def test_engine_flags_positive_edge() -> None:
    engine = ValueBetEngine(min_edge=0.01, min_confidence=0.0)
    model = [
        ModelProbability("1x2", "home", None, 0.55),
        ModelProbability("1x2", "draw", None, 0.25),
        ModelProbability("1x2", "away", None, 0.20),
    ]
    odds = [
        # Fair probability 1/2.00 = 0.5, model says 0.55 → edge positive.
        OddsQuote("Book", "1x2", "home", None, 2.00),
        OddsQuote("Book", "1x2", "draw", None, 3.80),
        OddsQuote("Book", "1x2", "away", None, 4.20),
    ]
    bets = engine.evaluate(match_id=1, model_probs=model, odds=odds, confidence=0.7)
    bets = engine.filter_and_rank(bets)
    assert len(bets) >= 1
    assert bets[0].selection == "home"
    assert bets[0].edge > 0


def test_engine_strips_overround() -> None:
    engine = ValueBetEngine(min_edge=0.0, min_confidence=0.0)
    odds = [
        OddsQuote("Book", "1x2", "home", None, 1.80),
        OddsQuote("Book", "1x2", "draw", None, 3.60),
        OddsQuote("Book", "1x2", "away", None, 4.40),
    ]
    model = [
        ModelProbability("1x2", "home", None, 0.55),
        ModelProbability("1x2", "draw", None, 0.25),
        ModelProbability("1x2", "away", None, 0.20),
    ]
    bets = engine.evaluate(match_id=1, model_probs=model, odds=odds, confidence=0.7)
    # Implied probabilities sum > 1 but p_fair should be normalised to ~1.
    fair_sum = sum(b.p_fair for b in bets)
    assert abs(fair_sum - 1.0) < 1e-6


def test_engine_respects_min_edge() -> None:
    engine = ValueBetEngine(min_edge=0.10, min_confidence=0.0)
    odds = [OddsQuote("Book", "1x2", "home", None, 2.00)]
    model = [ModelProbability("1x2", "home", None, 0.52)]
    bets = engine.evaluate(match_id=1, model_probs=model, odds=odds, confidence=0.9)
    assert engine.filter_and_rank(bets) == []
