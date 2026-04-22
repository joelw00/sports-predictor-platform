"""Value bet engine.

Compares model probabilities against bookmaker odds, removes the overround,
computes edge, expected value and Kelly stake, and ranks opportunities.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from app.config import get_settings


@dataclass(frozen=True)
class OddsQuote:
    bookmaker: str
    market: str
    selection: str
    line: float | None
    price: float


@dataclass(frozen=True)
class ModelProbability:
    market: str
    selection: str
    line: float | None
    probability: float


@dataclass
class ValueBet:
    match_id: int
    market: str
    selection: str
    line: float | None
    bookmaker: str
    price: float
    p_model: float
    p_implied: float
    p_fair: float
    edge: float
    expected_value: float
    kelly_fraction: float
    confidence: float
    rationale: dict = field(default_factory=dict)


class ValueBetEngine:
    def __init__(
        self,
        *,
        min_edge: float | None = None,
        min_confidence: float | None = None,
    ) -> None:
        settings = get_settings()
        self.min_edge = min_edge if min_edge is not None else settings.value_bet_min_edge
        self.min_confidence = (
            min_confidence if min_confidence is not None else settings.value_bet_min_confidence
        )

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        *,
        match_id: int,
        model_probs: Iterable[ModelProbability],
        odds: Iterable[OddsQuote],
        confidence: float,
    ) -> list[ValueBet]:
        prob_index: dict[tuple[str, str, float | None], float] = {
            (p.market, p.selection, p.line): p.probability for p in model_probs
        }
        # Group odds by (bookmaker, market, line) to compute the overround.
        grouped: dict[tuple[str, str, float | None], list[OddsQuote]] = defaultdict(list)
        for quote in odds:
            grouped[(quote.bookmaker, quote.market, quote.line)].append(quote)

        out: list[ValueBet] = []
        for (bookmaker, market, line), quotes in grouped.items():
            implied = [1.0 / q.price for q in quotes]
            overround = max(0.0, sum(implied) - 1.0)
            normalised = 1.0 + overround
            for q, p_implied in zip(quotes, implied, strict=False):
                p_fair = p_implied / normalised
                key = (market, q.selection, line)
                p_model = prob_index.get(key)
                if p_model is None:
                    continue
                edge = p_model - p_fair
                ev = p_model * (q.price - 1.0) - (1.0 - p_model)
                kelly = max(0.0, edge / max(1e-6, q.price - 1.0))
                vb = ValueBet(
                    match_id=match_id,
                    market=market,
                    selection=q.selection,
                    line=line,
                    bookmaker=bookmaker,
                    price=q.price,
                    p_model=p_model,
                    p_implied=p_implied,
                    p_fair=p_fair,
                    edge=edge,
                    expected_value=ev,
                    kelly_fraction=kelly,
                    confidence=confidence,
                    rationale={
                        "overround": round(overround, 4),
                        "p_model_pct": round(100 * p_model, 2),
                        "p_fair_pct": round(100 * p_fair, 2),
                        "edge_pct": round(100 * edge, 2),
                    },
                )
                out.append(vb)
        return out

    def filter_and_rank(self, bets: Iterable[ValueBet]) -> list[ValueBet]:
        keep = [
            b
            for b in bets
            if b.edge >= self.min_edge
            and b.confidence >= self.min_confidence
            and b.price > 1.05
        ]
        keep.sort(key=lambda b: (b.edge, b.expected_value, b.confidence), reverse=True)
        return keep
