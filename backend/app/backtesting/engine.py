"""Historical simulation of the value bet pipeline.

Walks through finished matches in chronological order, rebuilds features as they
would have looked pre-match, produces predictions, computes value bets against
stored odds, and tracks the resulting bankroll plus CLV (closing line value).

``entry_odds_strategy`` selects which snapshot is treated as the "price the
bettor took":

- ``"closing"`` — the ``is_closing=True`` snapshot. This is the legacy behaviour
  kept for back-compat; CLV is always 0 in this mode because the entry price
  equals the closing price.
- ``"opening"`` — the earliest pre-kickoff snapshot per bookmaker/selection.
  This is the realistic scenario a bettor faces; the backtester then looks up
  the matching closing snapshot and reports CLV per bet plus aggregate
  ``avg_clv`` / ``clv_win_rate`` / ``n_clv_tracked`` in the breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import models as m
from app.features.football import FootballFeatureBuilder
from app.ml.predictor import FootballPredictor
from app.odds.history import closing_price_lookup, entry_snapshots
from app.value_bet.engine import ModelProbability, OddsQuote, ValueBetEngine

EntryStrategy = Literal["closing", "opening"]


@dataclass
class BacktestResult:
    label: str
    sport_code: str
    market: str
    start_date: datetime
    end_date: datetime
    strategy: str
    min_edge: float
    stake: float
    n_bets: int
    n_wins: int
    total_staked: float
    total_return: float
    roi: float
    yield_pct: float
    max_drawdown: float
    profit_factor: float
    avg_clv: float = 0.0
    clv_win_rate: float = 0.0
    n_clv_tracked: int = 0
    equity_curve: list[dict] = field(default_factory=list)
    breakdown: dict = field(default_factory=dict)


def _entry_quotes(
    db: Session,
    *,
    match_id: int,
    market: str,
    strategy: EntryStrategy,
) -> list[OddsQuote]:
    """Return the quotes the bettor would have taken at ``strategy`` point."""
    if strategy == "closing":
        rows = (
            db.query(m.OddsSnapshot)
            .filter(
                m.OddsSnapshot.match_id == match_id,
                m.OddsSnapshot.market == market,
                m.OddsSnapshot.is_closing.is_(True),
            )
            .all()
        )
    else:
        rows = entry_snapshots(db, match_id=match_id, market=market)
    return [OddsQuote(r.bookmaker, r.market, r.selection, r.line, r.price) for r in rows]


def _clv(entry_price: float, closing_price: float) -> float:
    """Closing line value as a decimal gain over the closing market consensus.

    ``clv = entry_price / closing_price - 1``: positive means the bettor got
    a better price than where the market eventually settled. A neutral EV
    bettor who still consistently posts positive CLV is considered sharp.
    """
    if closing_price <= 0:
        return 0.0
    return entry_price / closing_price - 1.0


class Backtester:
    """Replays history and reports betting performance."""

    def __init__(
        self,
        predictor: FootballPredictor,
        *,
        engine: ValueBetEngine | None = None,
        stake: float = 1.0,
        strategy: str = "flat",
        entry_odds_strategy: EntryStrategy = "closing",
    ) -> None:
        self.predictor = predictor
        self.engine = engine or ValueBetEngine()
        self.stake = stake
        self.strategy = strategy
        self.entry_odds_strategy = entry_odds_strategy

    def run(
        self,
        db: Session,
        *,
        sport_code: str = "football",
        market: str = "1x2",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        label: str = "backtest",
    ) -> BacktestResult:
        stmt = (
            select(m.Match)
            .join(m.Sport, m.Match.sport_id == m.Sport.id)
            .where(m.Sport.code == sport_code, m.Match.status == "finished")
            .order_by(m.Match.kickoff.asc())
        )
        matches = list(db.scalars(stmt))
        if start_date:
            matches = [x for x in matches if x.kickoff >= start_date]
        if end_date:
            matches = [x for x in matches if x.kickoff <= end_date]

        builder = FootballFeatureBuilder()
        bankroll = 0.0
        peak = 0.0
        drawdown = 0.0
        curve: list[dict] = []
        n_bets = 0
        n_wins = 0
        total_staked = 0.0
        total_return = 0.0
        wins_total = 0.0
        losses_total = 0.0
        per_selection: dict[str, dict[str, float]] = {}
        clv_total = 0.0
        clv_pos = 0
        clv_n = 0

        for match in matches:
            feats = builder.snapshot(db, match)
            prediction = self.predictor.predict(feats)
            probs = [
                ModelProbability(p.market, p.selection, p.line, p.probability)
                for p in prediction.markets
                if p.market == market
            ]
            quotes = _entry_quotes(
                db, match_id=match.id, market=market, strategy=self.entry_odds_strategy
            )
            closing_lookup = (
                closing_price_lookup(db, match_id=match.id, market=market)
                if self.entry_odds_strategy == "opening"
                else {}
            )
            bets = self.engine.evaluate(
                match_id=match.id,
                model_probs=probs,
                odds=quotes,
                confidence=prediction.confidence,
            )
            bets = self.engine.filter_and_rank(bets)
            if bets:
                best = bets[0]
                stake = self.stake if self.strategy == "flat" else self.stake * best.kelly_fraction
                won = _is_win(match, best.market, best.selection, best.line)
                total_staked += stake
                n_bets += 1
                if won:
                    n_wins += 1
                    ret = stake * best.price
                    total_return += ret
                    wins_total += ret - stake
                    bankroll += ret - stake
                else:
                    losses_total += stake
                    bankroll -= stake
                peak = max(peak, bankroll)
                drawdown = max(drawdown, peak - bankroll)
                per = per_selection.setdefault(
                    best.selection, {"bets": 0, "staked": 0.0, "return": 0.0, "wins": 0}
                )
                per["bets"] += 1
                per["staked"] += stake
                per["return"] += ret if won else 0.0
                per["wins"] += 1 if won else 0
                key = (best.bookmaker, best.market, best.selection, best.line)
                closing_price = closing_lookup.get(key)
                clv_value: float | None = None
                if closing_price is not None and closing_price > 0:
                    clv_value = _clv(best.price, closing_price)
                    clv_total += clv_value
                    clv_n += 1
                    if clv_value > 0:
                        clv_pos += 1
                curve.append(
                    {
                        "date": match.kickoff.isoformat(),
                        "bankroll": round(bankroll, 4),
                        "edge": round(best.edge, 4),
                        "clv": round(clv_value, 4) if clv_value is not None else None,
                    }
                )
            builder._update_with_result(db, match)  # noqa: SLF001

        roi = (total_return - total_staked) / total_staked if total_staked else 0.0
        yield_pct = roi * 100.0
        profit_factor = wins_total / losses_total if losses_total > 0 else 0.0
        avg_clv = clv_total / clv_n if clv_n else 0.0
        clv_win_rate = clv_pos / clv_n if clv_n else 0.0

        return BacktestResult(
            label=label,
            sport_code=sport_code,
            market=market,
            start_date=start_date or (matches[0].kickoff if matches else datetime.now(UTC)),
            end_date=end_date or (matches[-1].kickoff if matches else datetime.now(UTC)),
            strategy=self.strategy,
            min_edge=self.engine.min_edge,
            stake=self.stake,
            n_bets=n_bets,
            n_wins=n_wins,
            total_staked=round(total_staked, 4),
            total_return=round(total_return, 4),
            roi=round(roi, 4),
            yield_pct=round(yield_pct, 3),
            max_drawdown=round(drawdown, 4),
            profit_factor=round(profit_factor, 3),
            avg_clv=round(avg_clv, 5),
            clv_win_rate=round(clv_win_rate, 4),
            n_clv_tracked=clv_n,
            equity_curve=curve,
            breakdown={
                "entry_odds_strategy": self.entry_odds_strategy,
                "per_selection": {
                    k: {
                        "bets": v["bets"],
                        "wins": v["wins"],
                        "hit_rate": round(v["wins"] / v["bets"], 3) if v["bets"] else 0.0,
                        "roi": round((v["return"] - v["staked"]) / v["staked"], 4)
                        if v["staked"]
                        else 0.0,
                    }
                    for k, v in per_selection.items()
                },
            },
        )


def _is_win(match: m.Match, market: str, selection: str, line: float | None) -> bool:
    if match.home_score is None or match.away_score is None:
        return False
    if market == "1x2":
        if selection == "home":
            return match.home_score > match.away_score
        if selection == "draw":
            return match.home_score == match.away_score
        if selection == "away":
            return match.home_score < match.away_score
    if market == "over_under" and line is not None:
        total = match.home_score + match.away_score
        if selection == "over":
            return total > line
        if selection == "under":
            return total < line
    if market == "btts":
        both = match.home_score > 0 and match.away_score > 0
        return (selection == "yes" and both) or (selection == "no" and not both)
    return False
