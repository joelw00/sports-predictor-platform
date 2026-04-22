"""Walk-forward backtester — proper out-of-sample evaluation.

The legacy `Backtester` re-uses a predictor that was trained on the entire
match history, so when it is later asked to score those same matches it reports
inflated performance. This module eliminates that leakage by retraining the
predictor at every fold boundary and only scoring matches whose kickoff is
strictly after the fold's training cut-off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.backtesting.engine import BacktestResult, _is_win
from app.db import models as m
from app.features.football import FootballFeatureBuilder
from app.logging import get_logger
from app.metrics.calibration import brier_score, log_loss_multi, reliability_bins
from app.ml.trainer import FootballTrainer
from app.value_bet.engine import ModelProbability, OddsQuote, ValueBetEngine

log = get_logger(__name__)


@dataclass
class WalkForwardConfig:
    n_folds: int = 6
    min_train_folds: int = 2
    stake: float = 1.0
    strategy: str = "flat"  # flat | kelly
    market: str = "1x2"
    sport_code: str = "football"


@dataclass
class _BetState:
    bankroll: float = 0.0
    peak: float = 0.0
    drawdown: float = 0.0
    n_bets: int = 0
    n_wins: int = 0
    total_staked: float = 0.0
    total_return: float = 0.0
    wins_total: float = 0.0
    losses_total: float = 0.0
    curve: list[dict] = field(default_factory=list)
    per_selection: dict[str, dict[str, float]] = field(default_factory=dict)


class WalkForwardBacktester:
    """Retrain at every fold, score the next fold, aggregate metrics."""

    def __init__(
        self,
        *,
        n_folds: int = 6,
        min_train_folds: int = 2,
        engine: ValueBetEngine | None = None,
        stake: float = 1.0,
        strategy: str = "flat",
    ) -> None:
        if n_folds < 3:
            raise ValueError("n_folds must be >= 3 to leave a held-out window")
        if min_train_folds < 1 or min_train_folds >= n_folds:
            raise ValueError("min_train_folds must satisfy 1 <= min_train_folds < n_folds")
        self.n_folds = n_folds
        self.min_train_folds = min_train_folds
        self.engine = engine or ValueBetEngine()
        self.stake = stake
        self.strategy = strategy

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(
        self,
        db: Session,
        *,
        sport_code: str = "football",
        market: str = "1x2",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        label: str = "walk_forward",
    ) -> BacktestResult:
        stmt = (
            select(m.Match)
            .join(m.Sport, m.Match.sport_id == m.Sport.id)
            .where(m.Sport.code == sport_code, m.Match.status == "finished")
            .order_by(m.Match.kickoff.asc())
        )
        finished = list(db.scalars(stmt))
        if start_date:
            finished = [x for x in finished if x.kickoff >= start_date]
        if end_date:
            finished = [x for x in finished if x.kickoff <= end_date]

        n = len(finished)
        if n < self.n_folds * 10:
            log.warning(
                "walk_forward.not_enough_data",
                n=n,
                required=self.n_folds * 10,
            )

        fold_size = max(1, n // self.n_folds)
        # Fold boundaries. Last fold absorbs the remainder.
        boundaries: list[tuple[int, int]] = []
        for k in range(self.n_folds):
            fold_start = k * fold_size
            fold_end = (k + 1) * fold_size if k < self.n_folds - 1 else n
            boundaries.append((fold_start, fold_end))

        builder = FootballFeatureBuilder()
        state = _BetState()
        # Per-market probability collectors for calibration metrics.
        probs_1x2: list[np.ndarray] = []
        labels_1x2: list[int] = []

        # --- 1. Fast-forward the feature-builder state through the initial train window.
        train_end_idx = boundaries[self.min_train_folds][0]
        for match in finished[:train_end_idx]:
            # Pre-match snapshot isn't needed here; just accumulate state.
            builder._update_with_result(db, match)  # noqa: SLF001

        # --- 2. Train first fold's predictor.
        predictor, report = FootballTrainer(version="wf").train(
            db, training_matches=finished[:train_end_idx]
        )
        log.info(
            "walk_forward.fold_train",
            fold=self.min_train_folds,
            n_train=report.n_samples,
            log_loss=round(report.log_loss, 4),
        )

        # --- 3. Walk forward through each test fold.
        for fold_idx in range(self.min_train_folds, self.n_folds):
            fold_start, fold_end = boundaries[fold_idx]
            for match in finished[fold_start:fold_end]:
                feats = builder.snapshot(db, match)
                prediction = predictor.predict(feats)

                # Collect 1X2 probabilities for calibration metrics regardless of market.
                p_1x2 = _extract_1x2(prediction)
                if p_1x2 is not None:
                    probs_1x2.append(p_1x2)
                    labels_1x2.append(_1x2_label(match))

                # Evaluate bets on the chosen market.
                probs = [
                    ModelProbability(p.market, p.selection, p.line, p.probability)
                    for p in prediction.markets
                    if p.market == market
                ]
                odds_rows = (
                    db.query(m.OddsSnapshot)
                    .filter(
                        m.OddsSnapshot.match_id == match.id,
                        m.OddsSnapshot.market == market,
                        m.OddsSnapshot.is_closing.is_(True),
                    )
                    .all()
                )
                quotes = [
                    OddsQuote(o.bookmaker, o.market, o.selection, o.line, o.price)
                    for o in odds_rows
                ]
                bets = self.engine.evaluate(
                    match_id=match.id,
                    model_probs=probs,
                    odds=quotes,
                    confidence=prediction.confidence,
                )
                bets = self.engine.filter_and_rank(bets)
                if bets:
                    _apply_bet(state, match, bets[0], stake=self.stake, strategy=self.strategy)

                # Update builder state AFTER scoring, ready for the next match.
                builder._update_with_result(db, match)  # noqa: SLF001

            # Retrain with the expanded window for the next fold.
            if fold_idx < self.n_folds - 1:
                predictor, report = FootballTrainer(version="wf").train(
                    db, training_matches=finished[:fold_end]
                )
                log.info(
                    "walk_forward.fold_train",
                    fold=fold_idx + 1,
                    n_train=report.n_samples,
                    log_loss=round(report.log_loss, 4),
                )

        # --- 4. Aggregate metrics.
        roi = (state.total_return - state.total_staked) / state.total_staked if state.total_staked else 0.0
        yield_pct = roi * 100.0
        profit_factor = (
            state.wins_total / state.losses_total if state.losses_total > 0 else 0.0
        )

        cal_metrics: dict[str, float | list] = {}
        if probs_1x2:
            probs_arr = np.vstack(probs_1x2)
            labels_arr = np.array(labels_1x2, dtype=int)
            br = brier_score(probs_arr, labels_arr)
            cal_metrics = {
                "brier": round(br.score, 5),
                "log_loss": round(log_loss_multi(probs_arr, labels_arr), 5),
                "n_holdout": br.n,
                "reliability_home": reliability_bins(probs_arr, labels_arr, positive_class=0),
            }

        breakdown = {
            "mode": "walk_forward",
            "n_folds": self.n_folds,
            "min_train_folds": self.min_train_folds,
            "calibration": cal_metrics,
            "per_selection": {
                k: {
                    "bets": v["bets"],
                    "wins": v["wins"],
                    "hit_rate": round(v["wins"] / v["bets"], 3) if v["bets"] else 0.0,
                    "roi": round((v["return"] - v["staked"]) / v["staked"], 4)
                    if v["staked"]
                    else 0.0,
                }
                for k, v in state.per_selection.items()
            },
        }

        return BacktestResult(
            label=label,
            sport_code=sport_code,
            market=market,
            start_date=start_date
            or (finished[boundaries[self.min_train_folds][0]].kickoff if finished else datetime.now(UTC)),
            end_date=end_date
            or (finished[-1].kickoff if finished else datetime.now(UTC)),
            strategy=self.strategy,
            min_edge=self.engine.min_edge,
            stake=self.stake,
            n_bets=state.n_bets,
            n_wins=state.n_wins,
            total_staked=round(state.total_staked, 4),
            total_return=round(state.total_return, 4),
            roi=round(roi, 4),
            yield_pct=round(yield_pct, 3),
            max_drawdown=round(state.drawdown, 4),
            profit_factor=round(profit_factor, 3),
            equity_curve=state.curve,
            breakdown=breakdown,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_1x2(prediction) -> np.ndarray | None:
    p_home = p_draw = p_away = None
    for mkt in prediction.markets:
        if mkt.market != "1x2":
            continue
        if mkt.selection == "home":
            p_home = mkt.probability
        elif mkt.selection == "draw":
            p_draw = mkt.probability
        elif mkt.selection == "away":
            p_away = mkt.probability
    if p_home is None or p_draw is None or p_away is None:
        return None
    return np.array([p_home, p_draw, p_away])


def _1x2_label(match: m.Match) -> int:
    if match.home_score is None or match.away_score is None:
        raise ValueError("cannot label non-finished match")
    if match.home_score > match.away_score:
        return 0
    if match.home_score == match.away_score:
        return 1
    return 2


def _apply_bet(
    state: _BetState,
    match: m.Match,
    bet,
    *,
    stake: float,
    strategy: str,
) -> None:
    used_stake = stake if strategy == "flat" else stake * bet.kelly_fraction
    if used_stake <= 0:
        return
    won = _is_win(match, bet.market, bet.selection, bet.line)
    state.total_staked += used_stake
    state.n_bets += 1
    ret = 0.0
    if won:
        state.n_wins += 1
        ret = used_stake * bet.price
        state.total_return += ret
        state.wins_total += ret - used_stake
        state.bankroll += ret - used_stake
    else:
        state.losses_total += used_stake
        state.bankroll -= used_stake
    state.peak = max(state.peak, state.bankroll)
    state.drawdown = max(state.drawdown, state.peak - state.bankroll)
    per = state.per_selection.setdefault(
        bet.selection, {"bets": 0, "staked": 0.0, "return": 0.0, "wins": 0}
    )
    per["bets"] += 1
    per["staked"] += used_stake
    per["return"] += ret if won else 0.0
    per["wins"] += 1 if won else 0
    state.curve.append(
        {
            "date": match.kickoff.isoformat(),
            "bankroll": round(state.bankroll, 4),
            "edge": round(bet.edge, 4),
        }
    )
