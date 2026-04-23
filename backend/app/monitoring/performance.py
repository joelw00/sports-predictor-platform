"""Rolling live performance of the active predictor.

Compares stored :class:`app.db.models.Prediction` rows with the realised
:class:`app.db.models.Match` outcome for the same ``(match_id, market)`` and
returns Brier / log-loss / accuracy on the intersection.

Only ``finished`` matches with non-null scores count as realised. Predictions
for matches that have not yet happened are ignored here; they don't have a
label.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import models as m


@dataclass(frozen=True)
class LivePerformance:
    market: str
    n: int
    brier: float | None
    log_loss: float | None
    accuracy: float | None

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {
            "market": self.market,
            "n": self.n,
            "brier": round(self.brier, 6) if self.brier is not None else None,
            "log_loss": round(self.log_loss, 6) if self.log_loss is not None else None,
            "accuracy": round(self.accuracy, 6) if self.accuracy is not None else None,
        }


_SELECTIONS_1X2 = ("home", "draw", "away")


def _outcome_1x2(match: m.Match) -> str | None:
    if match.home_score is None or match.away_score is None:
        return None
    if match.home_score > match.away_score:
        return "home"
    if match.home_score < match.away_score:
        return "away"
    return "draw"


def _outcome_btts(match: m.Match) -> str | None:
    if match.home_score is None or match.away_score is None:
        return None
    return "yes" if (match.home_score > 0 and match.away_score > 0) else "no"


def _outcome_over_2_5(match: m.Match) -> str | None:
    if match.home_score is None or match.away_score is None:
        return None
    total = match.home_score + match.away_score
    return "over" if total > 2.5 else "under"


_MARKET_OUTCOMES = {
    "1x2": (_outcome_1x2, _SELECTIONS_1X2),
    "btts": (_outcome_btts, ("yes", "no")),
    "over_2_5": (_outcome_over_2_5, ("over", "under")),
}


def _brier_multiclass(probs: np.ndarray, onehot: np.ndarray) -> float:
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def _log_loss_multiclass(probs: np.ndarray, onehot: np.ndarray, eps: float = 1e-12) -> float:
    clipped = np.clip(probs, eps, 1.0 - eps)
    return float(-np.mean(np.sum(onehot * np.log(clipped), axis=1)))


def compute_live_performance(
    db: Session,
    *,
    sport_code: str = "football",
    market: str = "1x2",
    window_days: int = 60,
    now: datetime | None = None,
) -> LivePerformance:
    """Return rolling performance for the given market over the last ``window_days``."""
    if market not in _MARKET_OUTCOMES:
        raise ValueError(f"Unsupported market: {market}")
    outcome_fn, selections = _MARKET_OUTCOMES[market]

    cutoff = (now or datetime.now(UTC)) - timedelta(days=window_days)

    stmt = (
        select(m.Match)
        .join(m.Sport, m.Match.sport_id == m.Sport.id)
        .where(
            m.Sport.code == sport_code,
            m.Match.status == "finished",
            m.Match.kickoff >= cutoff,
        )
    )
    matches = list(db.scalars(stmt))
    if not matches:
        return LivePerformance(market=market, n=0, brier=None, log_loss=None, accuracy=None)

    rows: list[tuple[np.ndarray, np.ndarray]] = []
    for match in matches:
        outcome = outcome_fn(match)
        if outcome is None:
            continue
        preds = (
            db.query(m.Prediction)
            .filter(m.Prediction.match_id == match.id, m.Prediction.market == market)
            .all()
        )
        by_selection = {p.selection: float(p.probability) for p in preds}
        if not all(s in by_selection for s in selections):
            continue
        probs = np.array([by_selection[s] for s in selections], dtype=float)
        total = probs.sum()
        if total <= 0:
            continue
        probs = probs / total  # defensive renorm
        onehot = np.array([1.0 if s == outcome else 0.0 for s in selections], dtype=float)
        rows.append((probs, onehot))

    if not rows:
        return LivePerformance(market=market, n=0, brier=None, log_loss=None, accuracy=None)

    probs_mat = np.vstack([r[0] for r in rows])
    labels_mat = np.vstack([r[1] for r in rows])
    correct = (np.argmax(probs_mat, axis=1) == np.argmax(labels_mat, axis=1)).mean()
    return LivePerformance(
        market=market,
        n=len(rows),
        brier=_brier_multiclass(probs_mat, labels_mat),
        log_loss=_log_loss_multiclass(probs_mat, labels_mat),
        accuracy=float(correct),
    )
