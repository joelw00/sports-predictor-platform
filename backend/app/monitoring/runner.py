"""Orchestrate a full monitoring pass and persist a :class:`MonitoringSnapshot`.

Flow:

1. Resolve the active model from :class:`app.db.models.ModelRegistry` (if any)
   and read its training Brier + training-window matches from ``metrics``.
2. Build features for the reference window (last ``reference_days`` before the
   live window) and the current window (last ``window_days``).
3. Compute PSI + KS per feature and pick the max PSI.
4. Compute live Brier / log-loss / accuracy on stored predictions vs realised
   matches in the live window.
5. Evaluate alert rules.
6. Upsert a ``MonitoringSnapshot`` row and return a :class:`MonitoringResult`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import models as m
from app.features.football import FootballFeatureBuilder, MatchFeatures
from app.logging import get_logger
from app.monitoring.alerts import Alert, AlertRule, evaluate_alerts
from app.monitoring.drift import DriftStat, compute_feature_drift
from app.monitoring.performance import LivePerformance, compute_live_performance

log = get_logger(__name__)

# Numeric feature keys we check for drift. Keep in sync with
# :class:`app.features.football.MatchFeatures`.
_NUMERIC_FEATURES: tuple[str, ...] = (
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_form",
    "away_form",
    "home_goals_scored_avg",
    "home_goals_conceded_avg",
    "away_goals_scored_avg",
    "away_goals_conceded_avg",
    "home_xg_for_avg",
    "home_xg_against_avg",
    "away_xg_for_avg",
    "away_xg_against_avg",
    "h2h_home_win_rate",
    "h2h_draw_rate",
    "h2h_goals_avg",
    "home_rest_days",
    "away_rest_days",
    "home_shots_avg",
    "away_shots_avg",
)


@dataclass
class MonitoringResult:
    sport_code: str
    market: str
    computed_at: datetime
    model_version: str | None
    n_recent_finished: int
    live: LivePerformance
    brier_training: float | None
    drift: list[DriftStat]
    max_psi: float | None
    alerts: list[Alert]

    def to_payload(self) -> dict[str, Any]:
        return {
            "sport": self.sport_code,
            "market": self.market,
            "computed_at": self.computed_at.isoformat(),
            "model_version": self.model_version,
            "n_recent_finished": self.n_recent_finished,
            "live": self.live.to_dict(),
            "brier_training": self.brier_training,
            "max_psi": self.max_psi,
            "drift": [d.to_dict() for d in self.drift],
            "alerts": [a.to_dict() for a in self.alerts],
        }


def _split_features(
    features: list[MatchFeatures],
    *,
    reference_cutoff: datetime,
    live_cutoff: datetime,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Split feature rows into reference (< reference_cutoff) and current (>= live_cutoff)."""
    ref_buckets: dict[str, list[float]] = {k: [] for k in _NUMERIC_FEATURES}
    cur_buckets: dict[str, list[float]] = {k: [] for k in _NUMERIC_FEATURES}
    for feat in features:
        if feat.kickoff < reference_cutoff:
            target = ref_buckets
        elif feat.kickoff >= live_cutoff:
            target = cur_buckets
        else:
            continue
        row = feat.as_dict()
        for key in _NUMERIC_FEATURES:
            val = row.get(key)
            if isinstance(val, int | float):
                target[key].append(float(val))
    return ref_buckets, cur_buckets


def _active_model(
    db: Session, *, sport_code: str, market: str
) -> m.ModelRegistry | None:
    return (
        db.query(m.ModelRegistry)
        .filter_by(sport_code=sport_code, market=market, is_active=True)
        .order_by(m.ModelRegistry.updated_at.desc())
        .first()
    )


def _finished_matches(
    db: Session, *, sport_code: str, since: datetime
) -> list[m.Match]:
    stmt = (
        select(m.Match)
        .join(m.Sport, m.Match.sport_id == m.Sport.id)
        .where(
            m.Sport.code == sport_code,
            m.Match.status == "finished",
            m.Match.kickoff >= since,
        )
        .order_by(m.Match.kickoff.asc())
    )
    return list(db.scalars(stmt))


def run_monitoring(
    db: Session,
    *,
    sport_code: str = "football",
    market: str = "1x2",
    window_days: int = 30,
    reference_days: int = 180,
    rule: AlertRule | None = None,
    now: datetime | None = None,
) -> MonitoringResult:
    """Run one monitoring pass and persist a :class:`MonitoringSnapshot`.

    ``window_days`` bounds the live window (most recent N days). The reference
    window is the ``reference_days`` immediately preceding the live window, so
    we compare "recent history" vs "current" rather than "all history" vs
    "current" (the latter masks gradual drift).
    """
    now = now or datetime.now(UTC)
    live_cutoff = now - timedelta(days=window_days)
    reference_cutoff = live_cutoff - timedelta(days=reference_days)

    # Features across both windows, computed with warm Elo / form state.
    builder = FootballFeatureBuilder()
    matches_window = _finished_matches(db, sport_code=sport_code, since=reference_cutoff)
    features = builder.build_for_matches(db, matches_window)
    ref_buckets, cur_buckets = _split_features(
        features, reference_cutoff=live_cutoff, live_cutoff=live_cutoff
    )
    drift_stats = compute_feature_drift(ref_buckets, cur_buckets)
    max_psi = max((d.psi for d in drift_stats), default=None)

    # Live performance on realised predictions.
    live = compute_live_performance(
        db,
        sport_code=sport_code,
        market=market,
        window_days=window_days,
        now=now,
    )

    # Active model metadata.
    active = _active_model(db, sport_code=sport_code, market=market)
    model_version = active.version if active else None
    model_trained_at = active.updated_at if active else None
    brier_training: float | None = None
    if active and isinstance(active.metrics, dict):
        # ``or`` would drop a legitimate 0.0 Brier; be explicit about None.
        cal = active.metrics.get("brier_calibrated")
        brier_training = cal if cal is not None else active.metrics.get("brier_raw")

    n_recent_finished = sum(1 for mt in matches_window if mt.kickoff >= live_cutoff)

    alerts = evaluate_alerts(
        n_recent_finished=n_recent_finished,
        n_predictions_evaluated=live.n,
        max_psi=max_psi,
        model_trained_at=model_trained_at,
        brier_live=live.brier,
        brier_training=brier_training,
        now=now,
        rule=rule,
    )

    snapshot = m.MonitoringSnapshot(
        sport_code=sport_code,
        market=market,
        computed_at=now,
        model_version=model_version,
        n_recent_finished=n_recent_finished,
        n_predictions_evaluated=live.n,
        brier_live=live.brier,
        log_loss_live=live.log_loss,
        accuracy_live=live.accuracy,
        brier_training=brier_training,
        max_psi=max_psi,
        drift={
            "window_days": window_days,
            "reference_days": reference_days,
            "features": [d.to_dict() for d in drift_stats],
        },
        alerts=[a.to_dict() for a in alerts],
        meta={
            "reference_cutoff": reference_cutoff.isoformat(),
            "live_cutoff": live_cutoff.isoformat(),
        },
    )
    db.add(snapshot)
    db.commit()
    log.info(
        "monitoring.snapshot",
        sport=sport_code,
        market=market,
        n_recent_finished=n_recent_finished,
        n_predictions_evaluated=live.n,
        brier_live=live.brier,
        brier_training=brier_training,
        max_psi=max_psi,
        alerts=[a.code for a in alerts],
    )

    return MonitoringResult(
        sport_code=sport_code,
        market=market,
        computed_at=now,
        model_version=model_version,
        n_recent_finished=n_recent_finished,
        live=live,
        brier_training=brier_training,
        drift=drift_stats,
        max_psi=max_psi,
        alerts=alerts,
    )
