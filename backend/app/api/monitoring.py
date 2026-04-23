"""Monitoring endpoints — surfaces the latest drift / calibration / alert pass.

The dashboard polls ``/monitoring/latest`` to render alert badges; ``/history``
is for the admin UI; ``/run`` is a synchronous trigger for operators (same
shape as ``/admin/retrain``).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import models as m
from app.db.session import get_db
from app.monitoring import run_monitoring

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


def _row_to_dict(row: m.MonitoringSnapshot) -> dict[str, Any]:
    return {
        "id": row.id,
        "sport": row.sport_code,
        "market": row.market,
        "computed_at": row.computed_at.isoformat() if row.computed_at else None,
        "model_version": row.model_version,
        "n_recent_finished": row.n_recent_finished,
        "n_predictions_evaluated": row.n_predictions_evaluated,
        "brier_live": row.brier_live,
        "log_loss_live": row.log_loss_live,
        "accuracy_live": row.accuracy_live,
        "brier_training": row.brier_training,
        "max_psi": row.max_psi,
        "drift": row.drift or {},
        "alerts": row.alerts or [],
        "meta": row.meta or {},
    }


@router.get("/latest")
def get_latest(
    sport: str = "football",
    market: str = "1x2",
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Return the most recent snapshot for ``(sport, market)``.

    404 if no monitoring pass has been run yet — the dashboard should treat
    that as a soft "no data" state rather than an error badge.
    """
    row = (
        db.query(m.MonitoringSnapshot)
        .filter_by(sport_code=sport, market=market)
        .order_by(m.MonitoringSnapshot.computed_at.desc())
        .first()
    )
    if row is None:
        raise HTTPException(404, detail="No monitoring snapshot recorded yet")
    return _row_to_dict(row)


@router.get("/history")
def get_history(
    sport: str = "football",
    market: str = "1x2",
    limit: int = 30,
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """List recent snapshots, newest first. ``limit`` capped at 200 for safety."""
    limit = max(1, min(limit, 200))
    rows = (
        db.query(m.MonitoringSnapshot)
        .filter_by(sport_code=sport, market=market)
        .order_by(m.MonitoringSnapshot.computed_at.desc())
        .limit(limit)
        .all()
    )
    return [_row_to_dict(r) for r in rows]


@router.post("/run")
def trigger_run(
    sport: str = "football",
    market: str = "1x2",
    window_days: int = 30,
    reference_days: int = 180,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Synchronous monitoring pass — same surface as ``/admin/retrain``."""
    result = run_monitoring(
        db,
        sport_code=sport,
        market=market,
        window_days=window_days,
        reference_days=reference_days,
    )
    return result.to_payload()
