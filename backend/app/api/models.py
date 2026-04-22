"""Model registry endpoints.

Currently surfaces calibration diagnostics (per-market Brier, reliability
diagrams and the chosen calibrator) so the dashboard can render them.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import models as m
from app.db.session import get_db

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/calibration")
def get_calibration(
    sport: str = "football",
    market: str = "1x2",
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Return the active model's calibration report.

    The response is intentionally shaped to be consumed directly by the
    frontend reliability chart and by any external monitoring.
    """
    row = (
        db.query(m.ModelRegistry)
        .filter_by(sport_code=sport, market=market, is_active=True)
        .order_by(m.ModelRegistry.updated_at.desc())
        .first()
    )
    if row is None:
        raise HTTPException(404, detail="No active model registered for this scope")

    metrics = row.metrics or {}
    markets = metrics.get("markets", []) or []
    return {
        "sport": sport,
        "market": market,
        "version": row.version,
        "family": row.family,
        "artifact_path": row.artifact_path,
        "calibrator": metrics.get("calibrator", "none"),
        "brier_raw": metrics.get("brier_raw"),
        "brier_calibrated": metrics.get("brier_calibrated"),
        "log_loss": metrics.get("log_loss"),
        "accuracy": metrics.get("accuracy"),
        "n_samples": metrics.get("n_samples"),
        "n_holdout": metrics.get("n_holdout"),
        "markets": markets,
    }


@router.get("/registry")
def list_registry(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    """List every model row — active or retired — for the admin UI."""
    rows = (
        db.query(m.ModelRegistry)
        .order_by(m.ModelRegistry.updated_at.desc())
        .all()
    )
    return [
        {
            "id": r.id,
            "sport": r.sport_code,
            "market": r.market,
            "family": r.family,
            "version": r.version,
            "artifact_path": r.artifact_path,
            "is_active": r.is_active,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
            "metrics": r.metrics or {},
        }
        for r in rows
    ]
