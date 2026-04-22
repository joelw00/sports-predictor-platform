"""Admin / operations endpoints: manual ingestion trigger, source status, recent runs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import models as m
from app.db.session import get_db
from app.ingestion.orchestrator import ingest_all
from app.ingestion.registry import get_active_sources

router = APIRouter(prefix="/admin", tags=["admin"])


class SourceStatus(BaseModel):
    name: str
    enabled: bool
    sports: list[str]


class IngestionRunSummary(BaseModel):
    id: int
    source: str
    trigger: str
    started_at: datetime
    finished_at: datetime | None
    matches_upserted: int
    stats_upserted: int
    odds_upserted: int
    ok: bool
    error: str | None
    meta: dict[str, Any]


class IngestResponse(BaseModel):
    triggered_at: datetime
    counts: dict[str, int]
    demo_mode: bool
    active_sources: list[str]


@router.get("/sources", response_model=list[SourceStatus])
def list_sources() -> list[SourceStatus]:
    """Return the list of sources the registry would use right now."""
    return [
        SourceStatus(name=s.name, enabled=s.is_enabled(), sports=list(s.sports))
        for s in get_active_sources()
    ]


@router.get("/ingestion-runs", response_model=list[IngestionRunSummary])
def recent_ingestion_runs(
    limit: int = 20,
    db: Session = Depends(get_db),
) -> list[IngestionRunSummary]:
    runs = (
        db.query(m.IngestionRun)
        .order_by(desc(m.IngestionRun.started_at))
        .limit(max(1, min(limit, 200)))
        .all()
    )
    return [
        IngestionRunSummary(
            id=r.id,
            source=r.source,
            trigger=r.trigger,
            started_at=r.started_at,
            finished_at=r.finished_at,
            matches_upserted=r.matches_upserted,
            stats_upserted=r.stats_upserted,
            odds_upserted=r.odds_upserted,
            ok=r.ok,
            error=r.error,
            meta=r.meta or {},
        )
        for r in runs
    ]


@router.post("/ingest", response_model=IngestResponse)
def trigger_ingestion(db: Session = Depends(get_db)) -> IngestResponse:
    """Run every active source once, synchronously.

    Intended for operator / dashboard usage. For production traffic use the
    scheduled APScheduler job instead — this endpoint is blocking and not
    rate-limited.
    """
    settings = get_settings()
    active = get_active_sources()
    if not active:
        raise HTTPException(status_code=500, detail="no ingestion sources configured")
    triggered_at = datetime.now(tz=UTC)
    counts = ingest_all(db, sources=active, trigger="api")
    return IngestResponse(
        triggered_at=triggered_at,
        counts=counts,
        demo_mode=settings.is_demo_mode,
        active_sources=[s.name for s in active],
    )
