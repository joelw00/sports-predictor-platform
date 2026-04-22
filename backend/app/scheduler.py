"""Scheduled ingestion job.

A thin wrapper around APScheduler that runs :func:`app.ingestion.orchestrator.ingest_all`
daily. We intentionally keep this lightweight: no persistence layer, no distributed
locking. When the project grows beyond a single worker this should move to a
proper queue (RQ / Celery / Arq) and a dedicated worker image.
"""

from __future__ import annotations

import atexit
from datetime import UTC, datetime
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import get_settings
from app.db.session import SessionLocal
from app.ingestion.orchestrator import ingest_all
from app.logging import get_logger

log = get_logger(__name__)

_scheduler: BackgroundScheduler | None = None


def _run_daily_ingestion() -> dict[str, Any]:
    """Job body. Opens a short-lived session and runs every active source."""
    started = datetime.now(tz=UTC)
    db = SessionLocal()
    try:
        counts = ingest_all(db, trigger="scheduled")
    except Exception as exc:  # noqa: BLE001
        log.exception("scheduler.ingest.failed", error=str(exc))
        raise
    finally:
        db.close()
    log.info(
        "scheduler.ingest.done",
        duration_s=(datetime.now(tz=UTC) - started).total_seconds(),
        **counts,
    )
    return counts


def start_scheduler() -> BackgroundScheduler | None:
    """Start the background scheduler if enabled. Idempotent."""
    global _scheduler
    settings = get_settings()
    if not settings.scheduler_enabled:
        log.info("scheduler.disabled")
        return None
    if _scheduler is not None and _scheduler.running:
        return _scheduler

    sched = BackgroundScheduler(timezone="UTC")
    trigger = CronTrigger(
        hour=settings.scheduler_cron_hour,
        minute=settings.scheduler_cron_minute,
    )
    sched.add_job(
        _run_daily_ingestion,
        trigger=trigger,
        id="ingest_all_daily",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
        name="Daily ingestion across all active sources",
    )
    sched.start()
    atexit.register(lambda: sched.shutdown(wait=False) if sched.running else None)
    _scheduler = sched
    log.info(
        "scheduler.started",
        cron_hour=settings.scheduler_cron_hour,
        cron_minute=settings.scheduler_cron_minute,
    )
    return sched


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=False)
    _scheduler = None
