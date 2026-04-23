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


def _run_weekly_retraining() -> dict[str, Any]:
    """Re-fit the football predictor on the latest finished matches.

    Imported lazily to avoid pulling sklearn / scipy into the FastAPI boot path
    when retraining is disabled.
    """
    from app.scripts.train_baseline import compute_version, train_once

    started = datetime.now(tz=UTC)
    version = compute_version()
    try:
        report = train_once(version=version)
    except Exception as exc:  # noqa: BLE001
        log.exception("scheduler.retrain.failed", error=str(exc))
        raise
    summary = {
        "version": version,
        "n_samples": report.n_samples,
        "log_loss": round(report.log_loss, 4),
        "accuracy": round(report.accuracy, 4),
        "calibrator": report.calibrator_kind,
        "brier_calibrated": report.brier_calibrated,
    }
    log.info(
        "scheduler.retrain.done",
        duration_s=(datetime.now(tz=UTC) - started).total_seconds(),
        **summary,
    )
    return summary


def start_scheduler() -> BackgroundScheduler | None:
    """Start the background scheduler if enabled. Idempotent."""
    global _scheduler
    settings = get_settings()
    if not settings.scheduler_enabled and not settings.retrain_enabled:
        log.info("scheduler.disabled")
        return None
    if _scheduler is not None and _scheduler.running:
        return _scheduler

    sched = BackgroundScheduler(timezone="UTC")

    if settings.scheduler_enabled:
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

    if settings.retrain_enabled:
        retrain_trigger = CronTrigger(
            day_of_week=settings.retrain_cron_day_of_week,
            hour=settings.retrain_cron_hour,
            minute=settings.retrain_cron_minute,
        )
        sched.add_job(
            _run_weekly_retraining,
            trigger=retrain_trigger,
            id="retrain_football_weekly",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
            name="Weekly retraining of the football predictor",
        )

    sched.start()
    atexit.register(lambda: sched.shutdown(wait=False) if sched.running else None)
    _scheduler = sched
    log.info(
        "scheduler.started",
        ingest_enabled=settings.scheduler_enabled,
        retrain_enabled=settings.retrain_enabled,
    )
    return sched


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=False)
    _scheduler = None
