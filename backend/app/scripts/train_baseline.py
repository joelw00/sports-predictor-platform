"""Train the baseline football predictor and persist the artifact.

Usage:
    python -m app.scripts.train_baseline [--if-missing] [--version VERSION]

If ``--version`` is omitted we derive a timestamped + git-sha version so every
retrain produces a unique artifact (and a fresh :class:`ModelRegistry` row).
"""

from __future__ import annotations

import argparse
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import SessionLocal
from app.db import models as m
from app.logging import configure_logging, get_logger
from app.ml.trainer import FootballTrainer, TrainingReport
from app.predictions.service import PredictionService

log = get_logger(__name__)


def compute_version() -> str:
    """Stable, monotonically-increasing version id.

    Format: ``YYYYMMDDTHHMMSSZ-<short_sha>`` so artifacts are sortable and carry
    the provenance of the code revision that produced them. Falls back to
    ``YYYYMMDDTHHMMSSZ-nogit`` when we aren't in a git checkout.
    """
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:  # noqa: BLE001
        sha = "nogit"
    return f"{ts}-{sha}"


def artifact_path_for_version(version: str) -> Path:
    settings = get_settings()
    return Path(settings.artifacts_dir) / "football" / f"predictor_{version}.joblib"


def _register_and_activate(
    db: Session,
    *,
    version: str,
    artifact: Path,
    report: TrainingReport,
) -> None:
    """Insert / update the ModelRegistry row for this version and deactivate older ones."""
    row = (
        db.query(m.ModelRegistry)
        .filter_by(sport_code="football", market="1x2", version=version)
        .one_or_none()
    )
    if row is None:
        row = m.ModelRegistry(
            sport_code="football",
            market="1x2",
            family="ensemble_poisson_gbm",
            version=version,
            artifact_path=str(artifact),
        )
        db.add(row)
    row.artifact_path = str(artifact)
    row.metrics = report.to_metrics_dict()
    row.is_active = True

    # Deactivate prior versions for the same scope so `is_active=True` remains
    # unique-ish (we keep the row for provenance but stop advertising it).
    (
        db.query(m.ModelRegistry)
        .filter(
            m.ModelRegistry.sport_code == "football",
            m.ModelRegistry.market == "1x2",
            m.ModelRegistry.version != version,
        )
        .update({m.ModelRegistry.is_active: False})
    )


def train_once(version: str | None = None) -> TrainingReport:
    """Train, persist the artifact, register it. Returns the TrainingReport."""
    version = version or compute_version()
    artifact = artifact_path_for_version(version)
    with SessionLocal() as db:
        predictor, report = FootballTrainer(version=version).train(db)
        predictor.save(artifact)
        _register_and_activate(db, version=version, artifact=artifact, report=report)
        db.commit()
        log.info(
            "train_baseline.done",
            version=version,
            artifact=str(artifact),
            log_loss=round(report.log_loss, 4),
            accuracy=round(report.accuracy, 4),
            calibrator=report.calibrator_kind,
        )

    # Refresh predictions & value bets for upcoming matches.
    with SessionLocal() as db:
        PredictionService().refresh(db)
    return report


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--if-missing", action="store_true")
    parser.add_argument(
        "--version",
        default=os.environ.get("TRAIN_VERSION"),
        help="Explicit version string (default: timestamp + git short sha).",
    )
    args = parser.parse_args()

    version = args.version or compute_version()
    artifact = artifact_path_for_version(version)
    if args.if_missing and Path(artifact).exists():
        log.info(
            "train_baseline.skip",
            reason="artifact already exists",
            path=str(artifact),
            version=version,
        )
        return

    train_once(version=version)


if __name__ == "__main__":
    main()
