"""Train the baseline football predictor and persist the artifact.

Usage:
    python -m app.scripts.train_baseline [--if-missing]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.db import SessionLocal
from app.db import models as m
from app.logging import configure_logging, get_logger
from app.ml.predictor import FootballPredictor
from app.ml.trainer import FootballTrainer
from app.predictions.service import PredictionService

log = get_logger(__name__)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--if-missing", action="store_true")
    args = parser.parse_args()

    artifact = FootballPredictor.default_artifact_path()
    if args.if_missing and Path(artifact).exists():
        log.info("train_baseline.skip", reason="artifact already exists", path=str(artifact))
    else:
        with SessionLocal() as db:
            predictor, report = FootballTrainer(version="v0").train(db)
            predictor.save(artifact)
            # Register metrics in model_registry.
            row = (
                db.query(m.ModelRegistry)
                .filter_by(sport_code="football", market="1x2", version=report.version)
                .one_or_none()
            )
            if row is None:
                row = m.ModelRegistry(
                    sport_code="football",
                    market="1x2",
                    family="ensemble_poisson_gbm",
                    version=report.version,
                    artifact_path=str(artifact),
                )
                db.add(row)
            row.metrics = {
                "log_loss": report.log_loss,
                "accuracy": report.accuracy,
                "n_samples": report.n_samples,
            }
            row.is_active = True
            db.commit()
            log.info("train_baseline.done", **report.__dict__)

    # Refresh predictions & value bets for upcoming matches.
    with SessionLocal() as db:
        PredictionService().refresh(db)


if __name__ == "__main__":
    main()
