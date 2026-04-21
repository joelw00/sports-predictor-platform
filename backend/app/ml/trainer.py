"""Walk-forward training for the football ensemble predictor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import models as m
from app.features.football import FootballFeatureBuilder
from app.logging import get_logger
from app.ml.calibration import IsotonicCalibrator
from app.ml.gbm import FEATURE_COLUMNS, Gbm1X2Model
from app.ml.poisson import PoissonFootballModel
from app.ml.predictor import FootballPredictor, _features_to_row

log = get_logger(__name__)


@dataclass
class TrainingReport:
    n_samples: int
    log_loss: float
    accuracy: float
    version: str


class FootballTrainer:
    """Builds features, fits Poisson + GBM, calibrates and bundles into a predictor."""

    def __init__(self, version: str = "v0") -> None:
        self.version = version

    def train(self, db: Session) -> tuple[FootballPredictor, TrainingReport]:
        stmt = (
            select(m.Match)
            .join(m.Sport, m.Match.sport_id == m.Sport.id)
            .where(m.Sport.code == "football", m.Match.status == "finished")
            .order_by(m.Match.kickoff.asc())
        )
        finished = list(db.scalars(stmt))
        log.info("train.data", n_finished=len(finished))

        # ---- Poisson -------------------------------------------------
        poisson_inputs: list[tuple[str, str, int, int]] = []
        for match in finished:
            if match.home_score is None or match.away_score is None:
                continue
            home = db.get(m.Team, match.home_team_id)
            away = db.get(m.Team, match.away_team_id)
            if home is None or away is None:
                continue
            poisson_inputs.append((home.name, away.name, match.home_score, match.away_score))
        poisson = PoissonFootballModel().fit(poisson_inputs)

        # ---- Feature dataset for GBM --------------------------------
        builder = FootballFeatureBuilder()
        feats_list = []
        labels: list[int] = []
        for match in finished:
            feats = builder.snapshot(db, match)
            feats_list.append(feats)
            if match.home_score > match.away_score:  # type: ignore[operator]
                labels.append(0)
            elif match.home_score == match.away_score:
                labels.append(1)
            else:
                labels.append(2)
            builder._update_with_result(db, match)  # noqa: SLF001

        if not feats_list:
            log.warning("train.no_features")
            predictor = FootballPredictor(poisson=poisson, version=self.version)
            return predictor, TrainingReport(0, 0.0, 0.0, self.version)

        df = pd.DataFrame([_features_to_row(f) for f in feats_list], columns=FEATURE_COLUMNS)
        y = np.array(labels)

        gbm = Gbm1X2Model().fit(df, y)

        # ---- Isotonic calibration on rolling out-of-fold predictions ----
        calibrator = IsotonicCalibrator()
        if len(df) >= 60:
            tss = TimeSeriesSplit(n_splits=5)
            oof = np.zeros((len(df), 3))
            for tr_idx, va_idx in tss.split(df):
                fold_model = Gbm1X2Model().fit(df.iloc[tr_idx], y[tr_idx])
                oof[va_idx] = fold_model.predict_proba(df.iloc[va_idx])
            # Fit one isotonic per class.
            calibrator.fit(oof, y)
            cal_probs = calibrator.transform(oof)
            ll = float(log_loss(y, cal_probs, labels=[0, 1, 2]))
        else:
            preds = gbm.predict_proba(df)
            ll = float(log_loss(y, preds, labels=[0, 1, 2]))

        acc = float((gbm.predict_proba(df).argmax(axis=1) == y).mean())

        predictor = FootballPredictor(
            poisson=poisson,
            gbm=gbm,
            calibrator=calibrator,
            version=self.version,
        )
        report = TrainingReport(
            n_samples=len(df), log_loss=ll, accuracy=acc, version=self.version
        )
        log.info("train.done", **report.__dict__)
        return predictor, report
