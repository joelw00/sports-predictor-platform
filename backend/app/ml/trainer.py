"""Walk-forward training for the football ensemble predictor.

The trainer fits the Poisson goal model and the GBM 1X2 classifier on the
supplied match window, then recalibrates them using a held-out time-series
split. Two calibrators are evaluated (isotonic + Platt) and the one with the
lower Brier score wins — see :mod:`app.ml.calibration`.

The training report also carries per-market Brier / reliability metrics so the
API can surface them in the dashboard (Phase 1 calibration tab).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import models as m
from app.features.football import FootballFeatureBuilder
from app.logging import get_logger
from app.metrics.calibration import brier_score, log_loss_multi, reliability_bins
from app.ml.calibration import (
    CalibrationSelector,
    IsotonicCalibrator,
    PlattCalibrator,
)
from app.ml.corners_cards import CornersCardsModel
from app.ml.gbm import FEATURE_COLUMNS, Gbm1X2Model
from app.ml.poisson import PoissonFootballModel
from app.ml.predictor import FootballPredictor, _features_to_row

log = get_logger(__name__)


@dataclass
class MarketCalibration:
    """Per-market calibration snapshot exposed through the API."""

    market: str
    n: int
    brier: float
    log_loss: float
    reliability: list[dict[str, float]] = field(default_factory=list)


@dataclass
class TrainingReport:
    n_samples: int
    log_loss: float
    accuracy: float
    version: str
    calibrator_kind: str = "none"
    brier_raw: float | None = None
    brier_calibrated: float | None = None
    n_holdout: int = 0
    markets: list[MarketCalibration] = field(default_factory=list)

    def to_metrics_dict(self) -> dict[str, Any]:
        """Flat dict compatible with :class:`app.db.models.ModelRegistry.metrics`."""
        return {
            "log_loss": self.log_loss,
            "accuracy": self.accuracy,
            "n_samples": self.n_samples,
            "calibrator": self.calibrator_kind,
            "brier_raw": self.brier_raw,
            "brier_calibrated": self.brier_calibrated,
            "n_holdout": self.n_holdout,
            "markets": [
                {
                    "market": mc.market,
                    "n": mc.n,
                    "brier": mc.brier,
                    "log_loss": mc.log_loss,
                    "reliability": mc.reliability,
                }
                for mc in self.markets
            ],
        }


class FootballTrainer:
    """Builds features, fits Poisson + GBM, calibrates and bundles into a predictor."""

    def __init__(self, version: str = "v0") -> None:
        self.version = version

    def train(
        self,
        db: Session,
        training_matches: list[m.Match] | None = None,
    ) -> tuple[FootballPredictor, TrainingReport]:
        """Train on the supplied matches (chronological). If None, uses all finished.

        Accepting an explicit list is what makes walk-forward backtesting honest:
        the caller passes only matches whose kickoff < fold_start.
        """
        if training_matches is None:
            stmt = (
                select(m.Match)
                .join(m.Sport, m.Match.sport_id == m.Sport.id)
                .where(m.Sport.code == "football", m.Match.status == "finished")
                .order_by(m.Match.kickoff.asc())
            )
            finished = list(db.scalars(stmt))
        else:
            finished = sorted(
                [x for x in training_matches if x.status == "finished"],
                key=lambda x: x.kickoff,
            )
        log.info("train.data", n_finished=len(finished))

        # ---- Poisson -------------------------------------------------
        # ``aligned`` has one entry per ``finished`` match — ``None`` where
        # the match is unusable for Poisson fitting (missing score, unknown
        # team). This keeps downstream fold indices (which are indices into
        # ``finished`` / ``df``) correctly aligned with the Poisson inputs.
        aligned: list[tuple[str, str, int, int] | None] = []
        for match in finished:
            if match.home_score is None or match.away_score is None:
                aligned.append(None)
                continue
            home = db.get(m.Team, match.home_team_id)
            away = db.get(m.Team, match.away_team_id)
            if home is None or away is None:
                aligned.append(None)
                continue
            aligned.append((home.name, away.name, match.home_score, match.away_score))
        poisson_inputs: list[tuple[str, str, int, int]] = [x for x in aligned if x is not None]
        poisson = PoissonFootballModel().fit(poisson_inputs)

        # ---- Feature dataset for GBM --------------------------------
        builder = FootballFeatureBuilder()
        feats_list = []
        labels: list[int] = []
        btts_labels: list[int] = []
        over25_labels: list[int] = []
        for match in finished:
            feats = builder.snapshot(db, match)
            feats_list.append(feats)
            if match.home_score > match.away_score:  # type: ignore[operator]
                labels.append(0)
            elif match.home_score == match.away_score:
                labels.append(1)
            else:
                labels.append(2)
            btts_labels.append(
                1 if (match.home_score or 0) > 0 and (match.away_score or 0) > 0 else 0
            )
            over25_labels.append(
                1 if ((match.home_score or 0) + (match.away_score or 0)) > 2 else 0
            )
            builder._update_with_result(db, match)  # noqa: SLF001

        if not feats_list:
            log.warning("train.no_features")
            predictor = FootballPredictor(poisson=poisson, version=self.version)
            return predictor, TrainingReport(0, 0.0, 0.0, self.version)

        df = pd.DataFrame([_features_to_row(f) for f in feats_list], columns=FEATURE_COLUMNS)
        y = np.array(labels)
        y_btts = np.array(btts_labels)
        y_ov25 = np.array(over25_labels)

        gbm = Gbm1X2Model().fit(df, y)

        # ---- Corners / cards totals model ---------------------------
        corners_totals: list[int | None] = []
        cards_totals: list[int | None] = []
        for match in finished:
            stats = db.query(m.MatchStat).filter_by(match_id=match.id).all()
            c_total = None
            k_total = None
            for s in stats:
                if s.corners is not None:
                    c_total = (c_total or 0) + int(s.corners)
                if s.yellow_cards is not None or s.red_cards is not None:
                    k_total = (
                        (k_total or 0) + int(s.yellow_cards or 0) + int(s.red_cards or 0)
                    )
            corners_totals.append(c_total)
            cards_totals.append(k_total)
        corners_cards = CornersCardsModel().fit(feats_list, corners_totals, cards_totals)

        # ---- Calibration on rolling out-of-fold predictions ------------
        calibrator: IsotonicCalibrator | PlattCalibrator = IsotonicCalibrator()
        calibrator_kind = "none"
        brier_raw: float | None = None
        brier_calibrated: float | None = None
        n_holdout = 0
        market_reports: list[MarketCalibration] = []
        ll_final: float

        if len(df) >= 60:
            tss = TimeSeriesSplit(n_splits=5)
            oof = np.zeros((len(df), 3))
            oof_btts = np.zeros(len(df))
            oof_ov25 = np.zeros(len(df))
            covered = np.zeros(len(df), dtype=bool)
            for tr_idx, va_idx in tss.split(df):
                fold_model = Gbm1X2Model().fit(df.iloc[tr_idx], y[tr_idx])
                oof[va_idx] = fold_model.predict_proba(df.iloc[va_idx])
                # Poisson-derived BTTS / O2.5 using team strengths fitted on
                # this fold only — keeps the calibration view honest too.
                fold_poisson = PoissonFootballModel().fit(
                    [aligned[i] for i in tr_idx if aligned[i] is not None]
                )
                for va in va_idx:
                    match = finished[va]
                    home = db.get(m.Team, match.home_team_id)
                    away = db.get(m.Team, match.away_team_id)
                    score_dist = fold_poisson.predict(
                        home.name if home else "?", away.name if away else "?"
                    )
                    oof_btts[va] = score_dist.prob_btts()
                    oof_ov25[va] = score_dist.prob_over(2.5)
                    covered[va] = True

            selector = CalibrationSelector()
            choice = selector.fit(oof[covered], y[covered])
            calibrator = choice.calibrator
            calibrator_kind = choice.kind
            brier_raw = choice.brier_raw
            brier_calibrated = min(choice.brier_isotonic, choice.brier_platt)
            n_holdout = choice.n_holdout

            cal_probs = calibrator.transform(oof[covered])
            ll_final = float(log_loss(y[covered], cal_probs, labels=[0, 1, 2]))

            # ---- per-market calibration reports ----------------------
            market_reports.append(
                _market_report_multiclass("1x2", cal_probs, y[covered])
            )
            # For BTTS / O2.5 we calibrate with a fresh selector — they're
            # independent binary forecasts, not a re-projection of the 1X2
            # probabilities. Each gets its own per-market report.
            for label, probs_1d, y_bin in [
                ("btts", oof_btts[covered], y_btts[covered]),
                ("over_2_5", oof_ov25[covered], y_ov25[covered]),
            ]:
                probs_2d = np.column_stack([1 - probs_1d, probs_1d])
                bin_selector = CalibrationSelector()
                bin_choice = bin_selector.fit(probs_2d, y_bin)
                bin_cal = bin_choice.calibrator.transform(probs_2d)
                market_reports.append(_market_report_binary(label, bin_cal, y_bin))
        else:
            preds = gbm.predict_proba(df)
            ll_final = float(log_loss(y, preds, labels=[0, 1, 2]))

        acc = float((gbm.predict_proba(df).argmax(axis=1) == y).mean())

        predictor = FootballPredictor(
            poisson=poisson,
            gbm=gbm,
            calibrator=calibrator,
            corners_cards=corners_cards,
            version=self.version,
        )
        report = TrainingReport(
            n_samples=len(df),
            log_loss=ll_final,
            accuracy=acc,
            version=self.version,
            calibrator_kind=calibrator_kind,
            brier_raw=brier_raw,
            brier_calibrated=brier_calibrated,
            n_holdout=n_holdout,
            markets=market_reports,
        )
        log.info(
            "train.done",
            n_samples=report.n_samples,
            log_loss=round(report.log_loss, 4),
            accuracy=round(report.accuracy, 4),
            calibrator=calibrator_kind,
            brier_raw=brier_raw,
            brier_calibrated=brier_calibrated,
        )
        return predictor, report


def _market_report_multiclass(
    market: str, probs: np.ndarray, labels: np.ndarray
) -> MarketCalibration:
    br = brier_score(probs, labels)
    return MarketCalibration(
        market=market,
        n=br.n,
        brier=round(br.score, 5),
        log_loss=round(log_loss_multi(probs, labels), 5),
        reliability=reliability_bins(probs, labels, positive_class=0),
    )


def _market_report_binary(
    market: str, probs_2d: np.ndarray, labels: np.ndarray
) -> MarketCalibration:
    br = brier_score(probs_2d, labels)
    return MarketCalibration(
        market=market,
        n=br.n,
        brier=round(br.score, 5),
        log_loss=round(log_loss_multi(probs_2d, labels), 5),
        reliability=reliability_bins(probs_2d, labels, positive_class=1),
    )
