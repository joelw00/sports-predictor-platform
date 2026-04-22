"""Inference orchestrator: combines Poisson + GBM into calibrated market probabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.config import get_settings
from app.features.football import MatchFeatures
from app.ml.calibration import IsotonicCalibrator
from app.ml.ensemble import EnsembleFootballModel
from app.ml.gbm import FEATURE_COLUMNS, Gbm1X2Model
from app.ml.poisson import PoissonFootballModel, ScoreDistribution


@dataclass
class MarketProbabilities:
    market: str
    selection: str
    line: float | None
    probability: float


@dataclass
class PredictionBundle:
    match_id: int
    home_team: str
    away_team: str
    score_distribution: ScoreDistribution | None
    markets: list[MarketProbabilities] = field(default_factory=list)
    confidence: float = 0.5
    drivers: dict[str, float] = field(default_factory=dict)
    model_version: str = "v0"


class FootballPredictor:
    """Bundles Poisson, GBM 1X2, isotonic calibration and the ensemble."""

    def __init__(
        self,
        poisson: PoissonFootballModel | None = None,
        gbm: Gbm1X2Model | None = None,
        calibrator: IsotonicCalibrator | None = None,
        ensemble: EnsembleFootballModel | None = None,
        version: str = "v0",
    ) -> None:
        self.poisson = poisson or PoissonFootballModel()
        self.gbm = gbm or Gbm1X2Model()
        self.calibrator = calibrator or IsotonicCalibrator()
        self.ensemble = ensemble or EnsembleFootballModel(weights=(0.55, 0.45))
        self.version = version

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, feats: MatchFeatures) -> PredictionBundle:
        score_dist = self.poisson.predict(feats.home_team, feats.away_team)
        p_h_poi, p_d_poi, p_a_poi = score_dist.prob_1x2()
        poisson_probs = np.array([[p_h_poi, p_d_poi, p_a_poi]])

        X = pd.DataFrame([_features_to_row(feats)], columns=FEATURE_COLUMNS)
        gbm_probs = self.gbm.predict_proba(X)

        if self.calibrator.fitted:
            gbm_probs_cal = self.calibrator.transform(gbm_probs)
        else:
            gbm_probs_cal = gbm_probs

        blended = self.ensemble.blend(poisson_probs, gbm_probs_cal)[0]
        blended = blended / blended.sum()

        markets: list[MarketProbabilities] = [
            MarketProbabilities("1x2", "home", None, float(blended[0])),
            MarketProbabilities("1x2", "draw", None, float(blended[1])),
            MarketProbabilities("1x2", "away", None, float(blended[2])),
            MarketProbabilities("double_chance", "1x", None, float(blended[0] + blended[1])),
            MarketProbabilities("double_chance", "12", None, float(blended[0] + blended[2])),
            MarketProbabilities("double_chance", "x2", None, float(blended[1] + blended[2])),
        ]
        for line in (0.5, 1.5, 2.5, 3.5):
            p_over = score_dist.prob_over(line)
            markets.append(MarketProbabilities("over_under", "over", line, p_over))
            markets.append(MarketProbabilities("over_under", "under", line, 1 - p_over))
        p_btts = score_dist.prob_btts()
        markets.append(MarketProbabilities("btts", "yes", None, p_btts))
        markets.append(MarketProbabilities("btts", "no", None, 1 - p_btts))

        top_prob = float(max(blended))
        second_prob = float(sorted(blended, reverse=True)[1])
        confidence = max(0.0, min(1.0, top_prob - 0.5 * second_prob + 0.25))

        drivers = {
            k: float(v)
            for k, v in sorted(
                self.gbm.feature_importance().items(), key=lambda x: x[1], reverse=True
            )[:5]
        }

        return PredictionBundle(
            match_id=feats.match_id,
            home_team=feats.home_team,
            away_team=feats.away_team,
            score_distribution=score_dist,
            markets=markets,
            confidence=confidence,
            drivers=drivers,
            model_version=self.version,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "version": self.version,
            "poisson": self.poisson.state(),
            "gbm_booster": (
                self.gbm._booster.model_to_string()
                if self.gbm.fitted and self.gbm._booster is not None
                else None
            ),
            "gbm_metrics": self.gbm.metrics,
            "calibrator": self.calibrator._regs if self.calibrator.fitted else None,
            "ensemble_weights": list(self.ensemble.weights),
        }
        joblib.dump(bundle, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> FootballPredictor:
        import lightgbm as lgb

        bundle = joblib.load(Path(path))
        poisson = PoissonFootballModel.from_state(bundle["poisson"])
        gbm = Gbm1X2Model()
        if bundle.get("gbm_booster"):
            gbm._booster = lgb.Booster(model_str=bundle["gbm_booster"])
            gbm._fitted = True
        calibrator = IsotonicCalibrator()
        if bundle.get("calibrator"):
            calibrator._regs = list(bundle["calibrator"])
            calibrator._fitted = True
        ensemble = EnsembleFootballModel(
            weights=tuple(bundle.get("ensemble_weights", (0.55, 0.45)))
        )
        return cls(
            poisson=poisson,
            gbm=gbm,
            calibrator=calibrator,
            ensemble=ensemble,
            version=bundle.get("version", "v0"),
        )

    @classmethod
    def default_artifact_path(cls) -> Path:
        settings = get_settings()
        return Path(settings.artifacts_dir) / "football" / "predictor_v0.joblib"


def _features_to_row(feats: MatchFeatures) -> list[float]:
    return [
        feats.home_elo,
        feats.away_elo,
        feats.elo_diff,
        feats.home_form,
        feats.away_form,
        feats.home_goals_scored_avg,
        feats.home_goals_conceded_avg,
        feats.away_goals_scored_avg,
        feats.away_goals_conceded_avg,
        feats.home_xg_for_avg,
        feats.home_xg_against_avg,
        feats.away_xg_for_avg,
        feats.away_xg_against_avg,
        feats.h2h_home_win_rate,
        feats.h2h_draw_rate,
        feats.h2h_goals_avg,
        feats.home_rest_days,
        feats.away_rest_days,
        feats.home_shots_avg,
        feats.away_shots_avg,
    ]
