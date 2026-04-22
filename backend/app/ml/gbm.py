"""Gradient-boosted 1X2 model.

Lightweight wrapper around LightGBM. Uses numeric features from
``features/football.py`` and is combined with the Poisson model via an ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from lightgbm import Booster

FEATURE_COLUMNS = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_form",
    "away_form",
    "home_goals_scored_avg",
    "home_goals_conceded_avg",
    "away_goals_scored_avg",
    "away_goals_conceded_avg",
    "home_xg_for_avg",
    "home_xg_against_avg",
    "away_xg_for_avg",
    "away_xg_against_avg",
    "h2h_home_win_rate",
    "h2h_draw_rate",
    "h2h_goals_avg",
    "home_rest_days",
    "away_rest_days",
    "home_shots_avg",
    "away_shots_avg",
]


@dataclass
class _TrainMetrics:
    log_loss: float
    accuracy: float
    n_samples: int


class Gbm1X2Model:
    """LightGBM multi-class classifier for home / draw / away."""

    def __init__(self) -> None:
        self._booster: Booster | None = None
        self._fitted = False
        self._metrics: _TrainMetrics | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Gbm1X2Model:
        import lightgbm as lgb
        from sklearn.metrics import accuracy_score, log_loss
        from sklearn.model_selection import train_test_split

        X = X[FEATURE_COLUMNS]
        if len(X) < 40:
            # Not enough data to bother with early stopping — fit on everything.
            train = lgb.Dataset(X, label=y)
            params = self._params()
            self._booster = lgb.train(params, train, num_boost_round=200)
            preds = np.asarray(self._booster.predict(X))
            ll = float(log_loss(y, preds, labels=[0, 1, 2]))
            acc = float(accuracy_score(y, preds.argmax(axis=1)))
            self._metrics = _TrainMetrics(ll, acc, len(X))
            self._fitted = True
            return self

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=7, shuffle=True
        )
        train = lgb.Dataset(X_tr, label=y_tr)
        valid = lgb.Dataset(X_val, label=y_val, reference=train)
        params = self._params()
        self._booster = lgb.train(
            params,
            train,
            num_boost_round=400,
            valid_sets=[valid],
            callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
        )
        preds = np.asarray(self._booster.predict(X_val))
        ll = float(log_loss(y_val, preds, labels=[0, 1, 2]))
        acc = float(accuracy_score(y_val, preds.argmax(axis=1)))
        self._metrics = _TrainMetrics(ll, acc, len(X))
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted or self._booster is None:
            # Uninformative prior.
            out = np.tile([0.45, 0.25, 0.30], (len(X), 1))
            return out
        return np.asarray(self._booster.predict(X[FEATURE_COLUMNS]))

    def feature_importance(self) -> dict[str, float]:
        if not self._fitted or self._booster is None:
            return {}
        imp = self._booster.feature_importance(importance_type="gain")
        return {name: float(v) for name, v in zip(FEATURE_COLUMNS, imp, strict=False)}

    @property
    def metrics(self) -> dict[str, float | int]:
        if self._metrics is None:
            return {}
        return {
            "log_loss": self._metrics.log_loss,
            "accuracy": self._metrics.accuracy,
            "n_samples": self._metrics.n_samples,
        }

    @property
    def fitted(self) -> bool:
        return self._fitted

    @staticmethod
    def _params() -> dict:
        return {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 15,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 5,
            "verbosity": -1,
        }
