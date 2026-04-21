"""Independent Poisson goal model for football.

Fits per-team attack and defence strengths plus a home advantage from historical
match results. Produces a full score distribution from which 1X2, Over/Under and
BTTS probabilities are derived.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import exp, factorial

import numpy as np
from scipy.optimize import minimize


@dataclass
class ScoreDistribution:
    lambda_home: float
    lambda_away: float
    matrix: np.ndarray  # shape (max_goals+1, max_goals+1)

    def prob_1x2(self) -> tuple[float, float, float]:
        mat = self.matrix
        n = mat.shape[0]
        p_home = float(sum(mat[h, a] for h in range(n) for a in range(n) if h > a))
        p_draw = float(mat.trace())
        p_away = float(sum(mat[h, a] for h in range(n) for a in range(n) if h < a))
        s = p_home + p_draw + p_away
        return p_home / s, p_draw / s, p_away / s

    def prob_over(self, line: float) -> float:
        mat = self.matrix
        n = mat.shape[0]
        cutoff = int(line)  # integer cutoff for half lines
        p = 0.0
        for h in range(n):
            for a in range(n):
                if h + a > cutoff:
                    p += mat[h, a]
        return float(p / mat.sum())

    def prob_btts(self) -> float:
        mat = self.matrix
        n = mat.shape[0]
        p = 0.0
        for h in range(1, n):
            for a in range(1, n):
                p += mat[h, a]
        return float(p / mat.sum())


def _poisson_pmf(k: int, lam: float) -> float:
    return (lam**k) * exp(-lam) / factorial(k)


class PoissonFootballModel:
    """Independent Poisson with team-specific attack / defence strengths."""

    def __init__(self, max_goals: int = 7) -> None:
        self.max_goals = max_goals
        self._team_idx: dict[str, int] = {}
        self._attack: np.ndarray | None = None
        self._defence: np.ndarray | None = None
        self._home_adv: float = 0.25
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        matches: Iterable[tuple[str, str, int, int]],
        *,
        regularisation: float = 0.01,
    ) -> PoissonFootballModel:
        """`matches` is an iterable of (home_team, away_team, home_score, away_score)."""
        matches = list(matches)
        teams: dict[str, int] = {}
        for h, a, _, _ in matches:
            teams.setdefault(h, len(teams))
            teams.setdefault(a, len(teams))
        self._team_idx = teams
        n = len(teams)
        if n == 0 or not matches:
            # not enough data: fall back to neutral priors
            self._attack = np.zeros(n)
            self._defence = np.zeros(n)
            self._home_adv = 0.25
            self._fitted = True
            return self

        x0 = np.zeros(2 * n + 1)
        x0[-1] = 0.25  # initial home advantage

        # Pre-compute index arrays for speed.
        idx_h = np.array([teams[h] for h, _, _, _ in matches], dtype=int)
        idx_a = np.array([teams[a] for _, a, _, _ in matches], dtype=int)
        hs = np.array([hs for _, _, hs, _ in matches], dtype=float)
        as_ = np.array([as_ for _, _, _, as_ in matches], dtype=float)

        def nll(params: np.ndarray) -> float:
            attack = params[:n]
            defence = params[n : 2 * n]
            home_adv = params[-1]
            lam_h = np.exp(attack[idx_h] - defence[idx_a] + home_adv)
            lam_a = np.exp(attack[idx_a] - defence[idx_h])
            # Poisson NLL up to constants; add light L2 on strengths for stability.
            ll = -(hs * np.log(lam_h) - lam_h + as_ * np.log(lam_a) - lam_a).sum()
            ll += regularisation * (np.sum(attack**2) + np.sum(defence**2))
            return float(ll)

        res = minimize(nll, x0, method="L-BFGS-B")
        params = res.x
        self._attack = params[:n]
        self._defence = params[n : 2 * n]
        self._home_adv = float(params[-1])
        # Identifiability: centre the attack/defence so they sum to zero.
        self._attack -= self._attack.mean()
        self._defence -= self._defence.mean()
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, home: str, away: str) -> ScoreDistribution:
        if not self._fitted or self._attack is None or self._defence is None:
            # Fall back to league-average lambdas.
            return self._predict_with_lambdas(1.45, 1.15)
        if home not in self._team_idx or away not in self._team_idx:
            return self._predict_with_lambdas(1.45, 1.15)
        i = self._team_idx[home]
        j = self._team_idx[away]
        lam_h = float(np.exp(self._attack[i] - self._defence[j] + self._home_adv))
        lam_a = float(np.exp(self._attack[j] - self._defence[i]))
        return self._predict_with_lambdas(lam_h, lam_a)

    def _predict_with_lambdas(self, lam_h: float, lam_a: float) -> ScoreDistribution:
        lam_h = max(0.05, min(5.0, lam_h))
        lam_a = max(0.05, min(5.0, lam_a))
        n = self.max_goals + 1
        mat = np.zeros((n, n), dtype=float)
        for h in range(n):
            for a in range(n):
                mat[h, a] = _poisson_pmf(h, lam_h) * _poisson_pmf(a, lam_a)
        mat /= mat.sum()
        return ScoreDistribution(lambda_home=lam_h, lambda_away=lam_a, matrix=mat)

    @property
    def fitted(self) -> bool:
        return self._fitted

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state(self) -> dict:
        return {
            "team_idx": self._team_idx,
            "attack": self._attack.tolist() if self._attack is not None else None,
            "defence": self._defence.tolist() if self._defence is not None else None,
            "home_adv": self._home_adv,
            "max_goals": self.max_goals,
            "fitted": self._fitted,
        }

    @classmethod
    def from_state(cls, state: dict) -> PoissonFootballModel:
        model = cls(max_goals=state.get("max_goals", 7))
        model._team_idx = state.get("team_idx", {})
        model._attack = np.array(state["attack"]) if state.get("attack") is not None else None
        model._defence = np.array(state["defence"]) if state.get("defence") is not None else None
        model._home_adv = float(state.get("home_adv", 0.25))
        model._fitted = bool(state.get("fitted", False))
        return model
