"""Gaussian baseline models for total-corners and total-cards markets.

Both markets follow the same approximation:

  total_X ~ Normal(μ, σ²)

where μ is the sum of the home and away rolling averages for the quantity
(corners for the home team + corners for the away team, adjusted by how many
each side typically concedes) and σ is estimated from the training residuals
per league (fallback to a fixed value when not enough data are available).

The O/U probability for a given line is then:

  P(total > line) = 1 - Φ((line + 0.5 - μ) / σ)

The 0.5 continuity correction is applied because totals are integer-valued.

These are intentionally simple baselines — they consume the same features the
1X2 stack already maintains and can be upgraded to a Poisson/NegBin model or a
gradient-boosted head once we have a real corners/cards training corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, sqrt
from statistics import mean, pstdev

from app.features.football import MatchFeatures

# Standard bookmaker lines.
DEFAULT_CORNER_LINES: tuple[float, ...] = (8.5, 9.5, 10.5, 11.5)
DEFAULT_CARD_LINES: tuple[float, ...] = (3.5, 4.5, 5.5)

# Safe fallbacks when there's no training data yet. Rough priors calibrated on
# top-5 league averages.
_FALLBACK_CORNERS_MU = 10.0
_FALLBACK_CORNERS_SIGMA = 3.2
_FALLBACK_CARDS_MU = 4.5
_FALLBACK_CARDS_SIGMA = 1.8


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _prob_over(line: float, mu: float, sigma: float) -> float:
    """P(total > line) with an integer continuity correction."""
    sigma = max(sigma, 1e-3)
    z = (line + 0.5 - mu) / sigma
    return max(0.0, min(1.0, 1.0 - _normal_cdf(z)))


@dataclass
class TotalsFit:
    mu_bias: float = 0.0
    sigma: float = 1.0
    n: int = 0


@dataclass
class CornersCardsModel:
    """Learns a scalar bias + residual σ for corners and cards totals.

    Prediction combines team-level rolling averages with the learned bias so
    the team features carry most of the signal; the bias only shifts things
    up/down if the league-wide μ differs from what the per-team averages
    sum to.
    """

    corners_fit: TotalsFit = field(default_factory=lambda: TotalsFit(0.0, _FALLBACK_CORNERS_SIGMA))
    cards_fit: TotalsFit = field(default_factory=lambda: TotalsFit(0.0, _FALLBACK_CARDS_SIGMA))
    fitted: bool = False

    def fit(
        self,
        train_features: list[MatchFeatures],
        corners_totals: list[int | None],
        cards_totals: list[int | None],
    ) -> CornersCardsModel:
        assert len(train_features) == len(corners_totals) == len(cards_totals)

        def _fit(target_idx: int) -> TotalsFit:
            # target_idx: 0 → corners, 1 → cards.
            residuals: list[float] = []
            for feats, value in zip(
                train_features,
                corners_totals if target_idx == 0 else cards_totals,
                strict=False,
            ):
                if value is None:
                    continue
                predicted = self._feature_mu(feats, target_idx)
                residuals.append(value - predicted)
            if not residuals:
                return TotalsFit(
                    0.0,
                    _FALLBACK_CORNERS_SIGMA if target_idx == 0 else _FALLBACK_CARDS_SIGMA,
                )
            mu_bias = float(mean(residuals))
            # σ is the residual spread *around* the (biased) feature-based μ —
            # i.e. the spread of (value − predicted) after the mean shift
            # mu_bias has been accounted for. Using observed totals here would
            # double-count feature-explained variance and inflate σ.
            centered = [r - mu_bias for r in residuals]
            sigma = float(pstdev(centered)) if len(centered) >= 2 else (
                _FALLBACK_CORNERS_SIGMA if target_idx == 0 else _FALLBACK_CARDS_SIGMA
            )
            if sigma <= 0.0:
                sigma = _FALLBACK_CORNERS_SIGMA if target_idx == 0 else _FALLBACK_CARDS_SIGMA
            return TotalsFit(mu_bias=mu_bias, sigma=sigma, n=len(residuals))

        self.corners_fit = _fit(0)
        self.cards_fit = _fit(1)
        self.fitted = self.corners_fit.n > 0 or self.cards_fit.n > 0
        return self

    @staticmethod
    def _feature_mu(feats: MatchFeatures, target_idx: int) -> float:
        if target_idx == 0:
            # Average of (home for + away against) and (away for + home against)
            # — this smooths out single-team outliers.
            a = 0.5 * (feats.home_corners_for_avg + feats.away_corners_against_avg)
            b = 0.5 * (feats.away_corners_for_avg + feats.home_corners_against_avg)
            return a + b
        return feats.home_cards_avg + feats.away_cards_avg

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_corners_mu_sigma(self, feats: MatchFeatures) -> tuple[float, float]:
        base = self._feature_mu(feats, 0)
        if base <= 0.0:
            base = _FALLBACK_CORNERS_MU
        return base + self.corners_fit.mu_bias, self.corners_fit.sigma

    def predict_cards_mu_sigma(self, feats: MatchFeatures) -> tuple[float, float]:
        base = self._feature_mu(feats, 1)
        if base <= 0.0:
            base = _FALLBACK_CARDS_MU
        return base + self.cards_fit.mu_bias, self.cards_fit.sigma

    def prob_corners_over(self, feats: MatchFeatures, line: float) -> float:
        mu, sigma = self.predict_corners_mu_sigma(feats)
        return _prob_over(line, mu, sigma)

    def prob_cards_over(self, feats: MatchFeatures, line: float) -> float:
        mu, sigma = self.predict_cards_mu_sigma(feats)
        return _prob_over(line, mu, sigma)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state(self) -> dict[str, float | int | bool]:
        return {
            "corners_mu_bias": self.corners_fit.mu_bias,
            "corners_sigma": self.corners_fit.sigma,
            "corners_n": self.corners_fit.n,
            "cards_mu_bias": self.cards_fit.mu_bias,
            "cards_sigma": self.cards_fit.sigma,
            "cards_n": self.cards_fit.n,
            "fitted": self.fitted,
        }

    @classmethod
    def from_state(cls, s: dict[str, float | int | bool]) -> CornersCardsModel:
        model = cls(
            corners_fit=TotalsFit(
                mu_bias=float(s.get("corners_mu_bias", 0.0)),
                sigma=float(s.get("corners_sigma", _FALLBACK_CORNERS_SIGMA)),
                n=int(s.get("corners_n", 0)),
            ),
            cards_fit=TotalsFit(
                mu_bias=float(s.get("cards_mu_bias", 0.0)),
                sigma=float(s.get("cards_sigma", _FALLBACK_CARDS_SIGMA)),
                n=int(s.get("cards_n", 0)),
            ),
        )
        model.fitted = bool(s.get("fitted", False))
        return model


__all__ = [
    "CornersCardsModel",
    "TotalsFit",
    "DEFAULT_CORNER_LINES",
    "DEFAULT_CARD_LINES",
]
