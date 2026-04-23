"""Probability calibrators for multi-class classifiers.

The project needs two complementary calibrators:

* :class:`IsotonicCalibrator` — non-parametric, very flexible, ideal when the
  raw probabilities are already monotone but locally biased. Tends to
  overfit on small samples.
* :class:`PlattCalibrator` — logistic regression on the predicted scores.
  Extremely data-efficient and numerically stable on small folds, at the cost
  of assuming a sigmoidal shape.

Both expose the same ``fit / transform / fitted`` contract so downstream code
can plug either one in. :class:`CalibrationSelector` evaluates both on a
held-out slice by Brier score and returns the winner, which is the strategy
recommended in the Phase 1 brief ("Platt scaling + Isotonic regression").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from app.metrics.calibration import brier_score


class IsotonicCalibrator:
    """One-vs-rest isotonic calibration for a probability vector."""

    kind = "isotonic"

    def __init__(self) -> None:
        self._regs: list[IsotonicRegression] = []
        self._fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> IsotonicCalibrator:
        if probs.ndim == 1:
            probs = np.column_stack([1 - probs, probs])
        n_classes = probs.shape[1]
        self._regs = []
        for k in range(n_classes):
            y = (labels == k).astype(float)
            reg = IsotonicRegression(out_of_bounds="clip")
            reg.fit(probs[:, k], y)
            self._regs.append(reg)
        self._fitted = True
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return probs
        single = probs.ndim == 1
        if single:
            probs = np.column_stack([1 - probs, probs])
        out = np.zeros_like(probs, dtype=float)
        for k, reg in enumerate(self._regs):
            out[:, k] = reg.transform(probs[:, k])
        # Renormalise to sum to 1.
        s = out.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        out /= s
        return out[:, 1] if single else out

    @property
    def fitted(self) -> bool:
        return self._fitted


class PlattCalibrator:
    """One-vs-rest Platt scaling (per-class logistic regression).

    For each class ``k`` we fit a 1-D logistic regression on the logit of the
    model's raw probability, mapping it onto the empirical hit rate. This is
    the classic Platt / sigmoid recalibration — parametric, stable, and a
    good foil to isotonic on small datasets.
    """

    kind = "platt"

    def __init__(self) -> None:
        self._regs: list[LogisticRegression] = []
        self._fitted = False

    @staticmethod
    def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        p = np.clip(p, eps, 1.0 - eps)
        return np.log(p / (1.0 - p))

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> PlattCalibrator:
        if probs.ndim == 1:
            probs = np.column_stack([1 - probs, probs])
        n_classes = probs.shape[1]
        self._regs = []
        for k in range(n_classes):
            y = (labels == k).astype(int)
            # Degenerate fold: one class is absent, skip fitting and use the
            # empirical base rate as a constant calibration.
            if y.sum() == 0 or y.sum() == len(y):
                reg = _ConstantLogReg(y.mean())
            else:
                reg = LogisticRegression(solver="lbfgs", max_iter=200)
                reg.fit(self._logit(probs[:, k]).reshape(-1, 1), y)
            self._regs.append(reg)
        self._fitted = True
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return probs
        single = probs.ndim == 1
        if single:
            probs = np.column_stack([1 - probs, probs])
        out = np.zeros_like(probs, dtype=float)
        for k, reg in enumerate(self._regs):
            z = self._logit(probs[:, k]).reshape(-1, 1)
            if isinstance(reg, _ConstantLogReg):
                out[:, k] = reg.rate
            else:
                out[:, k] = reg.predict_proba(z)[:, 1]
        s = out.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        out /= s
        return out[:, 1] if single else out

    @property
    def fitted(self) -> bool:
        return self._fitted


@dataclass
class _ConstantLogReg:
    """Fallback used when a class is absent from the calibration fold."""

    rate: float


@dataclass
class CalibrationChoice:
    """Outcome of :class:`CalibrationSelector.fit` — carries both the winning
    calibrator and the Brier scores used to pick it, so callers can log them."""

    calibrator: IsotonicCalibrator | PlattCalibrator
    kind: str
    brier_raw: float
    brier_isotonic: float
    brier_platt: float
    n_holdout: int


class CalibrationSelector:
    """Fit both isotonic and Platt, keep the one with the lower Brier score.

    The comparison is intentionally made on **held-out** predictions: the
    (already out-of-fold) input is chunked with a time-ordered K-fold split,
    each calibrator is fit on ``K-1`` chunks and scored on the last, and the
    averaged Brier is what drives selection. Only after the winner is picked
    do we refit it on the full input so the final calibration is as
    data-efficient as possible.

    This avoids the classic pitfall where isotonic wins every in-sample
    comparison simply because it can interpolate the training set.
    """

    def __init__(self, n_splits: int = 3) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.choice: CalibrationChoice | None = None

    # -- helpers -------------------------------------------------------

    @staticmethod
    def _make_calibrator(kind: str) -> IsotonicCalibrator | PlattCalibrator:
        return IsotonicCalibrator() if kind == "isotonic" else PlattCalibrator()

    def _held_out_brier(self, kind: str, probs: np.ndarray, labels: np.ndarray) -> float:
        """Average Brier across a K-fold time-ordered split.

        Falls back to an in-sample score only when the input is too small to
        build a meaningful held-out split — in that regime the selector
        result is advisory anyway.
        """
        n = len(labels)
        # Need every fold to see every class. Give up and fall back if that's
        # impossible (tiny inputs, severe class imbalance).
        if n < 2 * self.n_splits:
            cal = self._make_calibrator(kind).fit(probs, labels)
            return float(brier_score(cal.transform(probs), labels).score)

        fold_size = n // self.n_splits
        briers: list[float] = []
        for k in range(self.n_splits):
            va_start = k * fold_size
            va_end = (k + 1) * fold_size if k < self.n_splits - 1 else n
            tr_mask = np.ones(n, dtype=bool)
            tr_mask[va_start:va_end] = False
            if len(np.unique(labels[tr_mask])) < 2:
                continue
            cal = self._make_calibrator(kind).fit(probs[tr_mask], labels[tr_mask])
            preds = cal.transform(probs[va_start:va_end])
            briers.append(
                float(brier_score(preds, labels[va_start:va_end]).score)
            )
        if not briers:
            cal = self._make_calibrator(kind).fit(probs, labels)
            return float(brier_score(cal.transform(probs), labels).score)
        return float(np.mean(briers))

    # -- entry ---------------------------------------------------------

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> CalibrationChoice:
        if probs.ndim != 2:
            raise ValueError("probs must be 2-D")

        raw_brier = float(brier_score(probs, labels).score)
        iso_brier = self._held_out_brier("isotonic", probs, labels)
        platt_brier = self._held_out_brier("platt", probs, labels)

        kind = "isotonic" if iso_brier <= platt_brier else "platt"
        # Refit the winner on the full input so the deployed calibrator is as
        # data-efficient as possible; the held-out scores are only used for
        # model selection.
        winner = self._make_calibrator(kind).fit(probs, labels)

        self.choice = CalibrationChoice(
            calibrator=winner,
            kind=kind,
            brier_raw=raw_brier,
            brier_isotonic=iso_brier,
            brier_platt=platt_brier,
            n_holdout=int(len(labels)),
        )
        return self.choice
