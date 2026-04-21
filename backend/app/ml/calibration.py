from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    """One-vs-rest isotonic calibration for a probability vector."""

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
