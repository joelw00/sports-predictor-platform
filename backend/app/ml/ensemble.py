from __future__ import annotations

import numpy as np


class EnsembleFootballModel:
    """Weighted convex combination of calibrated probability vectors."""

    def __init__(self, weights: tuple[float, float] = (0.6, 0.4)) -> None:
        total = sum(weights)
        self._weights = tuple(w / total for w in weights)

    def blend(self, *prob_vectors: np.ndarray) -> np.ndarray:
        if len(prob_vectors) != len(self._weights):
            raise ValueError("weights / prob vector count mismatch")
        stacked = np.stack(prob_vectors, axis=0)
        return np.average(stacked, axis=0, weights=self._weights)

    @property
    def weights(self) -> tuple[float, ...]:
        return self._weights
