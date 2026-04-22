"""Probabilistic evaluation helpers used by the walk-forward backtester.

All functions operate on held-out (out-of-fold) predictions only — the caller is
responsible for never feeding in-sample predictions here.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np


@dataclass(frozen=True)
class BrierResult:
    """Multi-class Brier score (mean squared error in probability space)."""

    score: float
    n: int


def brier_score(probs: np.ndarray, labels: np.ndarray) -> BrierResult:
    """Compute the multi-class Brier score.

    `probs` has shape (n, k); `labels` is an int array of length n with values
    in [0, k). Lower is better; 0.0 is a perfect forecaster.
    """
    if probs.ndim != 2:
        raise ValueError("probs must be 2-D")
    n, k = probs.shape
    if labels.shape != (n,):
        raise ValueError("labels shape mismatch")
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), labels] = 1.0
    return BrierResult(score=float(((probs - one_hot) ** 2).sum(axis=1).mean()), n=n)


def log_loss_multi(probs: np.ndarray, labels: np.ndarray, *, eps: float = 1e-12) -> float:
    n = probs.shape[0]
    clipped = np.clip(probs, eps, 1.0 - eps)
    return float(-sum(log(clipped[i, labels[i]]) for i in range(n)) / n)


def reliability_bins(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 10,
    positive_class: int = 0,
) -> list[dict[str, float]]:
    """Split forecasts for `positive_class` into bins and report empirical rate.

    Returns a list of dicts with ``bin`` (midpoint), ``predicted`` (mean of p̂
    in the bin), ``observed`` (empirical frequency of the positive class in
    the bin), and ``count``.
    """
    if probs.ndim != 2:
        raise ValueError("probs must be 2-D")
    p = probs[:, positive_class]
    y = (labels == positive_class).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        rows.append(
            {
                "bin": round((lo + hi) / 2, 4),
                "predicted": round(float(p[mask].mean()), 4),
                "observed": round(float(y[mask].mean()), 4),
                "count": int(mask.sum()),
            }
        )
    return rows
