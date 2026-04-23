"""Tests for isotonic / Platt calibrators and the selector."""

from __future__ import annotations

import numpy as np

from app.metrics.calibration import brier_score
from app.ml.calibration import (
    CalibrationSelector,
    IsotonicCalibrator,
    PlattCalibrator,
)


def _synthetic_overconfident_three_class(n: int = 800, seed: int = 0):
    """Build a calibration-dataset where the raw probs are systematically
    overconfident: labels come from a tempered distribution, but we expose the
    peaky pre-temperature distribution as the "model output"."""
    rng = np.random.default_rng(seed)
    # Base rates: 0.45 / 0.25 / 0.30 (home / draw / away)
    labels = rng.choice([0, 1, 2], size=n, p=[0.45, 0.25, 0.30])
    peaky = np.zeros((n, 3))
    for i, y in enumerate(labels):
        # Generate a peaky distribution that overcalls the correct class 70%
        # of the time but otherwise mis-attributes probability to another
        # class — a realistic miscalibration profile.
        correct = rng.random() < 0.7
        main = y if correct else int(rng.choice([c for c in (0, 1, 2) if c != y]))
        p = np.full(3, 0.05)
        p[main] = 0.90
        p = p + rng.normal(0, 0.02, size=3)
        p = np.clip(p, 0.01, None)
        p /= p.sum()
        peaky[i] = p
    return peaky, labels


def test_isotonic_reduces_brier():
    probs, labels = _synthetic_overconfident_three_class()
    raw_brier = brier_score(probs, labels).score
    cal = IsotonicCalibrator().fit(probs, labels)
    cal_brier = brier_score(cal.transform(probs), labels).score
    assert cal.fitted
    assert cal_brier < raw_brier


def test_platt_reduces_brier():
    probs, labels = _synthetic_overconfident_three_class()
    raw_brier = brier_score(probs, labels).score
    cal = PlattCalibrator().fit(probs, labels)
    cal_brier = brier_score(cal.transform(probs), labels).score
    assert cal.fitted
    assert cal_brier < raw_brier


def test_platt_handles_degenerate_class_without_crashing():
    """When a class is never observed we must fall back to the empirical rate."""
    probs = np.array(
        [
            [0.5, 0.3, 0.2],
            [0.6, 0.2, 0.2],
            [0.7, 0.1, 0.2],
            [0.4, 0.4, 0.2],
        ]
    )
    labels = np.array([0, 0, 0, 0])  # draw and away are degenerate
    cal = PlattCalibrator().fit(probs, labels)
    out = cal.transform(probs)
    assert out.shape == probs.shape
    assert np.allclose(out.sum(axis=1), 1.0)


def test_calibration_selector_picks_winner():
    probs, labels = _synthetic_overconfident_three_class()
    selector = CalibrationSelector()
    choice = selector.fit(probs, labels)
    assert choice.kind in {"isotonic", "platt"}
    assert choice.brier_raw >= min(choice.brier_isotonic, choice.brier_platt)
    winner_brier = (
        choice.brier_isotonic if choice.kind == "isotonic" else choice.brier_platt
    )
    other_brier = (
        choice.brier_platt if choice.kind == "isotonic" else choice.brier_isotonic
    )
    assert winner_brier <= other_brier


def test_isotonic_binary_roundtrip():
    """Single-column input is automatically expanded to two classes."""
    rng = np.random.default_rng(0)
    p = rng.uniform(size=400)
    y = (rng.uniform(size=400) < p).astype(int)
    cal = IsotonicCalibrator().fit(p, y)
    q = cal.transform(p)
    assert q.shape == p.shape
    assert ((q >= 0) & (q <= 1)).all()
