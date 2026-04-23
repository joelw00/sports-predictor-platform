"""Unit tests for the drift primitives (PSI + KS)."""

from __future__ import annotations

import numpy as np

from app.monitoring.drift import compute_feature_drift, ks_two_sample, psi


def test_psi_zero_for_identical_samples() -> None:
    rng = np.random.default_rng(42)
    ref = rng.normal(size=5_000)
    # Same generator seed → same numbers; PSI must collapse to ~0.
    rng2 = np.random.default_rng(42)
    cur = rng2.normal(size=5_000)
    assert psi(ref, cur) < 1e-9


def test_psi_grows_with_mean_shift() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(loc=0.0, scale=1.0, size=5_000)
    small = rng.normal(loc=0.25, scale=1.0, size=5_000)
    large = rng.normal(loc=1.5, scale=1.0, size=5_000)
    psi_small = psi(ref, small)
    psi_large = psi(ref, large)
    assert psi_small > 0.0
    assert psi_large > psi_small
    # A 1.5σ mean shift on Gaussians must cross the canonical "significant"
    # threshold.
    assert psi_large > 0.25


def test_psi_handles_small_samples_without_crash() -> None:
    assert psi([1.0], [2.0]) == 0.0
    assert psi([], []) == 0.0


def test_ks_detects_mean_shift() -> None:
    rng = np.random.default_rng(1)
    ref = rng.normal(size=1_000)
    cur = rng.normal(loc=1.0, size=1_000)
    stat, p = ks_two_sample(ref, cur)
    assert stat > 0.2
    assert p < 1e-10


def test_ks_identical_distribution_has_high_p() -> None:
    rng = np.random.default_rng(2)
    ref = rng.normal(size=2_000)
    cur = rng.normal(size=2_000)
    _, p = ks_two_sample(ref, cur)
    assert p > 0.01


def test_compute_feature_drift_handles_missing_keys() -> None:
    ref = {"a": [0.1, 0.2, 0.3, 0.4], "b": [1.0, 2.0, 3.0, 4.0]}
    cur = {"a": [0.15, 0.25, 0.35, 0.45], "c": [9.0, 9.0, 9.0]}
    stats = compute_feature_drift(ref, cur)
    # Only "a" is present in both.
    assert [s.feature for s in stats] == ["a"]
    assert stats[0].n_ref == 4
    assert stats[0].n_cur == 4
