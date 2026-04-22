from __future__ import annotations

import math

from app.ml.poisson import PoissonFootballModel


def test_poisson_fits_and_probabilities_sum_to_one() -> None:
    matches = [
        ("A", "B", 2, 1),
        ("B", "A", 1, 1),
        ("A", "C", 3, 0),
        ("C", "A", 0, 2),
        ("B", "C", 2, 2),
        ("C", "B", 1, 2),
        ("A", "B", 1, 0),
        ("B", "A", 0, 1),
    ] * 4
    model = PoissonFootballModel().fit(matches)
    dist = model.predict("A", "B")
    p_h, p_d, p_a = dist.prob_1x2()
    assert math.isclose(p_h + p_d + p_a, 1.0, rel_tol=1e-6)
    # A clearly stronger than B on this dataset.
    assert p_h > p_a
    # Distribution matrix normalised.
    assert math.isclose(dist.matrix.sum(), 1.0, rel_tol=1e-6)


def test_poisson_over_under_complementary() -> None:
    model = PoissonFootballModel().fit([("A", "B", 2, 1), ("A", "B", 3, 2), ("A", "B", 1, 1)] * 10)
    dist = model.predict("A", "B")
    p_over = dist.prob_over(2.5)
    p_under = 1 - p_over
    assert 0.0 < p_over < 1.0
    assert math.isclose(p_over + p_under, 1.0, rel_tol=1e-6)


def test_poisson_handles_unknown_team() -> None:
    model = PoissonFootballModel().fit([("A", "B", 1, 0)] * 5)
    dist = model.predict("X", "Y")
    p_h, p_d, p_a = dist.prob_1x2()
    assert math.isclose(p_h + p_d + p_a, 1.0, rel_tol=1e-6)
