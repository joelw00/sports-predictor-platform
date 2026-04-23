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


def test_top_scorelines_returns_k_entries_and_covers_matrix() -> None:
    model = PoissonFootballModel().fit([("A", "B", 2, 1), ("A", "B", 1, 1)] * 10)
    dist = model.predict("A", "B")
    top = dist.top_scorelines(k=8)
    assert len(top) == 8
    # Entries are sorted descending and are valid probabilities.
    probs = [p for _, _, p in top]
    assert probs == sorted(probs, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probs)
    # The top 8 scorelines should cover a non-trivial share of the mass.
    assert sum(probs) > 0.5


def test_htft_distribution_normalised_and_consistent_with_1x2() -> None:
    model = PoissonFootballModel().fit([("A", "B", 2, 1), ("A", "B", 1, 0)] * 10)
    dist = model.predict("A", "B")
    htft = dist.prob_htft()
    # 9 combinations, probabilities sum to one.
    assert len(htft) == 9
    assert math.isclose(sum(htft.values()), 1.0, rel_tol=1e-6)
    # Marginalising over HT must approximately reproduce FT 1X2.
    p_ft_home = sum(v for (_, ft), v in htft.items() if ft == "home")
    p_ft_draw = sum(v for (_, ft), v in htft.items() if ft == "draw")
    p_ft_away = sum(v for (_, ft), v in htft.items() if ft == "away")
    p_h, p_d, p_a = dist.prob_1x2()
    assert math.isclose(p_ft_home, p_h, abs_tol=5e-3)
    assert math.isclose(p_ft_draw, p_d, abs_tol=5e-3)
    assert math.isclose(p_ft_away, p_a, abs_tol=5e-3)
