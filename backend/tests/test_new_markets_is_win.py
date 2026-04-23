"""Result evaluation for the Poisson-derived advanced markets."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from app.backtesting.engine import _is_win


def _fake_match(home: int, away: int) -> SimpleNamespace:
    return SimpleNamespace(
        home_score=home,
        away_score=away,
        kickoff=datetime(2025, 1, 1, tzinfo=UTC),
    )


def test_is_win_double_chance() -> None:
    m_1_0 = _fake_match(1, 0)
    m_0_2 = _fake_match(0, 2)
    m_1_1 = _fake_match(1, 1)

    assert _is_win(m_1_0, "double_chance", "1x", None)
    assert not _is_win(m_0_2, "double_chance", "1x", None)
    assert _is_win(m_1_1, "double_chance", "1x", None)

    assert _is_win(m_1_0, "double_chance", "12", None)
    assert _is_win(m_0_2, "double_chance", "12", None)
    assert not _is_win(m_1_1, "double_chance", "12", None)

    assert _is_win(m_0_2, "double_chance", "x2", None)
    assert _is_win(m_1_1, "double_chance", "x2", None)
    assert not _is_win(m_1_0, "double_chance", "x2", None)


def test_is_win_correct_score() -> None:
    m = _fake_match(2, 1)
    assert _is_win(m, "correct_score", "2-1", None)
    assert not _is_win(m, "correct_score", "1-1", None)
    # Degenerate / malformed selection must not crash — returns False.
    assert not _is_win(m, "correct_score", "abc", None)
    assert not _is_win(m, "correct_score", "2", None)


def test_is_win_no_result_yet() -> None:
    pending = SimpleNamespace(home_score=None, away_score=None, kickoff=datetime(2025, 1, 1, tzinfo=UTC))
    assert not _is_win(pending, "correct_score", "0-0", None)
    assert not _is_win(pending, "double_chance", "1x", None)
