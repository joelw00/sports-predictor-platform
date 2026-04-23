"""Helpers around the ``odds_snapshots`` table.

A single match accumulates many rows as prices evolve (opening / mid /
closing). The backtester and CLV analysis consume this history via three
helpers:

- :func:`mark_closing_odds` — flags the latest pre-kickoff snapshot per
  ``(match, bookmaker, market, selection, line)`` with ``is_closing=True``
  once the match has actually started.
- :func:`entry_snapshots` — returns the **earliest** pre-kickoff snapshot
  per key for a given match/market, i.e. the "opening line" view used by
  the backtester as the price the bettor would have taken.
- :func:`closing_price_lookup` — returns a key-indexed map of closing
  prices for a match/market, used to compute CLV for each bet.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from app.db import models as m
from app.logging import get_logger

log = get_logger(__name__)

OddsKey = tuple[str, str, str, float | None]


def mark_closing_odds(
    db: Session,
    *,
    match_ids: Iterable[int] | None = None,
    now: datetime | None = None,
) -> int:
    """Flag the last pre-kickoff snapshot per key as ``is_closing``.

    Safe to call multiple times: the helper always resets previously marked
    rows before re-computing, so re-running on a growing snapshot history
    converges on the real closing line.

    Returns the number of snapshot rows that were (re-)marked as closing.
    """
    now = now or datetime.now(tz=UTC)

    match_q = select(m.Match).where(m.Match.kickoff <= now)
    if match_ids is not None:
        ids = list(match_ids)
        if not ids:
            return 0
        match_q = match_q.where(m.Match.id.in_(ids))

    marked = 0
    for match in db.execute(match_q).scalars():
        # Work snapshot-by-snapshot: for each (book, market, selection, line)
        # the latest captured_at BEFORE kickoff wins. Everything else flips
        # back to False.
        snaps = (
            db.query(m.OddsSnapshot)
            .filter(
                m.OddsSnapshot.match_id == match.id,
                m.OddsSnapshot.captured_at <= match.kickoff,
            )
            .all()
        )
        if not snaps:
            continue
        # Reset any existing closing flags on this match first.
        db.query(m.OddsSnapshot).filter(
            and_(
                m.OddsSnapshot.match_id == match.id,
                m.OddsSnapshot.is_closing.is_(True),
            )
        ).update({m.OddsSnapshot.is_closing: False}, synchronize_session="fetch")

        latest: dict[tuple[str, str, str, float | None], m.OddsSnapshot] = {}
        for snap in snaps:
            key = (snap.bookmaker, snap.market, snap.selection, snap.line)
            existing = latest.get(key)
            if existing is None or snap.captured_at > existing.captured_at:
                latest[key] = snap
        for snap in latest.values():
            snap.is_closing = True
            marked += 1
    db.flush()
    return marked


def entry_snapshots(
    db: Session,
    *,
    match_id: int,
    market: str,
) -> list[m.OddsSnapshot]:
    """Return the earliest pre-kickoff snapshot per ``(book, selection, line)``.

    This is the "opening line" view used by the backtester: for each distinct
    offer we grab the first price ever captured before kickoff. If no
    snapshot exists for a key the key is absent from the returned list.
    """
    match = db.get(m.Match, match_id)
    if match is None:
        return []
    snaps = (
        db.query(m.OddsSnapshot)
        .filter(
            m.OddsSnapshot.match_id == match_id,
            m.OddsSnapshot.market == market,
            m.OddsSnapshot.captured_at <= match.kickoff,
        )
        .order_by(m.OddsSnapshot.captured_at.asc())
        .all()
    )
    earliest: dict[OddsKey, m.OddsSnapshot] = {}
    for snap in snaps:
        key = (snap.bookmaker, snap.market, snap.selection, snap.line)
        if key not in earliest:
            earliest[key] = snap
    return list(earliest.values())


def closing_price_lookup(
    db: Session,
    *,
    match_id: int,
    market: str,
) -> dict[OddsKey, float]:
    """Return ``(bookmaker, market, selection, line) -> closing_price`` for a match.

    Rows are the ones flagged ``is_closing=True`` by :func:`mark_closing_odds`.
    """
    rows = (
        db.query(m.OddsSnapshot)
        .filter(
            m.OddsSnapshot.match_id == match_id,
            m.OddsSnapshot.market == market,
            m.OddsSnapshot.is_closing.is_(True),
        )
        .all()
    )
    return {(r.bookmaker, r.market, r.selection, r.line): r.price for r in rows}


__all__ = ["mark_closing_odds", "entry_snapshots", "closing_price_lookup", "OddsKey"]
