"""Stub implementations for real data sources.

Each stub implements the ``BaseSource`` contract and reports its enabled state
based on configuration. Real ``fetch`` implementations are delivered in Phase 1
(see ``docs/ROADMAP.md``). Today the stubs raise ``NotImplementedError`` when
called while enabled, and the demo fallback keeps the platform fully functional.
"""

from __future__ import annotations

from app.ingestion.base import BaseSource, IngestionResult


class _FlagSource(BaseSource):
    def __init__(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def is_enabled(self) -> bool:
        return self._enabled

    def fetch(self, *, season: str | None = None) -> IngestionResult:
        raise NotImplementedError(
            f"{self.name} adapter is scheduled for Phase 1; keep the credential "
            "unset to stay on the demo adapter for now."
        )


class _KeySource(BaseSource):
    def __init__(self, key: str) -> None:
        self._key = key or ""

    def is_enabled(self) -> bool:
        return bool(self._key)

    def fetch(self, *, season: str | None = None) -> IngestionResult:
        raise NotImplementedError(
            f"{self.name} adapter is scheduled for Phase 1; unset the API key to "
            "stay on the demo adapter for now."
        )


class ApiFootballSource(_KeySource):
    name = "api-football"
    sports = ("football",)


class SofaScoreSource(_FlagSource):
    name = "sofascore"
    sports = ("football", "table_tennis")


class UnderstatSource(_FlagSource):
    name = "understat"
    sports = ("football",)


class SnaiSource(_FlagSource):
    name = "snai"
    sports = ("football", "table_tennis")
