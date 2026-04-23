"""football-data.co.uk CSV loader (cold-start historical data).

http://www.football-data.co.uk publishes free historical CSVs for top-5
European leagues (and several smaller ones) going back 20+ seasons. Each row
includes final score plus aggregate per-match statistics such as shots,
corners and cards — exactly what we need to bootstrap the corners / cards
totals model without consuming API-Football quota.

CSV columns we rely on (they've been stable for many years)::

    Date (dd/mm/yy or dd/mm/yyyy), HomeTeam, AwayTeam,
    FTHG, FTAG, FTR, HTHG, HTAG, HTR,
    HS,  AS,  HST, AST, HF, AF, HC, AC, HY, AY, HR, AR

Missing columns are tolerated (older seasons sometimes lack stats).

The loader is an adapter that returns an :class:`IngestionResult` so the
existing orchestrator can persist matches + stats via the usual pipeline.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.ingestion.base import BaseSource, IngestionResult, RawMatch, RawStat
from app.ingestion.team_names import normalise_team_name
from app.logging import get_logger

log = get_logger(__name__)


# Mapping from football-data.co.uk league codes → our internal competition codes.
# Not exhaustive — just the ones we care about for corners/cards training.
COMPETITION_CODE_MAP: dict[str, tuple[str, str]] = {
    "E0": ("eng-premier-league", "Premier League"),
    "SP1": ("esp-la-liga", "La Liga"),
    "I1": ("ita-serie-a", "Serie A"),
    "D1": ("ger-bundesliga", "Bundesliga"),
    "F1": ("fra-ligue-1", "Ligue 1"),
    "N1": ("ned-eredivisie", "Eredivisie"),
    "P1": ("por-primeira-liga", "Primeira Liga"),
}


def _parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _parse_date(value: str) -> datetime | None:
    """football-data.co.uk has mixed date formats across seasons."""
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(value.strip(), fmt).replace(tzinfo=UTC)
        except (TypeError, ValueError):
            continue
    return None


def parse_csv_rows(
    rows: Iterable[dict[str, Any]],
    *,
    league_code: str,
    season_tag: str | None = None,
) -> IngestionResult:
    """Parse an iterable of CSV rows (as dicts) into an IngestionResult.

    Unknown / malformed rows are skipped silently (with a log line at debug
    level). Each row produces 1 :class:`RawMatch` plus up to 2 :class:`RawStat`.
    """
    competition_code, competition_name = COMPETITION_CODE_MAP.get(
        league_code, (league_code.lower(), league_code)
    )
    matches: list[RawMatch] = []
    stats: list[RawStat] = []
    for row in rows:
        date = _parse_date(row.get("Date", ""))
        home = (row.get("HomeTeam") or "").strip()
        away = (row.get("AwayTeam") or "").strip()
        if date is None or not home or not away:
            continue
        home = normalise_team_name(home)
        away = normalise_team_name(away)

        fthg = _parse_int(row.get("FTHG"))
        ftag = _parse_int(row.get("FTAG"))
        status = "finished" if fthg is not None and ftag is not None else "scheduled"

        external_id = (
            f"fdcouk:{league_code}:{date.strftime('%Y%m%d')}:"
            f"{home.lower().replace(' ', '_')}-{away.lower().replace(' ', '_')}"
        )
        matches.append(
            RawMatch(
                external_id=external_id,
                sport_code="football",
                competition_code=competition_code,
                competition_name=competition_name,
                home_team=home,
                away_team=away,
                kickoff=date,
                status=status,
                home_score=fthg,
                away_score=ftag,
                season=season_tag,
            )
        )
        if status != "finished":
            continue

        hs = _parse_int(row.get("HS"))
        as_ = _parse_int(row.get("AS"))
        hst = _parse_int(row.get("HST"))
        ast = _parse_int(row.get("AST"))
        hf = _parse_int(row.get("HF"))
        af = _parse_int(row.get("AF"))
        hc = _parse_int(row.get("HC"))
        ac = _parse_int(row.get("AC"))
        hy = _parse_int(row.get("HY"))
        ay = _parse_int(row.get("AY"))
        hr = _parse_int(row.get("HR"))
        ar = _parse_int(row.get("AR"))

        stats.append(
            RawStat(
                match_external_id=external_id,
                team_name=home,
                shots=hs,
                shots_on_target=hst,
                corners=hc,
                yellow_cards=hy,
                red_cards=hr,
                fouls=hf,
            )
        )
        stats.append(
            RawStat(
                match_external_id=external_id,
                team_name=away,
                shots=as_,
                shots_on_target=ast,
                corners=ac,
                yellow_cards=ay,
                red_cards=ar,
                fouls=af,
            )
        )

    return IngestionResult(
        matches=matches,
        stats=stats,
        source="football_data_csv",
        meta={"league_code": league_code, "season": season_tag},
    )


def parse_csv_file(
    path: str | Path,
    *,
    league_code: str | None = None,
    season_tag: str | None = None,
) -> IngestionResult:
    """Load a single CSV file and return the parsed :class:`IngestionResult`.

    If ``league_code`` is not given, we try to infer it from the filename
    (e.g. ``E0_2023-24.csv`` → ``E0``).
    """
    path = Path(path)
    if league_code is None:
        stem = path.stem
        # support names like "E0", "E0_2023", "E0-2023-24"
        for sep in ("_", "-"):
            if sep in stem:
                candidate = stem.split(sep)[0]
                if candidate in COMPETITION_CODE_MAP:
                    league_code = candidate
                    break
        if league_code is None:
            league_code = stem.upper()
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    result = parse_csv_rows(rows, league_code=league_code, season_tag=season_tag)
    log.info(
        "fdcouk.parsed",
        file=str(path),
        matches=len(result.matches),
        stats=len(result.stats),
    )
    return result


class FootballDataCSVSource(BaseSource):
    """Ingestion adapter reading football-data.co.uk CSVs from a directory.

    Enable by setting :envvar:`FOOTBALL_DATA_CSV_DIR` to a directory containing
    one or more ``*.csv`` files from https://www.football-data.co.uk/. Each
    filename should start with the league code (``E0``, ``I1``, ``SP1`` …).
    """

    name = "football-data-csv"
    sports = ("football",)

    def __init__(self, csv_dir: str | None = None) -> None:
        self.csv_dir = csv_dir or os.environ.get("FOOTBALL_DATA_CSV_DIR")

    def is_enabled(self) -> bool:
        return bool(self.csv_dir and Path(self.csv_dir).is_dir())

    def fetch(self, *, season: str | None = None) -> IngestionResult:
        if not self.csv_dir:
            return IngestionResult(source=self.name)
        root = Path(self.csv_dir)
        matches: list[RawMatch] = []
        stats: list[RawStat] = []
        files = sorted(root.glob("*.csv"))
        for f in files:
            try:
                partial = parse_csv_file(f, season_tag=season)
            except Exception as exc:  # noqa: BLE001 — log and keep going
                log.warning("fdcouk.parse_failed", file=str(f), error=str(exc))
                continue
            matches.extend(partial.matches)
            stats.extend(partial.stats)
        return IngestionResult(
            matches=matches,
            stats=stats,
            source=self.name,
            meta={"files": [f.name for f in files]},
        )


__all__ = [
    "FootballDataCSVSource",
    "parse_csv_file",
    "parse_csv_rows",
    "COMPETITION_CODE_MAP",
]
