"""Unit tests for the football-data.co.uk CSV ingestion adapter."""

from __future__ import annotations

from pathlib import Path

from app.ingestion.football_data_csv import (
    COMPETITION_CODE_MAP,
    FootballDataCSVSource,
    parse_csv_file,
    parse_csv_rows,
)

SAMPLE_HEADER = (
    "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,"
    "HS,AS,HST,AST,HF,AF,HC,AC,HY,AY,HR,AR"
)


def _row(**overrides: str) -> dict[str, str]:
    base = {
        "Date": "12/08/2023",
        "HomeTeam": "Arsenal",
        "AwayTeam": "Chelsea",
        "FTHG": "2",
        "FTAG": "1",
        "FTR": "H",
        "HTHG": "1",
        "HTAG": "0",
        "HTR": "H",
        "HS": "14",
        "AS": "9",
        "HST": "6",
        "AST": "3",
        "HF": "11",
        "AF": "13",
        "HC": "7",
        "AC": "4",
        "HY": "2",
        "AY": "3",
        "HR": "0",
        "AR": "1",
    }
    base.update(overrides)
    return base


def test_parse_rows_extracts_corners_and_cards() -> None:
    result = parse_csv_rows([_row()], league_code="E0", season_tag="2023-24")
    assert len(result.matches) == 1
    assert len(result.stats) == 2
    match = result.matches[0]
    assert match.sport_code == "football"
    assert match.competition_code == "eng-premier-league"
    assert match.status == "finished"
    assert match.home_score == 2 and match.away_score == 1
    home_stat = next(s for s in result.stats if s.team_name == "Arsenal")
    away_stat = next(s for s in result.stats if s.team_name == "Chelsea")
    assert home_stat.corners == 7
    assert away_stat.corners == 4
    assert home_stat.yellow_cards == 2 and home_stat.red_cards == 0
    assert away_stat.yellow_cards == 3 and away_stat.red_cards == 1


def test_parse_rows_handles_missing_stats_gracefully() -> None:
    row = _row(HC="", AC="", HY="", AY="", HR="", AR="")
    result = parse_csv_rows([row], league_code="E0")
    assert len(result.stats) == 2
    for s in result.stats:
        assert s.corners is None
        assert s.yellow_cards is None
        assert s.red_cards is None


def test_unparseable_rows_are_skipped() -> None:
    result = parse_csv_rows(
        [
            _row(Date="not-a-date"),
            _row(HomeTeam=""),
            _row(Date="12/08/2023"),
        ],
        league_code="I1",
    )
    assert len(result.matches) == 1
    assert result.matches[0].competition_code == "ita-serie-a"


def test_short_year_date_is_parsed() -> None:
    result = parse_csv_rows([_row(Date="12/08/23")], league_code="E0")
    assert len(result.matches) == 1
    assert result.matches[0].kickoff.year == 2023


def test_parse_csv_file_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "E0_2023-24.csv"
    path.write_text(
        SAMPLE_HEADER
        + "\n"
        + "12/08/2023,Arsenal,Chelsea,2,1,H,1,0,H,14,9,6,3,11,13,7,4,2,3,0,1\n"
        + "13/08/2023,Man United,Liverpool,1,3,A,0,2,A,10,12,4,7,12,10,5,6,2,2,0,0\n"
    )
    result = parse_csv_file(path, season_tag="2023-24")
    assert len(result.matches) == 2
    assert result.matches[0].competition_code == COMPETITION_CODE_MAP["E0"][0]
    totals = sorted(
        sum(s.corners for s in result.stats if s.match_external_id == m.external_id and s.corners)
        for m in result.matches
    )
    assert totals == [11, 11]


def test_source_is_disabled_without_dir(tmp_path: Path) -> None:
    assert FootballDataCSVSource(csv_dir=None).is_enabled() is False
    assert FootballDataCSVSource(csv_dir=str(tmp_path / "does-not-exist")).is_enabled() is False


def test_source_fetch_reads_directory(tmp_path: Path) -> None:
    (tmp_path / "E0_2023-24.csv").write_text(
        SAMPLE_HEADER + "\n" + "12/08/2023,Arsenal,Chelsea,2,1,H,1,0,H,14,9,6,3,11,13,7,4,2,3,0,1\n"
    )
    (tmp_path / "I1_2023-24.csv").write_text(
        SAMPLE_HEADER + "\n" + "13/08/2023,Inter,Milan,1,1,D,0,1,A,11,10,5,4,12,11,6,5,3,3,0,0\n"
    )
    source = FootballDataCSVSource(csv_dir=str(tmp_path))
    assert source.is_enabled() is True
    result = source.fetch()
    assert len(result.matches) == 2
    codes = {m.competition_code for m in result.matches}
    assert codes == {"eng-premier-league", "ita-serie-a"}
