"""Deterministic synthetic data source.

Generates realistic football fixtures, match stats and odds so the whole
pipeline runs end-to-end without any external API. The random seed is fixed
so the same universe is produced every time.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from math import exp

from app.ingestion.base import BaseSource, IngestionResult, RawMatch, RawOdds, RawStat

# ---------------------------------------------------------------------------
# League catalogue — a subset that spans the major European markets plus MLS.
# Each team has a latent "strength" used to generate realistic results/odds.
# ---------------------------------------------------------------------------

_LEAGUES: dict[str, tuple[str, str, list[tuple[str, float]]]] = {
    "ita-serie-a": (
        "Serie A",
        "Italy",
        [
            ("Inter", 1.85),
            ("Juventus", 1.72),
            ("Milan", 1.68),
            ("Napoli", 1.70),
            ("Roma", 1.55),
            ("Lazio", 1.52),
            ("Atalanta", 1.60),
            ("Fiorentina", 1.40),
            ("Bologna", 1.35),
            ("Torino", 1.25),
            ("Udinese", 1.15),
            ("Genoa", 1.05),
            ("Cagliari", 1.00),
            ("Lecce", 0.95),
            ("Empoli", 0.90),
            ("Verona", 0.90),
            ("Parma", 1.05),
            ("Monza", 1.00),
            ("Como", 0.95),
            ("Venezia", 0.85),
        ],
    ),
    "eng-premier-league": (
        "Premier League",
        "England",
        [
            ("Manchester City", 2.00),
            ("Arsenal", 1.90),
            ("Liverpool", 1.92),
            ("Chelsea", 1.65),
            ("Tottenham", 1.62),
            ("Manchester United", 1.60),
            ("Newcastle", 1.55),
            ("Aston Villa", 1.50),
            ("Brighton", 1.40),
            ("West Ham", 1.35),
            ("Everton", 1.20),
            ("Fulham", 1.20),
            ("Crystal Palace", 1.15),
            ("Brentford", 1.20),
            ("Wolves", 1.15),
            ("Nottingham Forest", 1.10),
            ("Bournemouth", 1.10),
            ("Leicester", 1.20),
            ("Ipswich", 0.95),
            ("Southampton", 0.90),
        ],
    ),
    "esp-laliga": (
        "LaLiga",
        "Spain",
        [
            ("Real Madrid", 2.05),
            ("Barcelona", 1.95),
            ("Atletico Madrid", 1.78),
            ("Athletic Bilbao", 1.55),
            ("Real Sociedad", 1.45),
            ("Villarreal", 1.40),
            ("Real Betis", 1.38),
            ("Valencia", 1.25),
            ("Sevilla", 1.30),
            ("Girona", 1.45),
            ("Osasuna", 1.15),
            ("Celta Vigo", 1.15),
            ("Mallorca", 1.10),
            ("Getafe", 1.05),
            ("Rayo Vallecano", 1.05),
            ("Las Palmas", 1.00),
            ("Alaves", 0.95),
            ("Leganes", 0.90),
            ("Espanyol", 0.95),
            ("Valladolid", 0.85),
        ],
    ),
    "ger-bundesliga": (
        "Bundesliga",
        "Germany",
        [
            ("Bayern Munich", 2.00),
            ("Bayer Leverkusen", 1.90),
            ("Borussia Dortmund", 1.75),
            ("RB Leipzig", 1.70),
            ("Eintracht Frankfurt", 1.45),
            ("Wolfsburg", 1.35),
            ("Stuttgart", 1.55),
            ("Freiburg", 1.30),
            ("Union Berlin", 1.20),
            ("Mainz", 1.15),
            ("Hoffenheim", 1.15),
            ("Werder Bremen", 1.15),
            ("Augsburg", 1.10),
            ("Borussia M.gladbach", 1.15),
            ("Heidenheim", 1.00),
            ("Bochum", 0.95),
            ("St. Pauli", 0.95),
            ("Holstein Kiel", 0.90),
        ],
    ),
    "fra-ligue-1": (
        "Ligue 1",
        "France",
        [
            ("Paris Saint-Germain", 2.10),
            ("Marseille", 1.60),
            ("Monaco", 1.55),
            ("Lille", 1.50),
            ("Lyon", 1.40),
            ("Nice", 1.35),
            ("Rennes", 1.30),
            ("Lens", 1.30),
            ("Strasbourg", 1.15),
            ("Toulouse", 1.15),
            ("Nantes", 1.10),
            ("Reims", 1.10),
            ("Brest", 1.20),
            ("Montpellier", 1.00),
            ("Le Havre", 0.95),
            ("Auxerre", 1.00),
            ("Angers", 0.95),
            ("St Etienne", 0.95),
        ],
    ),
    "usa-mls": (
        "Major League Soccer",
        "USA",
        [
            ("Inter Miami", 1.60),
            ("LAFC", 1.55),
            ("Columbus Crew", 1.55),
            ("Cincinnati", 1.50),
            ("Philadelphia Union", 1.35),
            ("New York City FC", 1.30),
            ("Seattle Sounders", 1.30),
            ("Atlanta United", 1.25),
            ("Real Salt Lake", 1.20),
            ("Portland Timbers", 1.20),
            ("Houston Dynamo", 1.20),
            ("Orlando City", 1.25),
        ],
    ),
}

# Very small table-tennis roster (demo only) — full coverage lands in Phase 4.
_TT_PLAYERS: list[tuple[str, float]] = [
    ("Fan Zhendong", 2.1),
    ("Ma Long", 2.0),
    ("Wang Chuqin", 1.95),
    ("Tomokazu Harimoto", 1.85),
    ("Hugo Calderano", 1.78),
    ("Darko Jorgic", 1.70),
    ("Truls Moregard", 1.68),
    ("Lin Gaoyuan", 1.72),
    ("Kanak Jha", 1.40),
    ("Simon Gauzy", 1.55),
]


def _poisson_sample(rng: random.Random, lam: float) -> int:
    """Knuth's algorithm — tiny lambdas only, good enough for demo."""
    threshold = exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= threshold:
            return k - 1


class DemoSource(BaseSource):
    name = "demo"
    sports = ("football", "table_tennis")

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def is_enabled(self) -> bool:
        return True

    def fetch(self, *, season: str | None = None) -> IngestionResult:
        rng = random.Random(self._seed)
        result = IngestionResult(source=self.name)
        now = datetime.now(tz=UTC).replace(minute=0, second=0, microsecond=0)

        # Historical window (for feature engineering + training).
        history_start = now - timedelta(days=180)
        for comp_code, (comp_name, country, teams) in _LEAGUES.items():
            self._simulate_football_round_robin(
                rng=rng,
                competition_code=comp_code,
                competition_name=comp_name,
                country=country,
                teams=teams,
                start=history_start,
                end=now - timedelta(days=1),
                season=season or "demo-2025",
                result=result,
                played=True,
            )

        # Upcoming fixtures (today + next 3 days).
        for comp_code, (comp_name, country, teams) in _LEAGUES.items():
            self._simulate_football_round_robin(
                rng=rng,
                competition_code=comp_code,
                competition_name=comp_name,
                country=country,
                teams=teams,
                start=now,
                end=now + timedelta(days=3),
                season=season or "demo-2025",
                result=result,
                played=False,
                matches_per_day=min(4, len(teams) // 2),
            )

        # Table tennis (demo-only, small universe).
        self._simulate_table_tennis(rng, now, result, season or "demo-2025")

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _simulate_football_round_robin(
        self,
        *,
        rng: random.Random,
        competition_code: str,
        competition_name: str,
        country: str,
        teams: list[tuple[str, float]],
        start: datetime,
        end: datetime,
        season: str,
        result: IngestionResult,
        played: bool,
        matches_per_day: int = 5,
    ) -> None:
        days = max(1, int((end - start).total_seconds() // 86400))
        pairings: list[tuple[tuple[str, float], tuple[str, float]]] = []
        for i in range(len(teams)):
            for j in range(len(teams)):
                if i == j:
                    continue
                pairings.append((teams[i], teams[j]))
        rng.shuffle(pairings)

        per_day = matches_per_day if played else matches_per_day
        expected = days * per_day
        pairings = pairings[: min(expected, len(pairings))]

        for idx, ((home_name, home_str), (away_name, away_str)) in enumerate(pairings):
            day_offset = idx // max(1, per_day)
            hour = 14 + (idx % per_day) * 2
            kickoff = (start + timedelta(days=day_offset)).replace(hour=hour)
            external_id = f"{competition_code}:{home_name}-{away_name}:{kickoff.date()}"

            home_adv = 0.25
            lambda_home = max(0.15, home_str + home_adv - 0.5 * away_str + rng.gauss(0, 0.15))
            lambda_away = max(0.15, away_str - 0.5 * home_str + rng.gauss(0, 0.15))

            match = RawMatch(
                external_id=external_id,
                sport_code="football",
                competition_code=competition_code,
                competition_name=competition_name,
                home_team=home_name,
                away_team=away_name,
                kickoff=kickoff,
                status="finished" if played else "scheduled",
                season=season,
            )

            if played:
                hs = _poisson_sample(rng, lambda_home)
                as_ = _poisson_sample(rng, lambda_away)
                match.home_score = hs
                match.away_score = as_
                result.stats.append(
                    RawStat(
                        match_external_id=external_id,
                        team_name=home_name,
                        shots=int(rng.gauss(13, 3)),
                        shots_on_target=int(rng.gauss(5, 2)),
                        corners=int(rng.gauss(5, 2)),
                        yellow_cards=int(rng.gauss(2, 1)),
                        red_cards=1 if rng.random() < 0.05 else 0,
                        fouls=int(rng.gauss(12, 3)),
                        possession=round(rng.gauss(55, 6), 1),
                        xg=round(max(0.1, lambda_home + rng.gauss(0, 0.2)), 2),
                        xga=round(max(0.1, lambda_away + rng.gauss(0, 0.2)), 2),
                        passes=int(rng.gauss(450, 50)),
                        pass_accuracy=round(rng.gauss(82, 4), 1),
                    )
                )
                result.stats.append(
                    RawStat(
                        match_external_id=external_id,
                        team_name=away_name,
                        shots=int(rng.gauss(10, 3)),
                        shots_on_target=int(rng.gauss(4, 2)),
                        corners=int(rng.gauss(4, 2)),
                        yellow_cards=int(rng.gauss(2, 1)),
                        red_cards=1 if rng.random() < 0.05 else 0,
                        fouls=int(rng.gauss(13, 3)),
                        possession=round(100 - rng.gauss(55, 6), 1),
                        xg=round(max(0.1, lambda_away + rng.gauss(0, 0.2)), 2),
                        xga=round(max(0.1, lambda_home + rng.gauss(0, 0.2)), 2),
                        passes=int(rng.gauss(400, 50)),
                        pass_accuracy=round(rng.gauss(78, 4), 1),
                    )
                )

            # Fake closing 1X2 odds implied by the Poisson model (+ 5% overround).
            p_home, p_draw, p_away = self._poisson_1x2(lambda_home, lambda_away)
            overround = 1.05
            for selection, p in (("home", p_home), ("draw", p_draw), ("away", p_away)):
                raw_price = 1.0 / max(0.01, p * overround)
                price = round(raw_price + rng.uniform(-0.08, 0.08), 2)
                result.odds.append(
                    RawOdds(
                        match_external_id=external_id,
                        bookmaker="DemoBook",
                        market="1x2",
                        selection=selection,
                        price=max(1.05, price),
                        captured_at=kickoff - timedelta(hours=2),
                        is_closing=True,
                    )
                )
            # Over/Under 2.5 odds from goal total distribution.
            p_over = self._poisson_over(lambda_home + lambda_away, 2.5)
            for selection, p in (("over", p_over), ("under", 1 - p_over)):
                raw_price = 1.0 / max(0.01, p * 1.04)
                price = round(raw_price + rng.uniform(-0.05, 0.05), 2)
                result.odds.append(
                    RawOdds(
                        match_external_id=external_id,
                        bookmaker="DemoBook",
                        market="over_under",
                        selection=selection,
                        price=max(1.05, price),
                        line=2.5,
                        captured_at=kickoff - timedelta(hours=2),
                        is_closing=True,
                    )
                )

            result.matches.append(match)

    def _simulate_table_tennis(
        self,
        rng: random.Random,
        now: datetime,
        result: IngestionResult,
        season: str,
    ) -> None:
        pairings = [(a, b) for a in _TT_PLAYERS for b in _TT_PLAYERS if a[0] != b[0]]
        rng.shuffle(pairings)
        for idx, ((home_name, home_str), (away_name, away_str)) in enumerate(pairings[:40]):
            day_offset = idx // 8
            kickoff = now + timedelta(days=day_offset, hours=10 + (idx % 8))
            external_id = f"wtt-demo:{home_name}-{away_name}:{kickoff.date()}"
            result.matches.append(
                RawMatch(
                    external_id=external_id,
                    sport_code="table_tennis",
                    competition_code="wtt-demo",
                    competition_name="WTT Demo Tour",
                    home_team=home_name,
                    away_team=away_name,
                    kickoff=kickoff,
                    season=season,
                )
            )
            diff = home_str - away_str
            p_home = 1.0 / (1.0 + exp(-diff * 2.2))
            overround = 1.04
            for selection, p in (("home", p_home), ("away", 1 - p_home)):
                price = round(1.0 / max(0.01, p * overround) + rng.uniform(-0.05, 0.05), 2)
                result.odds.append(
                    RawOdds(
                        match_external_id=external_id,
                        bookmaker="DemoBook",
                        market="match_winner",
                        selection=selection,
                        price=max(1.05, price),
                        captured_at=kickoff - timedelta(hours=1),
                        is_closing=True,
                    )
                )

    # --- Poisson helpers -------------------------------------------------

    @staticmethod
    def _poisson_pmf(k: int, lam: float) -> float:
        from math import factorial

        return (lam**k) * exp(-lam) / factorial(k)

    def _poisson_1x2(self, lh: float, la: float, max_goals: int = 8) -> tuple[float, float, float]:
        p_h = p_d = p_a = 0.0
        for hg in range(max_goals + 1):
            for ag in range(max_goals + 1):
                p = self._poisson_pmf(hg, lh) * self._poisson_pmf(ag, la)
                if hg > ag:
                    p_h += p
                elif hg == ag:
                    p_d += p
                else:
                    p_a += p
        s = p_h + p_d + p_a
        return p_h / s, p_d / s, p_a / s

    def _poisson_over(self, total_lambda: float, line: float, max_goals: int = 12) -> float:
        cutoff = int(line)
        from math import factorial

        p_under = 0.0
        for g in range(cutoff + 1):
            p_under += (total_lambda**g) * exp(-total_lambda) / factorial(g)
        return 1.0 - p_under
