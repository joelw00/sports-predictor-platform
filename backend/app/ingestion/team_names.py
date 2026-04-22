"""Small team-name normaliser used to dedup rows that come from different sources.

The Odds API often uses full club names (``Manchester City FC``) while
Football-Data.org uses a slightly different form (``Manchester City FC`` or
``Man City``). We strip common suffixes and apply a handful of aliases so a
single match coming from two sources ends up on the same ``teams`` row
after the orchestrator upsert.

Kept deliberately small: the goal is to cover the Premier League + top-5
European leagues, not to be a comprehensive club registry.
"""

from __future__ import annotations

_SUFFIXES = (
    " FC",
    " F.C.",
    " CF",
    " C.F.",
    " AC",
    " A.C.",
    " AFC",
    " A.F.C.",
    " SC",
    " S.C.",
    " SSD",
    " BC",
    " SRL",
    " SpA",
    " U.S.",
    " 1893",
    " 1907",
)

# Short-form → canonical. Populate when we observe a real cross-source clash.
_ALIASES: dict[str, str] = {
    "man city": "Manchester City",
    "man utd": "Manchester United",
    "man united": "Manchester United",
    "spurs": "Tottenham Hotspur",
    "wolves": "Wolverhampton Wanderers",
    "inter": "Inter Milan",
    "internazionale": "Inter Milan",
    "internazionale milano": "Inter Milan",
    "milan": "AC Milan",
    "psg": "Paris Saint-Germain",
    "paris saint germain": "Paris Saint-Germain",
    "bayern": "Bayern Munich",
    "bayern münchen": "Bayern Munich",
    "bayern munchen": "Bayern Munich",
    "atleti": "Atletico Madrid",
    "atlético madrid": "Atletico Madrid",
    "atletico de madrid": "Atletico Madrid",
    "real": "Real Madrid",
    "barça": "Barcelona",
    "barca": "Barcelona",
    "fc barcelona": "Barcelona",
}


def normalise_team_name(name: str) -> str:
    cleaned = (name or "").strip()
    if not cleaned:
        return cleaned
    # Strip trailing suffixes iteratively.
    changed = True
    while changed:
        changed = False
        for suffix in _SUFFIXES:
            if cleaned.lower().endswith(suffix.lower()):
                cleaned = cleaned[: -len(suffix)].rstrip(" .,-")
                changed = True
    # Alias map (case-insensitive).
    alias = _ALIASES.get(cleaned.lower())
    if alias is not None:
        return alias
    return cleaned


__all__ = ["normalise_team_name"]
