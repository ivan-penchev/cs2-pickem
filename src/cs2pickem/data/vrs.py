"""Valve Regional Standings (VRS) from github.com/ValveSoftware/counter-strike_regional_standings.

Plain HTTP (raw.githubusercontent.com) – no scraping needed. We parse:

* the global standings markdown → ``standings`` table (rank, points, team, roster, details file)
* per-team details markdown → ``matches`` table (6-month series history incl. roster per match)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx
import pandas as pd

from .cache import get_text

REPO = "ValveSoftware/counter-strike_regional_standings"
RAW = f"https://raw.githubusercontent.com/{REPO}/main"
API = f"https://api.github.com/repos/{REPO}/contents"

_ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*\[details\]\(([^)]+)\)")
_MATCH = re.compile(
    r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*([^|]+?)\s*\|\s*([WL])\s*\|\s*([\d.]+)\s*\|"
    r"(?:[^|]*\|){4}\s*(-?[\d.]+)\s*\|\s*([^|]+?)\s*\|"
)


@dataclass
class VrsFile:
    year: str
    date: str  # e.g. 2026_11_02

    @property
    def standings_url(self) -> str:
        return f"{RAW}/live/{self.year}/standings_global_{self.date}.md"

    def details_url(self, rel: str) -> str:
        return f"{RAW}/live/{self.year}/{rel}"


def latest_standings(year: str, refresh: bool = False) -> VrsFile:
    """Find the newest ``standings_global_<date>.md`` in ``live/<year>`` via the GitHub contents API."""
    txt = get_text(
        f"{API}/live/{year}",
        ttl_hours=6,
        refresh=refresh,
        namespace="vrs",
        headers={"Accept": "application/vnd.github+json"},
    )
    import json

    names = [e["name"] for e in json.loads(txt) if e["name"].startswith("standings_global_")]
    if not names:
        raise RuntimeError("no global standings found")
    date = sorted(n.removeprefix("standings_global_").removesuffix(".md") for n in names)[-1]
    return VrsFile(year, date)


def parse_standings(md: str) -> pd.DataFrame:
    rows = []
    for line in md.splitlines():
        m = _ROW.match(line)
        if m:
            rows.append(
                {
                    "rank": int(m.group(1)),
                    "points": int(m.group(2)),
                    "team": m.group(3).strip(),
                    "roster": ", ".join(p.strip() for p in m.group(4).split(",")),
                    "details": m.group(5).strip(),
                }
            )
    return pd.DataFrame(rows)


def parse_details(md: str, team: str) -> pd.DataFrame:
    rows = []
    for line in md.splitlines():
        m = _MATCH.match(line)
        if m:
            rows.append(
                {
                    "team": team,
                    "date": m.group(3),
                    "opponent": m.group(4).strip(),
                    "result": m.group(5),
                    "vrs_match_id": int(m.group(2)),
                    "age_weight": float(m.group(6)),
                    "h2h_adj": float(m.group(7)),
                    "roster": ", ".join(p.strip() for p in m.group(8).split(",")),
                }
            )
    return pd.DataFrame(rows)


def fetch(
    year: str, teams: list[str], date: str | None = None, refresh: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (standings, matches, urls) for the given VRS team names.

    Only the first (highest-ranked) occurrence of a team name is used – academy/duplicate entries are ignored.
    """
    vf = VrsFile(year, date) if date else latest_standings(year, refresh)
    std = parse_standings(get_text(vf.standings_url, ttl_hours=24 * 30, refresh=refresh, namespace="vrs"))
    std = std.drop_duplicates("team", keep="first").reset_index(drop=True)
    std.insert(0, "vrs_date", vf.date)
    urls = [vf.standings_url]
    frames = []
    for t in teams:
        row = std[std.team == t]
        if row.empty:
            continue
        url = vf.details_url(row.iloc[0]["details"])
        try:
            frames.append(
                parse_details(get_text(url, ttl_hours=24 * 30, refresh=refresh, namespace="vrs"), t)
            )
            urls.append(url)
        except httpx.HTTPError:
            continue
    matches = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return std, matches, urls
