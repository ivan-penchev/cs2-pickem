"""Assemble model inputs from committed snapshots + user-editable CSVs in data/.

Precedence (highest first): ``data/*.csv`` manual rows  >  latest snapshot  >  nothing (component skipped).

Manual files (all optional, see data/README or DATA_SOURCES.md):

* ``teams.csv``            team, vrs_points, hltv_points           – override/fill strength inputs
* ``matches.csv``          date, team, opponent, result[, weight]   – extra series results (W/L) for form
* ``maps.csv``             team, map, wins, losses[, source]        – per-map record overrides
* ``ratings_override.csv`` team, delta[, note]                      – gut-feel Elo adjustment (+/- points)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import EventConfig
from .paths import data_dir
from .snapshots import load_table as _load_table


@dataclass
class Inputs:
    vrs_standings: pd.DataFrame  # team(canonical), vrs_team, points, rank  (all teams in VRS)
    hltv_ranking: pd.DataFrame  # team(canonical), points
    matches: pd.DataFrame  # team, date, opponent, result (1/0), roster, weight, source
    maps: pd.DataFrame  # team, map, wins, losses
    overrides: pd.DataFrame  # team, delta, note
    notes: list[str] = field(default_factory=list)


def _manual_csv(name: str) -> pd.DataFrame | None:
    p = data_dir() / name
    if not p.exists():
        return None
    df = pd.read_csv(p, comment="#", skipinitialspace=True)
    return df if not df.empty else None


def load_inputs(
    cfg: EventConfig, cutoff: str | None = None, extra_matches: pd.DataFrame | None = None
) -> Inputs:
    """Assemble inputs. ``cutoff`` (YYYY-MM-DD) drops every match on/after that day (no leakage in
    backtests); ``extra_matches`` (team, date, opponent, result[, weight]) are appended, e.g. earlier-stage
    results of the event being backtested."""
    notes: list[str] = []
    canon = cfg.canonical
    days = cfg.snapshot_days

    def load_table(source: str, table: str) -> pd.DataFrame | None:
        return _load_table(source, table, days.get(source))

    def _manual(name: str) -> pd.DataFrame | None:
        return _manual_csv(name) if cfg.use_manual_data else None

    # ---- VRS standings
    std = load_table("vrs", "standings")
    if std is None:
        std = pd.DataFrame(columns=["rank", "points", "team"])
        notes.append("no VRS snapshot – run `cs2pickem fetch` or fill data/teams.csv")
    std = std.rename(columns={"team": "vrs_team"})
    std["team"] = std["vrs_team"].map(canon)
    std = std.drop_duplicates("team", keep="first")[["team", "vrs_team", "rank", "points"]]

    # ---- HLTV ranking
    hr = load_table("hltv", "ranking")
    if hr is None:
        hr = pd.DataFrame(columns=["team", "points"])
        notes.append("no HLTV snapshot – HLTV component uses data/teams.csv only")
    hr = hr.assign(team=hr["team"].map(canon))[["team", "points"]].drop_duplicates("team")

    # ---- manual team overrides
    man = _manual("teams.csv")
    if man is not None:
        man["team"] = man["team"].map(canon)
        for _, r in man.iterrows():
            if pd.notna(r.get("vrs_points")):
                std = std[std.team != r.team]
                std = pd.concat(
                    [
                        std,
                        pd.DataFrame(
                            [
                                {
                                    "team": r.team,
                                    "vrs_team": r.team,
                                    "rank": None,
                                    "points": float(r.vrs_points),
                                }
                            ]
                        ),
                    ]
                )
            if pd.notna(r.get("hltv_points")):
                hr = hr[hr.team != r.team]
                hr = pd.concat([hr, pd.DataFrame([{"team": r.team, "points": float(r.hltv_points)}])])

    # ---- match history: VRS details (series) + HLTV recent results newer than the VRS cut-off
    frames = []
    vm = load_table("vrs", "matches")
    if vm is not None and not vm.empty:
        t = vm["team_canonical"] if "team_canonical" in vm else vm["team"].map(canon)
        frames.append(
            pd.DataFrame(
                {
                    "team": t.map(canon),
                    "date": vm["date"],
                    "opponent": vm["opponent"].map(canon),
                    "result": (vm["result"] == "W").astype(float),
                    "roster": vm["roster"],
                    "weight": 1.0,
                    "source": "vrs",
                }
            )
        )
    hres = load_table("hltv", "results")
    if hres is not None and not hres.empty:
        cutoff = vm["date"].max() if vm is not None and not vm.empty else "0000"
        h = hres[(hres["date"] > cutoff) & (hres["maps_won"] != hres["maps_lost"])]
        frames.append(
            pd.DataFrame(
                {
                    "team": h["team"].map(canon),
                    "date": h["date"],
                    "opponent": h["opponent"].map(canon),
                    "result": (h["maps_won"] > h["maps_lost"]).astype(float),
                    "roster": None,
                    "weight": 1.0,
                    "source": "hltv",
                }
            )
        )
    mm = _manual("matches.csv")
    if mm is not None:
        frames.append(
            pd.DataFrame(
                {
                    "team": mm["team"].map(canon),
                    "date": mm["date"].astype(str),
                    "opponent": mm["opponent"].map(canon),
                    "result": mm["result"].astype(str).str.upper().isin(["W", "1", "WIN"]).astype(float),
                    "roster": None,
                    "weight": mm["weight"] if "weight" in mm else 1.0,
                    "source": "manual",
                }
            )
        )
    matches = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["team", "date", "opponent", "result", "roster", "weight", "source"])
    )
    if extra_matches is not None and not extra_matches.empty:
        em = extra_matches.copy()
        em["team"], em["opponent"] = em["team"].map(canon), em["opponent"].map(canon)
        for col, val in (("roster", None), ("weight", 1.0), ("source", "extra")):
            if col not in em:
                em[col] = val
        matches = pd.concat([matches, em[list(matches.columns)]], ignore_index=True)
    matches = matches.drop_duplicates(["team", "date", "opponent", "source"])
    if cutoff:
        matches = matches[matches["date"].astype(str) < cutoff].reset_index(drop=True)

    # ---- maps
    mp = load_table("hltv", "maps")
    mp = (
        mp[["team", "map", "wins", "losses"]].copy()
        if mp is not None and not mp.empty
        else pd.DataFrame(columns=["team", "map", "wins", "losses"])
    )
    mp["team"] = mp["team"].map(canon)
    mman = _manual("maps.csv")
    if mman is not None:
        mman["team"] = mman["team"].map(canon)
        key = set(zip(mman.team, mman.map, strict=True))
        mp = mp[[(a, b) not in key for a, b in zip(mp.team, mp.map, strict=True)]]
        mp = pd.concat([mp, mman[["team", "map", "wins", "losses"]]], ignore_index=True)

    ov = _manual("ratings_override.csv")
    if ov is None:
        ov = pd.DataFrame(columns=["team", "delta", "note"])
    ov["team"] = ov["team"].map(canon)

    return Inputs(std, hr, matches, mp, ov, notes)
