"""``cs2pickem fetch``: pull every online source and write dated snapshots under data/snapshots/."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import pandas as pd

from ..config import Event
from ..snapshots import write_snapshot
from . import hltv, vrs


def fetch_vrs(event: Event, refresh: bool = False, log: Callable[[str], None] = print) -> None:
    cfg = event.config
    year = (cfg.vrs_date or str(date.today().year))[:4]
    names = [t.vrs_name or t.name for t in cfg.teams]
    std, matches, urls = vrs.fetch(year, names, date=cfg.vrs_date, refresh=refresh)
    vdate = std["vrs_date"].iloc[0]
    if not matches.empty:
        back = {t.vrs_name or t.name: t.name for t in cfg.teams}
        matches.insert(1, "team_canonical", matches["team"].map(back))
    day = vdate.replace("_", "-")
    write_snapshot(
        "vrs",
        day,
        {"standings": std, "matches": matches},
        urls=urls,
        notes="Valve Regional Standings (global) + per-team series history for event teams",
    )
    log(f"VRS {day}: {len(std)} standings rows, {len(matches)} match rows")


def fetch_hltv(
    event: Event, refresh: bool = False, teams: list[str] | None = None, log: Callable[[str], None] = print
) -> None:
    cfg = event.config
    urls: list[str] = []
    rank = hltv.ranking(refresh=refresh)
    urls.append(hltv.BASE + "/ranking/teams")
    maps, results, failed = [], [], []
    wanted = [cfg.team(t) for t in teams] if teams else cfg.teams
    for t in wanted:
        if not (t.hltv_id and t.hltv_slug):
            continue
        try:
            page = hltv.team_page(t.hltv_id, t.hltv_slug, refresh=refresh)
        except Exception as e:  # keep going – fallbacks cover gaps
            failed.append(t.name)
            log(f"  ! {t.name}: {e}")
            continue
        maps.append(hltv.parse_team_maps(page, t.name).assign(team=t.name))
        r = hltv.parse_team_results(page, t.name)
        if not r.empty:
            results.append(r.assign(team=t.name))
        urls.append(f"{hltv.BASE}/team/{t.hltv_id}/{t.hltv_slug}")
        log(f"  HLTV {t.name}: {len(maps[-1])} maps, {len(r)} results")
    tables = {
        "ranking": rank,
        "maps": pd.concat(maps, ignore_index=True) if maps else pd.DataFrame(),
        "results": pd.concat(results, ignore_index=True) if results else pd.DataFrame(),
    }
    write_snapshot(
        "hltv",
        date.today().isoformat(),
        tables,
        urls=urls,
        notes="HLTV world ranking + team-page map stats (past 3 months) and recent results",
        extra={"failed": failed},
    )
    log(f"HLTV: ranking {len(rank)} rows, maps {len(tables['maps'])}, results {len(tables['results'])}")


def fetch_all(
    event: Event, sources: list[str], refresh: bool = False, log: Callable[[str], None] = print
) -> None:
    if "vrs" in sources:
        fetch_vrs(event, refresh, log)
    if "hltv" in sources:
        fetch_hltv(event, refresh, log=log)
