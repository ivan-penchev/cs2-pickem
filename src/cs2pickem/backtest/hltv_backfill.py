"""Backfill point-in-time HLTV results + per-map stats into a backtest's HLTV snapshot.

The live scraper reads each team page (recent results + "past 3 months" map stats). Those pages only
show the present, so for a past Major we rebuild the same two tables from HLTV's results archive:

* ``/results?team=<id>&offset=<k>`` (newest first, 100 per page) gives every series with its date,
  score and format. Bo1 rows carry the map and the round score directly.
* ``/matches/<id>/<slug>`` gives the per-map scores of Bo2/Bo3/Bo5 series.

Only series played in ``[hltv_day - window, hltv_day)`` are kept, and map stats are summed from those
individual maps, so nothing after the cutoff can leak in. HLTV's aggregate stats pages are never used.
Pick/ban rates cannot be reconstructed this way and are left empty.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from ..config import _norm, load_event
from ..data import hltv
from ..paths import configs_dir
from ..snapshots import load_table, snapshot_root, write_snapshot
from .majors import MAJORS

WINDOW_DAYS = 90
TTL = 1e6  # historical pages never change
MAX_PAGES = 15


def _team_series(hltv_id: int, start: str, end: str, log: Callable[[str], None]) -> list[dict]:
    """All series of one team with ``start <= date < end`` (pages until we are past ``start``)."""
    out: list[dict] = []
    for k in range(MAX_PAGES):
        path = f"/results?team={hltv_id}" + (f"&offset={100 * k}" if k else "")
        rows = hltv.parse_results_list(hltv.get_page(path, ttl_hours=TTL))
        out += [r for r in rows if start <= r["date"] < end]
        if len(rows) < 100 or rows[-1]["date"] < start:
            break
    else:  # pragma: no cover
        log(f"  ! team {hltv_id}: still not past {start} after {MAX_PAGES} pages")
    return out


def _self_names(series: list[dict], cfg_names: set[str]) -> set[str]:
    """HLTV display name(s) of the team whose results page this is (handles renames in the window)."""
    c = Counter(n for r in series for n in (r["team1"], r["team2"]))
    own = {n for n, k in c.items() if k >= max(2, 0.25 * len(series))}
    own |= {n for n in c if _norm(n) in cfg_names}
    return own


def backfill(major: str, log: Callable[[str], None] = print, window_days: int = WINDOW_DAYS) -> dict:
    spec = MAJORS[major]
    ev = load_event(configs_dir() / major)
    cfg = ev.config
    day = spec.hltv_day
    end = day
    start = (date.fromisoformat(day) - timedelta(days=window_days)).isoformat()
    log(f"{spec.name}: HLTV backfill window [{start}, {end}) for {len(cfg.teams)} teams")

    results: list[dict] = []
    team_maps: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    gaps: list[str] = []
    urls: list[str] = []
    n_match_pages = 0
    seen_series = 0
    for t in cfg.teams:
        if not t.hltv_id:
            gaps.append(f"{t.name}: no hltv_id")
            continue
        try:
            series = _team_series(t.hltv_id, start, end, log)
        except Exception as e:
            gaps.append(f"{t.name}: results list failed ({e})")
            log(f"  ! {t.name}: {e}")
            continue
        urls.append(f"{hltv.BASE}/results?team={t.hltv_id}")
        own = _self_names(series, {_norm(n) for n in t.all_names()})
        n_res = n_maps = 0
        for r in series:
            if r["fmt"] == "def":
                continue
            if r["team1"] in own and r["team2"] not in own:
                opp, sw, sl, flip = r["team2"], r["score1"], r["score2"], False
            elif r["team2"] in own and r["team1"] not in own:
                opp, sw, sl, flip = r["team1"], r["score2"], r["score1"], True
            else:
                gaps.append(f"{t.name}: cannot orient match {r['match_id']} ({r['team1']} vs {r['team2']})")
                continue
            seen_series += 1
            results.append(
                {"date": r["date"], "team": t.name, "opponent": opp, "maps_won": sw, "maps_lost": sl}
            )
            n_res += 1
            if r["fmt"] in ("bo2", "bo3", "bo5"):
                try:
                    page = hltv.get_page(f"/matches/{r['match_id']}/{r['slug']}", ttl_hours=TTL)
                    n_match_pages += 1
                except Exception as e:
                    gaps.append(f"{t.name}: match {r['match_id']} page failed ({e})")
                    continue
                for m in hltv.parse_match_maps(page):
                    if m["score1"] == m["score2"]:
                        continue
                    if m["team1"] in own:
                        mine = m["team1"]
                    elif m["team2"] in own:
                        mine = m["team2"]
                    else:  # name differs from the results list; fall back to list order
                        mine = m["team2"] if flip else m["team1"]
                    won = (m["score1"] > m["score2"]) == (mine == m["team1"])
                    team_maps[(t.name, m["map"])][0 if won else 1] += 1
                    n_maps += 1
            elif sw != sl:  # Bo1: fmt is the map name, scores are rounds
                team_maps[(t.name, hltv.map_name(r["fmt"]))][0 if sw > sl else 1] += 1
                n_maps += 1
        log(f"  {t.name}: {n_res} series, {n_maps} maps (names {sorted(own)})")

    res = pd.DataFrame(results, columns=["date", "team", "opponent", "maps_won", "maps_lost"])
    res = res.sort_values(["team", "date"], ascending=[True, False]).reset_index(drop=True)
    mp = pd.DataFrame(
        [
            {"team": t, "map": m, "wins": w, "losses": lo, "pick_pct": None, "ban_pct": None}
            for (t, m), (w, lo) in sorted(team_maps.items())
        ],
        columns=["team", "map", "wins", "losses", "pick_pct", "ban_pct"],
    )
    ranking = load_table("hltv", "ranking", day)
    old_meta = json.loads((snapshot_root() / "hltv" / day / "meta.json").read_text())
    write_snapshot(
        "hltv",
        day,
        {"ranking": ranking, "maps": mp, "results": res},
        urls=[u for u in old_meta.get("urls", []) if "/ranking/" in u] + urls,
        notes=(
            f"Archived HLTV world ranking before {spec.name}, plus results and per-map W/L of the "
            f"{len(cfg.teams)} Major teams backfilled from HLTV results and match pages for "
            f"[{start}, {end}). Map stats are summed from individual maps in that window; pick/ban "
            "rates are not reconstructable and left empty."
        ),
        extra={
            "backfill": {
                "source": "hltv.org /results?team=<id> + /matches/<id> (Playwright)",
                "scraped_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
                "window": [start, end],
                "teams": len(cfg.teams),
                "series": seen_series,
                "match_pages": n_match_pages,
                "gaps": gaps,
            }
        },
        manifest=False,
    )
    log(f"hltv/{day}: {len(res)} results, {len(mp)} team-map rows, {len(gaps)} gaps")
    return {
        "results": len(res),
        "maps": len(mp),
        "map_games": int(mp[["wins", "losses"]].sum().sum()),
        "gaps": gaps,
    }
