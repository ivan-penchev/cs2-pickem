"""Build everything a backtest needs for a past Major – point-in-time, no leakage.

Writes (all committed, so the backtest is reproducible offline):

* ``configs/<major>/event.yaml`` + ``stage1..3.yaml`` + ``playoffs.yaml`` (teams in seed order, official
  Round-1 draw, ``snapshot_days`` pinned to the pre-event VRS file / HLTV ranking)
* ``data/snapshots/vrs/<day>/`` – VRS standings + per-team series history from the pre-event file
* ``data/snapshots/hltv/<day>/`` – archived HLTV ranking of that day; results + per-map W/L are
  added separately by :mod:`.hltv_backfill` (``cs2pickem backtest backfill-hltv``)
* ``data/backtests/<major>/results/<stage>.yaml`` + ``actual.yaml`` – the real outcomes (Liquipedia)
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from ..data import hltv, vrs
from ..data.liquipedia import LpMatch, stage_matches
from ..paths import home
from ..snapshots import load_table, write_snapshot
from .majors import LP_TO_VRS, MAJORS, MajorSpec

SINGAPORE_PARAMS = {
    "rating": {
        "method": "bt",
        "temperature": 0.85,
        "bt_half_life_days": 120,
        "bt_window_days": 180,
        "bt_l2": 3.0,
        "bt_prior_weight": 0.5,
        "w_vrs": 0.4,
        "w_hltv": 0.25,
        "w_elo": 0.35,
        "elo_k": 32,
        "elo_half_life_days": 30,
        "elo_window_days": 90,
        "min_roster_overlap": 4,
        "old_roster_weight": 0.35,
        "rating_scale": 200,
    },
    "maps": {"prior_maps": 8, "max_offset": 0.8, "veto_temperature": 0.0},
    "sim": {"n_sims": 100000, "seed": 2026},
    "pickem": {"n_30": 2, "n_03": 2, "n_advance": 6, "threshold": 5},
}


def lp_date(s: str | None) -> str | None:
    if not s:
        return None
    try:
        return datetime.strptime(s.split(" - ")[0].strip(), "%B %d, %Y").date().isoformat()
    except ValueError:
        return None


def stage_page(spec: MajorSpec, page: str) -> str:
    return f"{spec.liquipedia}/{page}" if page else spec.liquipedia


def fetch_stage_matches(spec: MajorSpec, ttl_hours: float = 1e6) -> dict[str, list[LpMatch]]:
    out = {}
    for st in spec.stages:
        kind = "swiss" if st.kind == "swiss" else "bracket"
        out[st.id] = stage_matches(stage_page(spec, st.page), kind, ttl_hours=ttl_hours)
    return out


def hltv_to_vrs_points(hltv_points: float) -> float:
    """Map HLTV ranking points onto the VRS points scale.

    Least-squares fit of VRS points on ln(HLTV points) over the top 40 of the Budapest 2025 and
    Cologne 2026 snapshots (corr 0.93-0.94). Only used for Majors that predate VRS."""
    return 215.0 * math.log(max(float(hltv_points), 1.0)) + 515.0


def synthetic_vrs(hr: pd.DataFrame, day: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """A VRS-shaped standings table derived from an HLTV ranking (rank kept, points rescaled)."""
    std = pd.DataFrame(
        {
            "vrs_date": day.replace("-", "_"),
            "rank": hr["rank"].astype(int),
            "points": [round(hltv_to_vrs_points(p), 1) for p in hr["points"]],
            "team": hr["team"],
            "roster": "",
            "details": "synthetic from HLTV ranking",
        }
    )
    vm = pd.DataFrame(
        columns=["team", "date", "opponent", "result", "vrs_match_id", "age_weight", "h2h_adj", "roster"]
    )
    return std, vm, []


def swiss_records(matches: list[LpMatch]) -> dict[str, list[int]]:
    rec: dict[str, list[int]] = {}
    for m in matches:
        for t in (m.a, m.b):
            rec.setdefault(t, [0, 0])
        if m.winner:
            rec[m.winner][0] += 1
            rec[m.b if m.winner == m.a else m.a][1] += 1
    return rec


def _dump(p: Path, body: dict, header: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(header + yaml.safe_dump(body, sort_keys=False, allow_unicode=True, width=120))


def prepare(major: str, log: Callable[[str], None] = print, ttl_hours: float = 1e6) -> Path:
    spec = MAJORS[major]
    root = home()
    cfg_dir = root / "configs" / spec.id
    bt_dir = root / "data" / "backtests" / spec.id
    matches = fetch_stage_matches(spec, ttl_hours)

    # ---- point-in-time strength inputs
    names = sorted({t for ms in matches.values() for m in ms for t in (m.a, m.b)})
    vrs_name = {n: LP_TO_VRS.get(n, n) for n in names}
    rank_path = "/ranking/teams/" + spec.hltv_ranking
    hr = hltv.parse_ranking(hltv.get_page(rank_path, ttl_hours=ttl_hours))
    if spec.synthetic_vrs:
        std_m, vm, urls = synthetic_vrs(hr, spec.hltv_day)
        vrs_notes = f"SYNTHETIC: {spec.name} predates VRS. {spec.note or ''}".strip()
    else:
        std, _, _ = vrs.fetch(spec.vrs_date[:4], [], date=spec.vrs_date)
        missing = [n for n, v in vrs_name.items() if v not in set(std.team)]
        if missing:
            raise ValueError(f"{spec.id}: teams missing from VRS {spec.vrs_date}: {missing}")
        std_m, vm, urls = vrs.fetch(spec.vrs_date[:4], list(vrs_name.values()), date=spec.vrs_date)
        vrs_notes = f"Valve Regional Standings published before {spec.name} (backtest input)"
    missing = [n for n, v in vrs_name.items() if v not in set(std_m.team)]
    if missing:
        raise ValueError(f"{spec.id}: teams missing from the strength table: {missing}")
    back = {v: k for k, v in vrs_name.items()}
    if not vm.empty:
        vm.insert(1, "team_canonical", vm["team"].map(back))
    write_snapshot(
        "vrs",
        spec.vrs_day,
        {"standings": std_m, "matches": vm},
        urls=urls,
        notes=vrs_notes,
        manifest=False,
    )
    # Keep results/maps that ``backtest backfill-hltv`` already rebuilt for this snapshot.
    hmaps = load_table("hltv", "maps", spec.hltv_day)
    hres = load_table("hltv", "results", spec.hltv_day)
    if hmaps is not None and not hmaps.empty and hres is not None and not hres.empty:
        log(f"hltv/{spec.hltv_day}: keeping backfilled results/maps (ranking left as is)")
    else:
        empty_maps = pd.DataFrame(columns=["team", "map", "wins", "losses", "pick_pct", "ban_pct"])
        empty_res = pd.DataFrame(columns=["date", "team", "opponent", "maps_won", "maps_lost"])
        write_snapshot(
            "hltv",
            spec.hltv_day,
            {"ranking": hr, "maps": empty_maps, "results": empty_res},
            urls=[hltv.BASE + rank_path],
            notes=(
                f"Archived HLTV world ranking before {spec.name} (backtest input; results and map stats "
                "are added by `cs2pickem backtest backfill-hltv`)"
            ),
            manifest=False,
        )
    log(f"snapshots: vrs/{spec.vrs_day} ({len(vm)} match rows), hltv/{spec.hltv_day} ({len(hr)} teams)")

    vrs_rank = dict(zip(std_m.team, std_m["rank"], strict=False))
    hl = {str(t).lower(): (int(i), s) for t, i, s in zip(hr.team, hr.hltv_id, hr.hltv_slug, strict=False)}

    # ---- event.yaml
    teams = []
    for n in names:
        v = vrs_name[n]
        aliases = sorted({a for a in (v, v.upper()) if a != n})
        hid = hl.get(v.lower())
        t: dict = {"name": n, "vrs_name": v}
        if aliases:
            t["aliases"] = aliases
        if hid:
            t["hltv_id"], t["hltv_slug"] = hid
        teams.append(t)
    event = {
        "id": spec.id,
        "name": spec.name,
        "start": spec.start,
        "end": spec.end,
        "liquipedia": spec.liquipedia,
        "vrs_date": spec.vrs_date,
        **({"notes": spec.note} if spec.note else {}),
        "snapshot_days": {"vrs": spec.vrs_day, "hltv": spec.hltv_day},
        "use_manual_data": False,
        "map_pool": spec.map_pool,
        "stages": [s.id for s in spec.stages],
        **SINGAPORE_PARAMS,
        "teams": teams,
    }
    _dump(
        cfg_dir / "event.yaml",
        event,
        f"# {spec.name} — backtest configuration generated by `cs2pickem backtest prepare`.\n"
        "# Point-in-time: VRS/HLTV snapshots pinned to the days before the event; data/*.csv ignored.\n",
    )

    # ---- stage files + results
    actual: dict = {"event": spec.name, "source": f"https://liquipedia.net/counterstrike/{spec.liquipedia}"}
    seen: set[str] = set()
    for st in spec.stages:
        ms = matches[st.id]
        res_rel = f"data/backtests/{spec.id}/results/{st.id}.yaml"
        if st.kind == "swiss":
            r1 = [[m.a, m.b] for m in ms if m.round == 1]
            stage_teams = [t for p in r1 for t in p]
            # Valve pairs seed k with seed k+8 in Round 1: order the pairs by the better-ranked side.
            pairs = sorted(r1, key=lambda p: min(vrs_rank[vrs_name[t]] for t in p))
            pairs = [sorted(p, key=lambda t: vrs_rank[vrs_name[t]]) for p in pairs]
            seeds = [p[0] for p in pairs] + [p[1] for p in pairs]
            seen |= set(stage_teams)
            body = {
                "name": st.name,
                "type": "swiss",
                "start": st.start,
                "end": st.end,
                "wins_to_advance": 3,
                "losses_to_eliminate": 3,
                "default_bo": 3 if all(m.bo == 3 for m in ms) else 1,
                "decider_bo": 3,
                "results_file": res_rel,
                "liquipedia": stage_page(spec, st.page),
                "teams": seeds,
                "round1": r1,
            }
            rec = swiss_records(ms)
            actual[st.id] = {
                "records": {
                    t: f"{w}-{lo}" for t, (w, lo) in sorted(rec.items(), key=lambda kv: (-kv[1][0], kv[1][1]))
                },
                "n_matches": len(ms),
            }
        else:
            qf = sorted((m for m in ms if m.round == 1), key=lambda m: m.slot or "")
            seeds = [t for m in qf for t in (m.a, m.b)]
            body = {
                "name": st.name,
                "type": "playoffs",
                "start": st.start,
                "end": st.end,
                "playoff_bo": 3,
                "final_bo": max(m.bo for m in ms if m.round == max(x.round for x in ms)),
                "bracket": [[2 * i + 1, 2 * i + 2] for i in range(len(qf))],
                "results_file": res_rel,
                "liquipedia": stage_page(spec, st.page),
                "teams": seeds,
            }
            actual[st.id] = {
                "rounds": {
                    r: [
                        {"a": m.a, "b": m.b, "winner": m.winner}
                        for m in sorted(ms, key=lambda m: m.slot or "")
                        if m.round == r
                    ]
                    for r in sorted({m.round for m in ms})
                },
                "champion": next(m.winner for m in ms if m.round == max(x.round for x in ms)),
            }
        _dump(cfg_dir / f"{st.id}.yaml", body, f"# {spec.name} — {st.name} (generated from Liquipedia)\n")
        _dump(
            root / res_rel,
            {
                "matches": [
                    {
                        "round": m.round,
                        "a": m.a,
                        "b": m.b,
                        "winner": m.winner,
                        "bo": m.bo,
                        "date": lp_date(m.date),
                    }
                    for m in ms
                ]
            },
            f"# {spec.name} — {st.name} actual results (Liquipedia, CC-BY-SA 3.0)\n",
        )
        bos = Counter(m.bo for m in ms)
        log(f"{st.id}: {len(ms)} matches, Bo mix {dict(bos)}")
    _dump(bt_dir / "actual.yaml", actual, f"# {spec.name} — actual outcomes (Liquipedia, CC-BY-SA 3.0)\n")
    log(f"wrote {cfg_dir} and {bt_dir}")
    return cfg_dir
