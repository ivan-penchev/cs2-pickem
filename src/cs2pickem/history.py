"""Append-only prediction history: ``data/predictions/<stage>/<UTC timestamp>.json``.

Every time the known results of a stage change, a new entry records the per-team outcome probabilities, the
recommended Pick'Em, P(≥threshold), the results known at the time and the match(es) that triggered the
update. The first entry of a stage is the **baseline** (the original pre-stage prediction). It is never
rewritten, and no entry is ever modified or deleted.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .paths import data_dir

SCHEMA = 1


def root_dir(root: Path | None = None) -> Path:
    return Path(root) if root else data_dir() / "predictions"


def _key(m: dict) -> tuple:
    return (m["round"], *sorted((m["a"], m["b"])), m.get("winner"))


def _finished(known: list[dict]) -> list[dict]:
    return [m for m in known if m.get("winner")]


def describe(m: dict) -> str:
    loser = m["b"] if m["winner"] == m["a"] else m["a"]
    return f"R{m['round']}: {m['winner']} beat {loser}"


def entry_from_payload(payload: dict[str, Any], source: str | None = None) -> dict[str, Any]:
    """Condense a viz payload (Swiss or playoffs) into a history entry (without trigger/baseline)."""
    final, cats = payload["final"], payload["cats"]
    names = [t["name"] for t in payload["teams"]]
    metrics = []
    for b in payload["bars"]:
        vals = [round(sum(final[cats[c]][i] for c in b["cats"]), 5) for i in range(len(names))]
        metrics.append({"label": b["label"], "colour": b["colour"], "values": vals})
    pk = payload["pickem"]
    return {
        "schema": SCHEMA,
        "stage": payload["stage"],
        "stage_name": payload["stage_name"],
        "type": payload["kind"],
        "teams_from": payload.get("teams_from"),
        "n_sims": payload["n_sims"],
        "teams": [
            {"name": t["name"], "tag": t["tag"], "colour": t["colour"], "seed": t["seed"]}
            for t in payload["teams"]
        ],
        "metrics": metrics,
        "picks": [{"group": p["group"], "team": names[p["team"]], "cats": p["cats"]} for p in pk["picks"]],
        "cats": cats,
        "threshold": pk.get("threshold"),
        "p_threshold": pk["p_threshold"],
        "expected": pk["expected"],
        "n_picks": pk["n_picks"],
        "known": [dict(m) for m in payload.get("known", [])],
        "source": source,
    }


def load(stage: str, root: Path | None = None) -> list[dict[str, Any]]:
    d = root_dir(root) / stage
    if not d.is_dir():
        return []
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]


def stages(root: Path | None = None) -> list[str]:
    r = root_dir(root)
    return sorted(p.name for p in r.iterdir() if p.is_dir()) if r.is_dir() else []


def record(
    entry: dict[str, Any],
    root: Path | None = None,
    now: datetime | None = None,
    force: bool = False,
    note: str | None = None,
) -> Path | None:
    """Append ``entry`` unless the known results are unchanged since the last entry. Returns the new file."""
    hist = load(entry["stage"], root)
    fin = _finished(entry["known"])
    if hist:
        prev = {_key(m) for m in _finished(hist[-1]["known"])}
        new = [m for m in fin if _key(m) not in prev]
        same_teams = [t["name"] for t in hist[-1]["teams"]] == [t["name"] for t in entry["teams"]]
        if not new and same_teams and len(prev) == len(fin) and not force and not note:
            return None
        trigger = "; ".join(describe(m) for m in new) if new else "re-run"
        baseline = False
    else:
        trigger = "pre-stage baseline" if not fin else f"baseline ({len(fin)} results already known)"
        baseline = True
    if note and hist:
        trigger = note
    now = now or datetime.now(UTC)
    out = {**entry, "ts": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "baseline": baseline, "trigger": trigger}
    d = root_dir(root) / entry["stage"]
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    i = 1
    while p.exists():  # never overwrite
        p = d / f"{now.strftime('%Y%m%dT%H%M%SZ')}-{i}.json"
        i += 1
    p.write_text(json.dumps(out, indent=1) + "\n")
    return p


# ------------------------------------------------------------------------------------------- analysis
def records(known: list[dict]) -> dict[str, tuple[int, int]]:
    rec: dict[str, list[int]] = {}
    for m in _finished(known):
        loser = m["b"] if m["winner"] == m["a"] else m["a"]
        rec.setdefault(m["winner"], [0, 0])[0] += 1
        rec.setdefault(loser, [0, 0])[1] += 1
    return {k: (v[0], v[1]) for k, v in rec.items()}


def grade_pick(pick: dict, entry_type: str, known: list[dict], wins: int = 3, losses: int = 3) -> bool | None:
    """True/False once a pick is decided by the known results, None while still open."""
    if entry_type == "swiss":
        w, lo = records(known).get(pick["team"], (0, 0))
        g = pick["group"].lower()
        if "3-0" in g:
            return True if (w, lo) == (wins, 0) else (False if lo > 0 else None)
        if "0-3" in g:
            return True if (w, lo) == (0, losses) else (False if w > 0 else None)
        return True if w == wins else (False if lo == losses else None)
    # playoffs: cats = rounds-won categories; pick in round r needs r+1 wins (cats[0] - 0)
    need = min(pick["cats"]) if pick["cats"] else 1
    team = pick["team"]
    won = sum(1 for m in _finished(known) if m["winner"] == team)
    lost = any(team in (m["a"], m["b"]) and m["winner"] != team for m in _finished(known))
    if won >= need:
        return True
    return False if lost else None


def pick_changes(hist: list[dict]) -> list[dict]:
    """Which recommended picks flipped between consecutive entries."""
    out = []
    for prev, cur in zip(hist, hist[1:], strict=False):
        a = {(p["group"], p["team"]) for p in prev["picks"]}
        b = {(p["group"], p["team"]) for p in cur["picks"]}
        if a != b:
            out.append(
                {
                    "ts": cur["ts"],
                    "trigger": cur["trigger"],
                    "added": sorted(b - a),
                    "removed": sorted(a - b),
                }
            )
    return out
