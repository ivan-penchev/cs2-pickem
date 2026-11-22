"""Export Monte Carlo output for the animated HTML visualisation.

The payload is plain JSON embedded in a single static HTML page (no build step, no network):

* ``runs`` – a sample of fully logged simulations, every match as ``[round, slot, a, b, bo, winner, forced]``
  plus each team's final category index (``fin``) so the page can replay them round by round;
* ``ck`` – cumulative tallies of *all* simulations at log-spaced checkpoints (per-team final-category
  counts, correct-pick histogram, upsets) so the page can animate convergence 1 → N without the raw data;
* ``prob`` / ``maps`` – match win probabilities and the most likely veto line for hover tooltips.

Categories are the terminal Swiss records (``3-0`` … ``0-3``) or, for playoffs, how far a team got
(``QF exit`` … ``Champion``). Pick'Em picks reference the categories that make them correct, so the
front-end treats both stage types the same way.
"""

from __future__ import annotations

import colorsys
import hashlib
import html as _html
import json
import re
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..config import StageConfig
from ..live import load_results, swiss_format

if TYPE_CHECKING:
    from ..pipeline import Context

SCHEMA_VERSION = 1

# Distinct, saturated colours that read well on a dark background (seed order).
PALETTE = [
    "#ff4655", "#3fa9f5", "#f5b400", "#22d3a6", "#a66bff", "#ff8a3d", "#4ade80", "#ec4899",
    "#06b6d4", "#eab308", "#8b5cf6", "#f97316", "#14b8a6", "#ef4444", "#60a5fa", "#84cc16",
]  # fmt: skip


# --------------------------------------------------------------------------------------------- helpers
def team_tag(name: str, aliases: list[str] | None = None, tag: str | None = None) -> str:
    """Short badge label: explicit tag, else the shortest short alias, else initials / truncated name."""
    if tag:
        return tag
    short = sorted((a for a in aliases or [] if 2 <= len(a) <= 4 and " " not in a), key=len)
    if short:
        return short[0]
    words = re.findall(r"[A-Za-z0-9]+", name)
    if len(words) >= 2:
        return "".join(w[0] if not w.isdigit() else w for w in words)[:4].upper()
    return name[:5]


def team_colour(name: str, seed: int) -> str:
    if seed < len(PALETTE):
        return PALETTE[seed]
    h = int(hashlib.sha1(name.encode()).hexdigest()[:6], 16) / 0xFFFFFF
    r, g, b = colorsys.hls_to_rgb(h, 0.6, 0.75)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def checkpoints(n: int, k: int = 240) -> list[int]:
    """~k log-spaced sim counts in [1, n] (always including 1 and n)."""
    if n <= 0:
        return []
    pts = np.unique(np.round(np.logspace(0, np.log10(n), k)).astype(int))
    pts = pts[(pts >= 1) & (pts <= n)]
    if pts[-1] != n:
        pts = np.append(pts, n)
    return [int(x) for x in pts]


def _cum_at(x: np.ndarray, pts: list[int]) -> np.ndarray:
    """Cumulative sums of ``x`` (n, ...) along axis 0 evaluated at the 1-based counts ``pts``."""
    ends = np.asarray(pts)
    starts = np.concatenate([[0], ends[:-1]])
    seg = np.add.reduceat(np.asarray(x, dtype=np.int64), starts, axis=0)
    return np.cumsum(seg, axis=0)


def _teams_meta(ctx: Context, teams: list[str]) -> list[dict[str, Any]]:
    info = {t.name: t for t in ctx.event.config.teams}
    ratings = ctx.ratings.as_dict()
    out = []
    for i, name in enumerate(teams):
        ti = info.get(name)
        out.append(
            {
                "name": name,
                "tag": team_tag(name, ti.aliases if ti else None, ti.tag if ti else None),
                "seed": i + 1,
                "colour": team_colour(name, i),
                "rating": round(float(ratings[name]), 1) if name in ratings else None,
            }
        )
    return out


def _veto_lines(mm, pairs: set[tuple[int, int, int]], pool: list[str]) -> dict[str, list[int]]:
    idx = {m: k for k, m in enumerate(pool)}
    return {f"{a}-{b}-{bo}": [idx[m] for m in mm.veto_line(a, b, bo)] for a, b, bo in sorted(pairs)}


def _meta(ctx: Context, st: StageConfig, n: int, how: str, kind: str) -> dict[str, Any]:
    ev = ctx.event.config
    return {
        "schema": SCHEMA_VERSION,
        "kind": kind,
        "event": ev.name,
        "event_id": ev.id,
        "stage": st.id,
        "stage_name": st.name,
        "n_sims": n,
        "seed": ev.sim.seed,
        "teams_from": how,
        "generated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    }


# --------------------------------------------------------------------------------------------- swiss
def build_swiss_payload(
    ctx: Context, st: StageConfig, n_sims: int | None = None, n_runs: int = 300, use_results: bool = True
) -> dict[str, Any]:
    from ..pickem import CAT_LABEL, optimise_swiss
    from ..pipeline import run_swiss, stage_teams

    teams, how = stage_teams(ctx, st)
    st = st.model_copy(update={"teams": teams})
    fmt = swiss_format(st)
    res = run_swiss(ctx, st, n_sims, use_results=use_results, record_runs=n_runs)
    n, T = res.n_sims, len(teams)
    W, L = fmt.wins_to_advance, fmt.losses_to_eliminate

    cats = list(res.record_distribution())  # "3-0", "3-1", ..., "0-3"
    cat_of = {(int(c.split("-")[0]), int(c.split("-")[1])): k for k, c in enumerate(cats)}
    fin = np.full((n, T), -1, np.int8)
    for (w, lo), k in cat_of.items():
        fin[(res.wins == w) & (res.losses == lo)] = k
    onehot = np.zeros((n, T, len(cats)), np.int8)
    np.put_along_axis(onehot, fin[..., None].astype(np.int64), 1, axis=2)

    r = ctx.event.config.pickem
    pk = optimise_swiss(res, r.n_30, r.n_03, r.n_advance, r.threshold)
    cat_sets = {
        "perfect": [cat_of[(W, 0)]],
        "advance": [cat_of[(W, lo)] for lo in range(1, L)],
        "winless": [cat_of[(0, L)]],
    }
    picks = [
        {"group": CAT_LABEL[c], "cat": c, "team": int(t), "cats": cat_sets[c]}
        for c in ("perfect", "advance", "winless")
        for t in pk.picks[c]
    ]
    correct = sum(np.isin(fin[:, p["team"]], p["cats"]).astype(np.int16) for p in picks)
    n_picks = len(picks)

    mm = ctx.model(teams)
    bos = sorted({fmt.default_bo, fmt.decider_bo})
    pts = checkpoints(n)
    runs, pairs = [], set()
    for s, log in enumerate(res.runs):
        slot: dict[int, int] = {}
        ms = []
        for rnd, a, b, bo, w, forced in log:
            ms.append([rnd, slot.get(rnd, 0), a, b, bo, w, int(forced)])
            slot[rnd] = slot.get(rnd, 0) + 1
            pairs.add((min(a, b), max(a, b), bo))
        runs.append({"m": ms, "fin": fin[s].tolist()})

    known = [m.as_dict() for m in load_results(ctx.event, st)] if use_results else []
    upsets = res.upsets if res.upsets is not None else np.zeros(n, np.int16)
    n_matches = res.n_matches if res.n_matches is not None else np.zeros(n, np.int16)
    return {
        **_meta(ctx, st, n, how, "swiss"),
        "format": {
            "wins": W,
            "losses": L,
            "default_bo": fmt.default_bo,
            "decider_bo": fmt.decider_bo,
            "rounds": fmt.max_rounds,
        },
        "teams": _teams_meta(ctx, teams),
        "pool": list(mm.pool),
        "prob": {str(bo): np.round(mm.matrix(bo), 4).tolist() for bo in bos},
        "maps": _veto_lines(mm, pairs, list(mm.pool)),
        "known": known,
        "cats": cats,
        "cat_style": ["adv" if int(c[0]) == W else "elim" for c in cats],
        "bars": [
            {"label": "P(3-0)", "cats": cat_sets["perfect"], "colour": "#f5b400"},
            {"label": "P(advance)", "cats": cat_sets["advance"], "colour": "#22d3a6"},
            {"label": "P(0-3)", "cats": cat_sets["winless"], "colour": "#ff4655"},
        ],
        "runs": runs,
        "ck": {
            "n": pts,
            "cat": _cum_at(onehot, pts).tolist(),
            "hist": _cum_at(np.eye(n_picks + 1, dtype=np.int8)[correct], pts).tolist(),
            "upsets": _cum_at(upsets, pts).tolist(),
            "matches": _cum_at(n_matches, pts).tolist(),
        },
        "final": {c: np.round(p, 5).tolist() for c, p in res.record_distribution().items()},
        "pickem": {
            "picks": picks,
            "threshold": pk.threshold,
            "n_picks": n_picks,
            "p_threshold": round(pk.p_threshold, 5),
            "expected": round(pk.expected, 4),
            "distribution": np.round(pk.distribution, 5).tolist(),
            "rule": f"{r.n_30}× 3-0, {r.n_03}× 0-3, {r.n_advance}× advance – {r.threshold}+ correct wins",
        },
    }


# --------------------------------------------------------------------------------------------- playoffs
def build_playoff_payload(
    ctx: Context, st: StageConfig, n_sims: int | None = None, n_runs: int = 300
) -> dict[str, Any]:
    from ..pickem import optimise_bracket
    from ..pipeline import playoff_known, run_playoffs

    res, how = run_playoffs(ctx, st, n_sims)
    teams = res.teams
    n, T, R = res.n_sims, len(teams), len(res.rounds)
    mm = ctx.model(teams)
    known = playoff_known(st, teams, load_results(ctx.event, st))

    # Rounds won per team per sim → categories "QF exit" … "Champion".
    won = np.zeros((n, T), np.int8)
    for rnd in res.rounds:
        for w in rnd:
            won[np.arange(n), w] += 1
    exits = {"Quarter-finals": "QF exit", "Semi-finals": "SF exit", "Final": "Runner-up"}
    cats = [exits.get(lab, f"{lab} exit") for lab in res.labels] + ["Champion"]
    onehot = np.zeros((n, T, R + 1), np.int8)
    np.put_along_axis(onehot, won[..., None].astype(np.int64), 1, axis=2)

    bo_of = [st.final_bo if r == R - 1 else st.playoff_bo for r in range(R)]
    mats = {bo: mm.matrix(bo) for bo in set(bo_of)}
    # Reconstruct every match (a, b) per sim and count upsets.
    sides: list[list[tuple[np.ndarray, np.ndarray]]] = []
    current = [(np.full(n, a - 1), np.full(n, b - 1)) for a, b in st.bracket]
    upsets = np.zeros(n, np.int16)
    n_matches = np.zeros(n, np.int16)
    for r, rnd in enumerate(res.rounds):
        sides.append(current)
        P = mats[bo_of[r]]
        for m, w in enumerate(rnd):
            a, b = current[m]
            if (r, m) in known:
                continue
            lo = np.where(w == a, b, a)
            n_matches += 1
            upsets += (P[w, lo] < 0.5).astype(np.int16)
        current = [(rnd[2 * k], rnd[2 * k + 1]) for k in range(len(rnd) // 2)]

    bp = optimise_bracket(res)
    picks = [
        {
            "group": f"{lab} {m + 1}" if len(row) > 1 else lab,
            "round": r,
            "slot": m,
            "team": int(t),
            "cats": list(range(r + 1, R + 1)),
        }
        for r, (lab, row) in enumerate(zip(res.labels, bp.picks, strict=True))
        for m, t in enumerate(row)
    ]
    correct = sum((res.rounds[p["round"]][p["slot"]] == p["team"]).astype(np.int16) for p in picks)

    k = min(n_runs, n)
    runs, pairs = [], set()
    for s in range(k):
        ms = []
        for r, rnd in enumerate(res.rounds):
            for m, w in enumerate(rnd):
                a, b = int(sides[r][m][0][s]), int(sides[r][m][1][s])
                a, b = min(a, b), max(a, b)
                ms.append([r + 1, m, a, b, bo_of[r], int(w[s]), int((r, m) in known)])
                pairs.add((a, b, bo_of[r]))
        runs.append({"m": ms, "fin": won[s].tolist()})

    pts = checkpoints(n)
    return {
        **_meta(ctx, st, n, how, "playoffs"),
        "format": {"rounds": R, "labels": res.labels, "bo": bo_of, "bracket": st.bracket},
        "teams": _teams_meta(ctx, teams),
        "pool": list(mm.pool),
        "prob": {str(bo): np.round(m, 4).tolist() for bo, m in mats.items()},
        "maps": _veto_lines(mm, pairs, list(mm.pool)),
        "known": [m.as_dict() for m in load_results(ctx.event, st)],
        "cats": cats,
        "cat_style": ["elim"] * R + ["adv"],
        "bars": [
            {"label": "P(semis)", "cats": list(range(1, R + 1)), "colour": "#3fa9f5"},
            {"label": "P(final)", "cats": list(range(2, R + 1)), "colour": "#a66bff"},
            {"label": "P(champion)", "cats": [R], "colour": "#f5b400"},
        ][-R:],
        "runs": runs,
        "ck": {
            "n": pts,
            "cat": _cum_at(onehot, pts).tolist(),
            "hist": _cum_at(np.eye(len(picks) + 1, dtype=np.int8)[correct], pts).tolist(),
            "upsets": _cum_at(upsets, pts).tolist(),
            "matches": _cum_at(n_matches, pts).tolist(),
        },
        "final": {c: np.round(onehot[:, :, k].mean(axis=0), 5).tolist() for k, c in enumerate(cats)},
        "pickem": {
            "picks": picks,
            "threshold": None,
            "n_picks": len(picks),
            "p_threshold": round(bp.p_all, 5),
            "expected": round(bp.expected, 4),
            "distribution": np.round(bp.distribution, 5).tolist(),
            "rule": "bracket: every correct match winner scores – maximise expected correct",
        },
    }


def build_payload(
    ctx: Context, st: StageConfig, n_sims: int | None = None, n_runs: int = 300, use_results: bool = True
) -> dict[str, Any]:
    if st.type == "playoffs":
        return build_playoff_payload(ctx, st, n_sims, n_runs)
    return build_swiss_payload(ctx, st, n_sims, n_runs, use_results)


# --------------------------------------------------------------------------------------------- html
def _template() -> str:
    return resources.files(__package__).joinpath("template.html").read_text(encoding="utf-8")


def render_html(payload: dict[str, Any]) -> str:
    """Inline the payload as JSON into the self-contained HTML template."""
    blob = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    blob = blob.replace("<", "\\u003c")  # keeps "</script>" / "<!--" out of the inline script
    title = _html.escape(f"{payload['event']} – {payload['stage_name']} · Monte Carlo")
    html = _template()
    return html.replace("__TITLE__", title).replace("__DATA__", blob)


def write_html(payload: dict[str, Any], out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(payload), encoding="utf-8")
    return out
