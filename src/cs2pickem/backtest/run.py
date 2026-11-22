"""Point-in-time backtest of a past Major.

For every stage the model is rebuilt with data available *before that stage started*: the pre-event VRS
file / HLTV ranking (pinned via ``snapshot_days``), match history cut at the stage start, plus the real
results of the Major's earlier stages. Then we compare against what actually happened:

* Pick'Em: optimised picks graded vs actual records (coin = >= threshold correct); bracket picks for playoffs
* per-match Brier score / log-loss of the series win probability actually faced (correct Bo1/Bo3/Bo5)
* calibration: reliability bins of match probabilities and of P(3-0)/P(advance)/P(0-3)
* baselines: VRS-points-only model and a coin flip (p = 0.5 everywhere; Pick'Em from equal-strength sims)
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..config import Event, StageConfig, load_event
from ..live import MatchResult, swiss_format
from ..paths import home
from ..pickem.bracket_opt import optimise_bracket
from ..pickem.swiss_opt import CAT_LABEL, optimise_swiss
from ..pipeline import Context, playoff_known
from ..playoffs import simulate_bracket
from ..swiss.simulator import simulate_swiss
from .majors import MAJORS

MODELS = ("model", "legacy", "vrs_only", "coin_flip")
MODEL_LABEL = {
    "model": "Configured model",
    "legacy": "Legacy blend (v1)",
    "vrs_only": "VRS seed/points only",
    "coin_flip": "Coin flip",
}
EPS = 1e-6


def load_actual(event: Event, stage: StageConfig) -> list[dict]:
    p = Path(stage.results_file or "")
    p = p if p.is_absolute() else home() / p
    raw = yaml.safe_load(p.read_text()) or {}
    canon = event.config.canonical
    out = []
    for m in raw.get("matches", []):
        out.append({**m, "a": canon(m["a"]), "b": canon(m["b"]), "winner": canon(m["winner"])})
    return out


def _orient(matches: list[dict]) -> pd.DataFrame:
    rows = []
    for m in matches:
        if not m.get("winner") or not m.get("date"):
            continue
        for t, o in ((m["a"], m["b"]), (m["b"], m["a"])):
            rows.append({"team": t, "date": m["date"], "opponent": o, "result": float(m["winner"] == t)})
    return pd.DataFrame(rows, columns=["team", "date", "opponent", "result"])


def _with_rating(event: Event, **update) -> Event:
    r = event.config.rating.model_copy(update=update)
    return event.model_copy(update={"config": event.config.model_copy(update={"rating": r})})


def _legacy(event: Event) -> Event:
    """The original v1 model: VRS/HLTV/Elo z-blend at 200 Elo per sd, no temperature."""
    return _with_rating(event, method="blend", temperature=1.0, rating_scale=200.0)


def _vrs_only(event: Event) -> Event:
    r = event.config.rating.model_copy(
        update={"method": "blend", "temperature": 1.0, "w_hltv": 0.0, "w_elo": 0.0}
    )
    return event.model_copy(update={"config": event.config.model_copy(update={"rating": r})})


def _records(matches: list[dict], teams: list[str]) -> dict[str, tuple[int, int]]:
    rec = {t: [0, 0] for t in teams}
    for m in matches:
        rec[m["winner"]][0] += 1
        rec[m["b"] if m["winner"] == m["a"] else m["a"]][1] += 1
    return {t: (w, lo) for t, (w, lo) in rec.items()}


def _category(w: int, lo: int, W: int, L: int) -> str | None:
    if w == W:
        return "perfect" if lo == 0 else "advance"
    if lo == L and w == 0:
        return "winless"
    return None


def _match_scores(probs: list[float], outcomes: list[int]) -> dict:
    p = np.clip(np.array(probs, float), EPS, 1 - EPS)
    y = np.array(outcomes, float)
    if not len(p):
        return {"n": 0, "brier": None, "log_loss": None, "accuracy": None}
    return {
        "n": int(len(p)),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "accuracy": float(np.mean(np.where(p == 0.5, 0.5, (p > 0.5) == (y == 1)))),
    }


def reliability(probs: list[float], outcomes: list[int], bins: int = 10) -> list[dict]:
    p, y = np.array(probs, float), np.array(outcomes, float)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        hi = edges[i + 1] if i < bins - 1 else 1.0 + 1e-9
        sel = (p >= edges[i]) & (p < hi)
        if sel.any():
            out.append(
                {
                    "lo": float(edges[i]),
                    "hi": float(edges[i + 1]),
                    "n": int(sel.sum()),
                    "mean_p": float(p[sel].mean()),
                    "freq": float(y[sel].mean()),
                }
            )
    return out


def _swiss_stage(
    ctxs: dict[str, Context], event: Event, stage: StageConfig, actual: list[dict], n_sims: int
) -> dict:
    teams = list(stage.teams)
    fmt = swiss_format(stage)
    W, L = fmt.wins_to_advance, fmt.losses_to_eliminate
    rec = _records(actual, teams)
    pk = event.config.pickem
    out: dict = {"type": "swiss", "teams": teams, "actual": {t: f"{w}-{lo}" for t, (w, lo) in rec.items()}}
    actual_cat = {t: _category(w, lo, W, L) for t, (w, lo) in rec.items()}
    for name in MODELS:
        if name == "coin_flip":
            mats = {bo: np.full((len(teams), len(teams)), 0.5) for bo in {fmt.default_bo, fmt.decider_bo}}
        else:
            mm = ctxs[name].model(teams)
            mats = {bo: mm.matrix(bo) for bo in {fmt.default_bo, fmt.decider_bo, 1, 3}}
        idx = {t: i for i, t in enumerate(teams)}
        res = simulate_swiss(
            teams,
            {bo: mats[bo] for bo in {fmt.default_bo, fmt.decider_bo}},
            fmt,
            n_sims=n_sims,
            seed=event.config.sim.seed,
            round1=[tuple(p) for p in stage.round1] if stage.round1 else None,  # type: ignore[misc]
        )
        picks = optimise_swiss(res, pk.n_30, pk.n_03, pk.n_advance, pk.threshold)
        named = picks.named()
        graded = {
            c: [{"team": t, "correct": actual_cat[t] == c, "actual": out["actual"][t]} for t in ts]
            for c, ts in named.items()
        }
        n_correct = sum(g["correct"] for gs in graded.values() for g in gs)
        probs, ys, rows = [], [], []
        for m in actual:
            p = float(mats[m["bo"]][idx[m["a"]], idx[m["b"]]])
            y = int(m["winner"] == m["a"])
            probs.append(p)
            ys.append(y)
            rows.append(
                {
                    "round": m["round"],
                    "a": m["a"],
                    "b": m["b"],
                    "bo": m["bo"],
                    "p_a": round(p, 4),
                    "winner": m["winner"],
                }
            )
        summ = res.summary()
        team_rows = []
        for i, t in enumerate(teams):
            team_rows.append(
                {
                    "team": t,
                    "seed": i + 1,
                    "p_30": float(summ["perfect"][i]),
                    "p_adv": float(summ["qualified"][i]),
                    "p_03": float(summ["winless"][i]),
                    "actual": out["actual"][t],
                    "advanced": rec[t][0] == W,
                }
            )
        out[name] = {
            "picks": graded,
            "n_correct": int(n_correct),
            "coin": bool(n_correct >= pk.threshold),
            "p_coin": float(picks.p_threshold),
            "expected_correct": float(picks.expected),
            "matches": rows,
            "match_scores": _match_scores(probs, ys),
            "_probs": probs,
            "_ys": ys,
            "teams": team_rows,
        }
        if name == "model":
            out["ratings"] = {
                t: round(float(ctxs[name].ratings.as_dict().get(t, float("nan"))), 1) for t in teams
            }
    return out


def _playoff_stage(
    ctxs: dict[str, Context], event: Event, stage: StageConfig, actual: list[dict], n_sims: int
) -> dict:
    teams = list(stage.teams)
    res_known = [MatchResult(m["round"], m["a"], m["b"], m["winner"]) for m in actual]
    known = playoff_known(stage, teams, res_known)
    n_rounds = int(np.log2(len(stage.bracket))) + 1
    out: dict = {"type": "playoffs", "teams": teams, "champion": known.get((n_rounds - 1, 0))}
    idx = {t: i for i, t in enumerate(teams)}
    for name in MODELS:
        bos = {stage.playoff_bo, stage.final_bo}
        if name == "coin_flip":
            mats = {bo: np.full((len(teams), len(teams)), 0.5) for bo in bos}
        else:
            mm = ctxs[name].model(teams)
            mats = {bo: mm.matrix(bo) for bo in bos}
        res = simulate_bracket(
            teams,
            mats,
            stage.bracket,
            bo=stage.playoff_bo,
            final_bo=stage.final_bo,
            n_sims=n_sims,
            seed=event.config.sim.seed,
        )
        bp = optimise_bracket(res)
        graded = []
        for r, row in enumerate(bp.named()):
            for m, t in enumerate(row):
                graded.append(
                    {
                        "round": res.labels[r],
                        "pick": t,
                        "actual": known.get((r, m)),
                        "correct": known.get((r, m)) == t,
                        "p": bp.probs[r][m],
                    }
                )
        probs, ys, rows = [], [], []
        for m in actual:
            p = float(mats[m["bo"]][idx[m["a"]], idx[m["b"]]])
            probs.append(p)
            ys.append(int(m["winner"] == m["a"]))
            rows.append(
                {
                    "round": m["round"],
                    "a": m["a"],
                    "b": m["b"],
                    "bo": m["bo"],
                    "p_a": round(p, 4),
                    "winner": m["winner"],
                }
            )
        champ = res.reach()[res.labels[-1]]
        out[name] = {
            "picks": graded,
            "n_correct": int(sum(g["correct"] for g in graded)),
            "n_picks": len(graded),
            "expected_correct": float(bp.expected),
            "matches": rows,
            "match_scores": _match_scores(probs, ys),
            "_probs": probs,
            "_ys": ys,
            "p_champion": {t: float(champ[i]) for i, t in enumerate(teams)},
        }
    return out


def run_backtest(major: str, n_sims: int | None = None, log: Callable[[str], None] = print) -> dict:
    spec = MAJORS[major]
    root = home() / "configs" / spec.id
    event = load_event(root)
    n = n_sims or event.config.sim.n_sims
    report: dict = {
        "id": spec.id,
        "name": spec.name,
        "generated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_sims": n,
        "inputs": {"vrs": spec.vrs_day, "hltv": spec.hltv_day, "map_pool": spec.map_pool},
        "stages": {},
    }
    earlier: list[dict] = []
    for sid, stage in event.stages.items():
        actual = load_actual(event, stage)
        extra = _orient(earlier)
        ctxs = {
            "model": Context.load(event, as_of=stage.start, extra_matches=extra),
            "legacy": Context.load(_legacy(event), as_of=stage.start, extra_matches=extra),
            "vrs_only": Context.load(_vrs_only(event), as_of=stage.start, extra_matches=extra),
        }
        fn = _swiss_stage if stage.type == "swiss" else _playoff_stage
        st = fn(ctxs, event, stage, actual, n)
        st["name"] = stage.name
        st["start"] = stage.start.isoformat() if stage.start else None
        report["stages"][sid] = st
        m = st["model"]
        msg = f"{spec.id} {sid}: picks {m['n_correct']}/{m.get('n_picks', 10)}"
        if stage.type == "swiss":
            msg += f" coin={'yes' if m['coin'] else 'no'} (P≥{event.config.pickem.threshold} was {m['p_coin']:.0%})"
        msg += "  Brier " + " / ".join(f"{k} {st[k]['match_scores']['brier']:.3f}" for k in MODELS)
        log(msg)
        earlier += actual

    report["summary"] = _summarise(report)
    _strip(report)
    out = home() / "data" / "backtests" / spec.id / "report.json"
    out.write_text(json.dumps(report, indent=1, default=_json))
    return report


def _summarise(report: dict) -> dict:
    summ: dict = {}
    for name in MODELS:
        probs = [p for st in report["stages"].values() for p in st[name]["_probs"]]
        ys = [y for st in report["stages"].values() for y in st[name]["_ys"]]
        swiss = [st[name] for st in report["stages"].values() if st["type"] == "swiss"]
        summ[name] = {
            "label": MODEL_LABEL[name],
            "match_scores": _match_scores(probs, ys),
            "reliability": reliability(probs, ys) if name != "coin_flip" else [],
            "coins": sum(s["coin"] for s in swiss),
            "swiss_stages": len(swiss),
            "pickem_correct": sum(s["n_correct"] for s in swiss),
        }
    # outcome calibration of the full model: P(3-0)/P(adv)/P(0-3) vs what happened
    oc: dict = {}
    for key, cat in (("p_30", "3-0"), ("p_adv", "adv"), ("p_03", "0-3")):
        ps, ys = [], []
        for st in report["stages"].values():
            if st["type"] != "swiss":
                continue
            for r in st["model"]["teams"]:
                w, lo = map(int, r["actual"].split("-"))
                hit = {"3-0": w == 3 and lo == 0, "adv": w == 3, "0-3": w == 0 and lo == 3}[cat]
                ps.append(r[key])
                ys.append(int(hit))
        oc[cat] = {
            "brier": _match_scores(ps, ys)["brier"],
            "reliability": reliability(ps, ys, bins=5),
            "n": len(ps),
        }
    summ["outcome_calibration"] = oc
    return summ


def _strip(report: dict) -> None:
    for st in report["stages"].values():
        for name in MODELS:
            st[name].pop("_probs", None)
            st[name].pop("_ys", None)


def _json(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, float) and math.isnan(o):
        return None
    raise TypeError(type(o))


__all__ = ["run_backtest", "reliability", "CAT_LABEL"]
