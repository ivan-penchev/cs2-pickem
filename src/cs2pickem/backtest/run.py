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

MODELS = ("model", "legacy", "vrs_only", "ens_logit", "ens_prob", "coin_flip")
MODEL_LABEL = {
    "model": "Configured model (BT)",
    "legacy": "Legacy blend (v1)",
    "vrs_only": "VRS seed/points only",
    "ens_logit": "BT+VRS 50/50 (logit)",
    "ens_prob": "BT+VRS 50/50 (prob)",
    "coin_flip": "Coin flip",
}
# Pick'Em strategies: probability source, optimiser objective, seed-anchored tie-break.
STRATEGIES: dict[str, tuple[str, str, bool]] = {
    "vrs_only": ("vrs_only", "threshold", False),
    "legacy": ("legacy", "threshold", False),
    "bt": ("model", "threshold", False),
    "ensemble": ("ens_logit", "threshold", False),
    "ensemble_prob": ("ens_prob", "threshold", False),
    "seed_anchored": ("model", "threshold", True),
}
for _k, (_src, _obj, _anc) in list(STRATEGIES.items()):
    STRATEGIES[f"{_k}_exp"] = (_src, "expected", _anc)
STRATEGY_LABEL = {
    "vrs_only": "VRS only",
    "legacy": "Legacy blend",
    "bt": "Bradley-Terry",
    "ensemble": "BT+VRS ensemble (logit)",
    "ensemble_prob": "BT+VRS ensemble (prob)",
    "seed_anchored": "BT, seed-anchored ties",
}
for _k in list(STRATEGY_LABEL):
    STRATEGY_LABEL[f"{_k}_exp"] = STRATEGY_LABEL[_k] + ", max E[correct]"
# Tolerance for "near-tie" in the seed-anchored tie-break, in units of the primary objective.
TIE_TOL = {"threshold": 0.01, "expected": 0.05}
CATS3 = ("perfect", "advance", "winless")
EPS = 1e-6


def _ensemble(a: np.ndarray, b: np.ndarray, scale: str) -> np.ndarray:
    if scale == "prob":
        return 0.5 * (a + b)
    la = np.log(np.clip(a, EPS, 1 - EPS) / np.clip(1 - a, EPS, 1 - EPS))
    lb = np.log(np.clip(b, EPS, 1 - EPS) / np.clip(1 - b, EPS, 1 - EPS))
    m = 1.0 / (1.0 + np.exp(-0.5 * (la + lb)))
    np.fill_diagonal(m, 0.5)
    return m


def _matrices(ctxs: dict[str, Context], name: str, teams: list[str], bos: set[int]) -> dict[int, np.ndarray]:
    n = len(teams)
    if name == "coin_flip":
        return {bo: np.full((n, n), 0.5) for bo in bos}
    if name in ("ens_logit", "ens_prob"):
        a, b = ctxs["model"].model(teams), ctxs["vrs_only"].model(teams)
        scale = "prob" if name == "ens_prob" else "logit"
        return {bo: _ensemble(a.matrix(bo), b.matrix(bo), scale) for bo in bos}
    mm = ctxs[name].model(teams)
    return {bo: mm.matrix(bo) for bo in bos}


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
    out["strategies"] = {}
    for name in MODELS:
        mats = _matrices(ctxs, name, teams, {fmt.default_bo, fmt.decider_bo, 1, 3})
        idx = {t: i for i, t in enumerate(teams)}
        res = simulate_swiss(
            teams,
            {bo: mats[bo] for bo in {fmt.default_bo, fmt.decider_bo}},
            fmt,
            n_sims=n_sims,
            seed=event.config.sim.seed,
            round1=[tuple(p) for p in stage.round1] if stage.round1 else None,  # type: ignore[misc]
        )
        for sname, (src, obj, anchored) in STRATEGIES.items():
            if src != name:
                continue
            sp = optimise_swiss(
                res,
                pk.n_30,
                pk.n_03,
                pk.n_advance,
                pk.threshold,
                objective=obj,
                seed_rank=list(range(len(teams))) if anchored else None,
                tie_tol=TIE_TOL[obj],
            )
            out["strategies"][sname] = _grade(sp, actual_cat, out["actual"], pk.threshold)
        picks = optimise_swiss(res, pk.n_30, pk.n_03, pk.n_advance, pk.threshold)
        g = _grade(picks, actual_cat, out["actual"], pk.threshold)
        graded, n_correct = g["picks"], g["n_correct"]
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


def _grade(picks, actual_cat: dict, actual: dict, threshold: int) -> dict:
    graded = {
        c: [{"team": t, "correct": actual_cat[t] == c, "actual": actual[t]} for t in ts]
        for c, ts in picks.named().items()
    }
    n_correct = sum(g["correct"] for gs in graded.values() for g in gs)
    return {
        "picks": graded,
        "n_correct": int(n_correct),
        "coin": bool(n_correct >= threshold),
        "p_coin": float(picks.p_threshold),
        "expected_correct": float(picks.expected),
        "hits": {c: int(sum(x["correct"] for x in graded.get(c, []))) for c in CATS3},
        "n": {c: len(graded.get(c, [])) for c in CATS3},
    }


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
        mats = _matrices(ctxs, name, teams, {stage.playoff_bo, stage.final_bo})
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
        "inputs": {
            "vrs": spec.vrs_day,
            "hltv": spec.hltv_day,
            "map_pool": spec.map_pool,
            "synthetic_vrs": spec.synthetic_vrs,
            "note": spec.note,
        },
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


# ------------------------------------------------------------------------------------- cross-Major summary
def bootstrap_ci(x: np.ndarray, n_boot: int = 10_000, seed: int = 2026, level: float = 0.95) -> tuple:
    """Percentile bootstrap CI of the mean of ``x`` (resampling stages)."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    a = (1 - level) / 2
    return (float(np.quantile(means, a)), float(np.quantile(means, 1 - a)))


def _brier_sources(reports: list[dict]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for name in MODELS:
        num, den = 0.0, 0
        for r in reports:
            ms = r["summary"][name]["match_scores"]
            if ms["n"]:
                num += ms["brier"] * ms["n"]
                den += ms["n"]
        out[name] = num / den if den else None
    return out


def summarise_majors(reports: list[dict], baseline: str = "vrs_only", default: str = "bt") -> dict:
    """Aggregate the per-Major reports into a strategy comparison over all Swiss stages."""
    rows = []
    for r in reports:
        for sid, st in r["stages"].items():
            if st["type"] == "swiss" and "strategies" in st:
                rows.append((r["id"], sid, st["strategies"]))
    brier = _brier_sources(reports)
    correct = {k: np.array([s[k]["n_correct"] for *_, s in rows], float) for k in STRATEGIES}
    strategies: dict = {}
    for k, (src, obj, anchored) in STRATEGIES.items():
        c = correct[k]
        hits = {cat: sum(s[k]["hits"][cat] for *_, s in rows) for cat in CATS3}
        ns = {cat: sum(s[k]["n"][cat] for *_, s in rows) for cat in CATS3}
        entry = {
            "label": STRATEGY_LABEL[k],
            "source": src,
            "objective": obj,
            "seed_anchored": anchored,
            "mean_correct": float(c.mean()),
            "ci": bootstrap_ci(c),
            "coins": int(sum(s[k]["coin"] for *_, s in rows)),
            "stages": len(rows),
            "mean_p_coin": float(np.mean([s[k]["p_coin"] for *_, s in rows])),
            "hit_rate": {cat: (hits[cat] / ns[cat] if ns[cat] else None) for cat in CATS3},
            "brier": brier[src],
            "per_stage": [int(v) for v in c],
        }
        for ref in (baseline, default):
            d = c - correct[ref]
            entry[f"diff_vs_{ref}"] = {"mean": float(d.mean()), "ci": bootstrap_ci(d)}
        strategies[k] = entry
    return {
        "generated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "majors": [{"id": r["id"], "name": r["name"], "note": r["inputs"].get("note")} for r in reports],
        "stages": [f"{m}/{s}" for m, s, _ in rows],
        "baseline": baseline,
        "default": default,
        "strategies": strategies,
        "brier": brier,
    }


def write_summary(majors: list[str] | None = None) -> dict:
    root = home() / "data" / "backtests"
    reports = []
    for m in majors or list(MAJORS):
        p = root / m / "report.json"
        if p.exists():
            reports.append(json.loads(p.read_text()))
    summ = summarise_majors(reports)
    dec = root / "decision.yaml"
    if dec.exists():
        summ.update(yaml.safe_load(dec.read_text()) or {})
    (root / "summary.json").write_text(json.dumps(summ, indent=1))
    return summ
