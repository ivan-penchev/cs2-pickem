"""Render ``site/backtest.html`` from the ``data/backtests/<major>/report.json`` files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..paths import home
from .common import bar, calibration_svg, e, page, pct

MODELS = (
    ("model", "Model (BT + VRS prior)", "#ff4655"),
    ("legacy", "Legacy blend", "#f5a623"),
    ("vrs_only", "VRS only", "#3fa9f5"),
    ("coin_flip", "Coin flip", "#8a96b1"),
)
CAT_TITLE = {"perfect": "3-0 picks", "winless": "0-3 picks", "advance": "Advance picks"}


def load_reports(root: Path | None = None) -> list[dict[str, Any]]:
    root = root or home() / "data" / "backtests"
    out = [json.loads(p.read_text()) for p in sorted(root.glob("*/report.json"))]
    return sorted(out, key=lambda r: r.get("inputs", {}).get("vrs", ""), reverse=True)


def _score_table(rep: dict) -> str:
    rows = []
    stages = rep["stages"]
    for sid, st in stages.items():
        cells = [f'<td class="l">{e(st.get("name", sid))}</td>']
        for key, _, _ in MODELS:
            m = st[key]
            ms = m["match_scores"]
            if st["type"] == "swiss":
                pk = f"{m['n_correct']}/10 " + (
                    '<span class="pill ok">coin</span>' if m["coin"] else '<span class="pill bad">no</span>'
                )
            else:
                pk = f"{m['n_correct']}/{m['n_picks']}"
            cells.append(f"<td>{pk}</td><td>{ms['brier']:.3f}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    tot = ['<td class="l"><b>All matches</b></td>']
    for key, _, _ in MODELS:
        s = rep["summary"][key]
        tot.append(
            f"<td><b>{s['coins']}/{s['swiss_stages']} coins</b></td><td><b>{s['match_scores']['brier']:.3f}</b></td>"
        )
    rows.append("<tr>" + "".join(tot) + "</tr>")
    head = "".join(f'<th colspan="2">{e(lbl)}</th>' for _, lbl, _ in MODELS)
    sub = "".join("<th>Pick'Em</th><th>Brier</th>" for _ in MODELS)
    return (
        f'<div class="scroll"><table><thead><tr><th class="l">Stage</th>{head}</tr>'
        f"<tr><th></th>{sub}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _picks(m: dict) -> str:
    out = []
    for cat, title in CAT_TITLE.items():
        chips = "".join(
            f'<span class="chip {"ok" if p["correct"] else "bad"}">{"✔" if p["correct"] else "✘"} {e(p["team"])}'
            f"<small>{e(p['actual'])}</small></span>"
            for p in m["picks"][cat]
        )
        out.append(f'<div class="cat">{title}</div><div class="picks">{chips}</div>')
    return "".join(out)


def _bracket_picks(m: dict) -> str:
    chips = "".join(
        f'<span class="chip {"ok" if p["correct"] else "bad"}">{"✔" if p["correct"] else "✘"} {e(p["pick"])}'
        f"<small>{e(p['round'])} · {pct(p['p'])} · won: {e(p['actual'])}</small></span>"
        for p in m["picks"]
    )
    return f'<div class="picks">{chips}</div>'


def _swiss_stage(sid: str, st: dict) -> str:
    m = st["model"]
    rows = []
    for t in sorted(m["teams"], key=lambda t: -t["p_adv"]):
        rec = t["actual"]
        cls = "ok" if t["advanced"] else "bad"
        rows.append(
            f'<tr><td class="l">{e(t["team"])} <span class="dim">#{t["seed"]}</span></td>'
            f"<td>{bar(t['p_30'], 'var(--gold)')}</td><td>{bar(t['p_adv'], 'var(--win)')}</td>"
            f'<td>{bar(t["p_03"], "var(--lose)")}</td><td><span class="pill {cls}">{e(rec)}</span></td></tr>'
        )
    verdict = (
        f'<span class="pill {"ok" if m["coin"] else "bad"}">{m["n_correct"]}/10 · '
        f"{'coin earned' if m['coin'] else 'no coin'}</span>"
    )
    return (
        f'<div class="panel"><h2>{e(st.get("name", sid))} {verdict}'
        f'<span class="tl">pre-stage P(≥5) {pct(m["p_coin"])} · E[correct] {m["expected_correct"]:.2f}</span></h2>'
        f"{_picks(m)}"
        f'<div class="scroll" style="margin-top:10px"><table><thead><tr><th class="l">Team</th><th>P(3-0)</th>'
        f"<th>P(adv)</th><th>P(0-3)</th><th>Actual</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></div>"
    )


def _playoff_stage(sid: str, st: dict) -> str:
    m = st["model"]
    champs = sorted(m["p_champion"].items(), key=lambda kv: -kv[1])
    rows = "".join(
        f'<tr><td class="l">{e(t)}{" 🏆" if t == st["champion"] else ""}</td><td>{bar(p, "var(--gold)")}</td></tr>'
        for t, p in champs
    )
    return (
        f'<div class="panel"><h2>{e(st.get("name", sid))} <span class="pill">{m["n_correct"]}/{m["n_picks"]} bracket picks</span>'
        f'<span class="tl">E[correct] {m["expected_correct"]:.2f} · champion: {e(st["champion"])}</span></h2>'
        f"{_bracket_picks(m)}"
        f'<table style="margin-top:10px"><thead><tr><th class="l">Team</th><th>P(champion)</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
    )


def _outcome_cal(rep: dict) -> str:
    oc = rep["summary"]["outcome_calibration"]
    series = [
        (f"P({k}) · Brier {v['brier']:.3f}", c, v["reliability"])
        for (k, v), c in zip(oc.items(), ("#f5b400", "#22d3a6", "#ff4655"), strict=False)
    ]
    return calibration_svg(series)


def _major(rep: dict) -> str:
    s = rep["summary"]
    kpis = []
    for key, lbl, col in MODELS:
        ms = s[key]["match_scores"]
        kpis.append(
            f'<div class="kpi" style="border-color:{col}55"><div class="v" style="color:{col}">{ms["brier"]:.3f}'
            f' <small>Brier</small></div><div class="l">{e(lbl)} · log-loss {ms["log_loss"]:.3f} · acc {pct(ms["accuracy"])}'
            f"<br>{s[key]['coins']}/{s[key]['swiss_stages']} coins · {s[key]['pickem_correct']}/30 Swiss picks</div></div>"
        )
    cal = calibration_svg([(lbl, col, s[key]["reliability"]) for key, lbl, col in MODELS])
    stages = "".join(
        _swiss_stage(sid, st) if st["type"] == "swiss" else _playoff_stage(sid, st)
        for sid, st in rep["stages"].items()
    )
    inp = rep.get("inputs", {})
    return (
        f'<section id="{e(rep["id"])}" style="margin-top:26px"><div class="kicker">Backtest</div>'
        f"<h1>{e(rep['name'])} <span>· point-in-time</span></h1>"
        f'<div class="sub">Inputs frozen before each stage: VRS {e(inp.get("vrs", "?"))} · HLTV ranking {e(inp.get("hltv", "?"))}'
        f" · match history cut at stage start (+ earlier stages of the same Major) · {rep['n_sims']:,} sims</div>"
        f'<div class="kpis" style="margin:14px 0">{"".join(kpis)}</div>'
        f'<div class="grid"><div class="panel"><h2>Match calibration<span class="tl">per-match win probability, {s["model"]["match_scores"]["n"]} matches</span></h2>{cal}</div>'
        f'<div class="panel"><h2>Outcome calibration<span class="tl">model P(3-0)/P(adv)/P(0-3) vs actual</span></h2>{_outcome_cal(rep)}</div></div>'
        f'<div class="panel" style="margin-top:14px"><h2>Scoreboard<span class="tl">Pick\'Em needs 5/10 for the coin · lower Brier is better</span></h2>{_score_table(rep)}</div>'
        f'<div class="grid" style="margin-top:14px">{stages}</div></section>'
    )


def render(reports: list[dict]) -> str:
    nav = "".join(f'<a href="#{e(r["id"])}">{e(r["name"])}</a>' for r in reports)
    body = (
        '<header><div><div class="kicker">cs2pickem</div><h1>Model backtests <span>· recent Majors</span></h1>'
        '<div class="sub">How would the model have done? Same code, data as of each stage start, no leakage.</div></div>'
        f'<nav><a href="index.html">← Dashboard</a>{nav}</nav></header>'
        + "".join(_major(r) for r in reports)
        + "<footer>Brier score = mean (p − outcome)²; coin flip = 0.250. Reliability bins with n≥1 shown; dot size ∝ n. "
        "Data: Valve VRS (GitHub), HLTV rankings, Liquipedia (CC-BY-SA 3.0).</footer>"
    )
    return page("cs2pickem · backtests", body, "cs2pickem backtest")


def write(out: Path | None = None, reports: list[dict] | None = None) -> Path:
    reports = load_reports() if reports is None else reports
    out = Path(out or home() / "site" / "backtest.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(reports), encoding="utf-8")
    return out
