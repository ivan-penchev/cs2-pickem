"""Dashboard (``index.html``) and per-stage prediction-history pages (``<stage>-history.html``).

Both are rendered from the append-only prediction history (:mod:`cs2pickem.history`): the original
pre-stage baseline is always shown next to the latest prediction, with deltas, a per-team probability
timeline annotated with the matches that triggered each update, a log of flipped picks and the baseline
picks graded against the results as the stage resolves.
"""

from __future__ import annotations

import json
from typing import Any

from .. import history as H
from .common import bar, delta, e, page, pct

GEN = "cs2pickem site build"


# ------------------------------------------------------------------------------------------- pieces
def _metric(entry: dict, label: str, team: str) -> float | None:
    names = [t["name"] for t in entry["teams"]]
    if team not in names:
        return None
    for m in entry["metrics"]:
        if m["label"] == label:
            return m["values"][names.index(team)]
    return None


def _chip(p: dict, verdict: bool | None, extra: str = "") -> str:
    cls = {True: "ok", False: "bad", None: "pend"}[verdict]
    mark = {True: "✓ ", False: "✗ ", None: ""}[verdict]
    return f'<span class="chip {cls}">{mark}{e(p["team"])}{extra}</span>'


def picks_block(entry: dict, known: list[dict], title: str) -> str:
    groups: dict[str, list[dict]] = {}
    for p in entry["picks"]:
        groups.setdefault(p["group"], []).append(p)
    rows = []
    n_ok = n_bad = 0
    for g, ps in groups.items():
        chips = []
        for p in ps:
            v = H.grade_pick(p, entry["type"], known)
            n_ok += v is True
            n_bad += v is False
            chips.append(_chip(p, v))
        rows.append(f'<div class="cat">{e(g)}</div><div class="picks">{"".join(chips)}</div>')
    if entry["threshold"]:
        need = entry["threshold"]
        if n_ok >= need:
            verdict = '<span class="pill ok">coin secured</span>'
        elif entry["n_picks"] - n_bad < need:
            verdict = '<span class="pill bad">coin lost</span>'
        else:
            verdict = f'<span class="pill">{n_ok}/{need} so far</span>'
        head = (
            f'{verdict} <span class="dim">{n_ok} correct · {n_bad} wrong · P(≥{need}) at the time '
            f"{pct(entry['p_threshold'], 1)} · E {entry['expected']:.2f}</span>"
        )
    else:
        head = (
            f'<span class="pill">{n_ok} correct · {n_bad} wrong</span> '
            f'<span class="dim">E {entry["expected"]:.2f}/{entry["n_picks"]} at the time</span>'
        )
    return f"<h3>{e(title)}</h3><div>{head}</div>{''.join(rows)}"


def compare_table(base: dict, cur: dict, known: list[dict]) -> str:
    labels = [m["label"] for m in cur["metrics"]]
    rec = H.records(known)
    head = "".join(f'<th colspan="3" style="text-align:center">{e(lb)}</th>' for lb in labels)
    sub = "".join("<th>orig</th><th>now</th><th>Δ</th>" for _ in labels)
    order = sorted(
        cur["teams"], key=lambda t: -(_metric(cur, labels[min(1, len(labels) - 1)], t["name"]) or 0)
    )
    rows = []
    for t in order:
        n = t["name"]
        w, lo = rec.get(n, (0, 0))
        cells = []
        for lb in labels:
            b, c = _metric(base, lb, n), _metric(cur, lb, n)
            cells.append(
                f'<td class="dim">{pct(b)}</td><td>{pct(c)}</td>'
                f"<td>{delta(c, b) if b is not None and c is not None else ''}</td>"
            )
        rows.append(
            f'<tr><td class="l"><span style="color:{t["colour"]}">■</span> {e(n)}</td>'
            f'<td class="dim">{w}-{lo}</td>{"".join(cells)}</tr>'
        )
    return (
        '<div class="scroll"><table><thead><tr><th class="l" rowspan="2">team</th><th rowspan="2">rec</th>'
        f"{head}</tr><tr>{sub}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def timeline_svg(hist: list[dict], label: str, w: int = 860, h: int = 300) -> str:
    """Per-team probability lines over the history entries, with the triggering matches annotated."""
    pad_l, pad_r, pad_t, pad_b = 40, 130, 14, 26
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b
    n = len(hist)
    X = lambda i: pad_l + (iw * i / (n - 1) if n > 1 else iw / 2)  # noqa: E731
    Y = lambda p: pad_t + (1 - p) * ih  # noqa: E731
    g = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{e(label)} timeline">']
    for t in (0, 0.25, 0.5, 0.75, 1):
        g.append(f'<line x1="{pad_l}" x2="{pad_l + iw}" y1="{Y(t)}" y2="{Y(t)}" stroke="#232d44"/>')
        g.append(
            f'<text x="{pad_l - 6}" y="{Y(t) + 3}" fill="#55607a" font-size="10" text-anchor="end">{t:.0%}</text>'
        )
    for i, en in enumerate(hist):
        x = X(i)
        g.append(
            f'<line x1="{x}" x2="{x}" y1="{pad_t}" y2="{pad_t + ih}" stroke="#2e3a57" stroke-dasharray="2 4"/>'
        )
        short = "baseline" if en.get("baseline") else en["trigger"].split(";")[0].replace(" beat ", ">")
        g.append(
            f'<text x="{x}" y="{h - 8}" fill="#8a96b1" font-size="9" text-anchor="middle">'
            f"{e(short[:26])}<title>{e(en['ts'])} – {e(en['trigger'])}</title></text>"
        )
    teams = hist[-1]["teams"]
    finals = []
    for t in teams:
        vals = [_metric(en, label, t["name"]) for en in hist]
        pts = [(X(i), Y(v)) for i, v in enumerate(vals) if v is not None]
        if not pts:
            continue
        line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        tip = " → ".join(pct(v) for v in vals if v is not None)
        g.append(
            f'<g class="tl-line"><polyline points="{line}" fill="none" stroke="{t["colour"]}" stroke-width="2" '
            f'stroke-opacity=".85"/>'
            + "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{t["colour"]}"/>' for x, y in pts)
            + f"<title>{e(t['name'])}: {tip}</title></g>"
        )
        finals.append((pts[-1][1], t, vals[-1]))
    # right-hand labels, de-overlapped
    finals.sort(key=lambda z: z[0])
    last = -1e9
    for y, t, v in finals:
        y = max(y, last + 11)
        last = y
        g.append(
            f'<text x="{pad_l + iw + 8}" y="{y + 3:.1f}" fill="{t["colour"]}" font-size="10">'
            f"{e(t['tag'])} {pct(v)}</text>"
        )
    g.append("</svg>")
    return "".join(g)


def timeline_panel(hist: list[dict], uid: str) -> str:
    labels = [m["label"] for m in hist[-1]["metrics"]]
    default = labels[1] if len(labels) > 1 and hist[-1]["type"] == "swiss" else labels[-1]
    tabs, svgs = [], []
    for k, lb in enumerate(labels):
        on = lb == default
        tabs.append(f'<button class="tab{" on" if on else ""}" data-g="{uid}" data-k="{k}">{e(lb)}</button>')
        svgs.append(
            f'<div class="tlv" data-g="{uid}" data-k="{k}"{"" if on else " hidden"}>{timeline_svg(hist, lb)}</div>'
        )
    note = (
        "Only the baseline so far – the timeline grows with every scraped result."
        if len(hist) == 1
        else f"{len(hist)} snapshots · hover a line for its values, a label for the triggering match."
    )
    return (
        f'<div class="panel"><h2>Probability timeline<span class="tl">{"".join(tabs)}</span></h2>'
        f'{"".join(svgs)}<div class="note">{e(note)}</div></div>'
    )


def changes_log(hist: list[dict]) -> str:
    ch = H.pick_changes(hist)
    if not ch:
        return '<div class="note">No recommended pick has changed since the baseline.</div>'
    rows = []
    for c in reversed(ch):
        add = ", ".join(f"{t} ({g})" for g, t in c["added"])
        rem = ", ".join(f"{t} ({g})" for g, t in c["removed"])
        rows.append(
            f'<tr><td class="l dim">{e(c["ts"].replace("T", " ")[:16])}</td><td class="l">{e(c["trigger"])}</td>'
            f'<td class="l up">+ {e(add)}</td><td class="l down">− {e(rem)}</td></tr>'
        )
    return (
        '<div class="scroll"><table><thead><tr><th class="l">when</th><th class="l">after</th>'
        f'<th class="l">in</th><th class="l">out</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )


def history_log(hist: list[dict]) -> str:
    rows = []
    for en in reversed(hist):
        tag = '<span class="pill gold">baseline</span> ' if en.get("baseline") else ""
        p = (
            f"P(≥{en['threshold']}) {pct(en['p_threshold'], 1)}"
            if en["threshold"]
            else f"E {en['expected']:.2f}"
        )
        rows.append(
            f'<tr><td class="l dim">{e(en["ts"].replace("T", " ")[:16])}</td><td class="l">{tag}{e(en["trigger"])}</td>'
            f"<td>{sum(1 for m in en['known'] if m.get('winner'))}</td><td>{p}</td>"
            f'<td class="dim">{e(en.get("source") or "–")}</td></tr>'
        )
    return (
        '<div class="scroll"><table><thead><tr><th class="l">UTC</th><th class="l">trigger</th><th>results</th>'
        f"<th>pick'em</th><th>source</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def standings(entry: dict, known: list[dict], wins: int = 3, losses: int = 3) -> str:
    rec = H.records(known)
    label = entry["metrics"][1]["label"] if len(entry["metrics"]) > 1 else entry["metrics"][0]["label"]
    rows = []

    def key(t):
        w, lo = rec.get(t["name"], (0, 0))
        return (-(w - lo), -w, -(_metric(entry, label, t["name"]) or 0))

    for t in sorted(entry["teams"], key=key):
        w, lo = rec.get(t["name"], (0, 0))
        if entry["type"] == "swiss":
            st = (
                '<span class="pill ok">advanced</span>'
                if w >= wins
                else '<span class="pill bad">eliminated</span>'
                if lo >= losses
                else '<span class="dim">alive</span>'
            )
        else:
            st = '<span class="pill bad">out</span>' if lo else '<span class="dim">alive</span>'
        p = _metric(entry, label, t["name"]) or 0
        rows.append(
            f'<tr><td class="l"><span style="color:{t["colour"]}">■</span> {e(t["name"])} '
            f'<span class="dim">#{t["seed"]}</span></td><td>{w}-{lo}</td><td class="l">{st}</td>'
            f'<td style="width:40%">{bar(p, "var(--win)")}</td></tr>'
        )
    return (
        f'<div class="scroll"><table><thead><tr><th class="l">team</th><th>W-L</th><th class="l">status</th>'
        f'<th class="l">{e(label)} now</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )


def last_matches(known: list[dict], n: int = 6) -> str:
    fin = [m for m in known if m.get("winner")]
    if not fin:
        pend = [m for m in known if not m.get("winner")]
        if pend:
            return "Next up: " + ", ".join(f"{e(m['a'])} vs {e(m['b'])}" for m in pend[:8])
        return "No matches played yet."
    return " · ".join(e(H.describe(m)) for m in reversed(fin[-n:]))


def stage_section(st: dict[str, Any], full: bool = True) -> str:
    """One stage: baseline vs current, standings, timeline, picks (graded), pick changes."""
    hist, cur = st["history"], st["current"]
    base = hist[0] if hist else cur
    known = cur["known"]
    tl_hist = hist if hist else [cur | {"ts": cur.get("ts", ""), "trigger": "projection", "baseline": True}]
    proj = st["how"] == "projected"
    kp = []
    if cur["threshold"]:
        kp.append(
            f'<div class="kpi"><div class="v">{pct(cur["p_threshold"], 1)} '
            f"<small>{delta(cur['p_threshold'], base['p_threshold'], 1)}</small></div>"
            f'<div class="l">P(≥{cur["threshold"]} correct) now · orig {pct(base["p_threshold"], 1)}</div></div>'
        )
    kp.append(
        f'<div class="kpi"><div class="v">{cur["expected"]:.2f}<small>/{cur["n_picks"]}</small></div>'
        f'<div class="l">expected correct picks · orig {base["expected"]:.2f}</div></div>'
    )
    played = sum(1 for m in known if m.get("winner"))
    kp.append(f'<div class="kpi"><div class="v">{played}</div><div class="l">matches played</div></div>')
    kp.append(
        f'<div class="kpi"><div class="v">{len(hist)}</div><div class="l">prediction snapshots</div></div>'
    )
    links = f'<a href="{e(st["page"])}">▶ animated Monte Carlo</a>'
    if full is False and hist:
        links += f' · <a href="{e(st["history_page"])}">full history</a>'
    note = (
        '<div class="note">Teams are <b>projected</b> from simulations of the previous stage – no baseline '
        "is recorded until the seeding is final.</div>"
        if proj
        else ""
    )
    parts = [
        f'<section style="margin-top:22px"><div class="kicker">{e(st["status"])}</div>'
        f'<h1 style="font-size:21px">{e(cur["stage_name"])} <span>· {links}</span></h1>{note}'
        f'<div class="kpis" style="margin:12px 0">{"".join(kp)}</div>',
        '<div class="grid">',
        f'<div class="panel"><h2>Live standings<span class="tl">{last_matches(known, 3)}</span></h2>'
        f"{standings(cur, known)}</div>",
        '<div class="panel"><h2>Pick\'Em<span class="tl">original vs current, graded live</span></h2>'
        + picks_block(base, known, "Original (baseline) picks")
        + (picks_block(cur, known, "Current recommendation") if hist and len(hist) > 1 else "")
        + "</div></div>",
        f'<div class="grid wide" style="margin-top:14px">{timeline_panel(tl_hist, st["id"])}</div>',
        '<div class="grid wide" style="margin-top:14px"><div class="panel"><h2>Original vs current'
        f'<span class="tl">orig = {e(base.get("ts", "now")[:16].replace("T", " "))} baseline</span></h2>'
        f"{compare_table(base, cur, known)}</div></div>",
    ]
    if full:
        parts.append(
            '<div class="grid" style="margin-top:14px">'
            f'<div class="panel"><h2>Picks changed</h2>{changes_log(hist)}</div>'
            f'<div class="panel"><h2>Prediction log<span class="tl">append-only</span></h2>{history_log(hist)}</div></div>'
        )
    else:
        parts.append(
            f'<div class="grid wide" style="margin-top:14px"><div class="panel"><h2>Picks changed</h2>'
            f"{changes_log(hist)}</div></div>"
        )
    parts.append("</section>")
    return "".join(parts)


EXTRA_CSS = """<style>
.tab{background:var(--panel2);border:1px solid var(--line2);color:var(--dim);border-radius:7px;padding:3px 8px;
font:700 10px var(--mono);margin-left:4px;cursor:pointer}.tab.on{color:var(--text);border-color:var(--acc2)}
.tl-line:hover polyline{stroke-width:4;stroke-opacity:1}
.status{display:flex;flex-wrap:wrap;gap:8px 18px;color:var(--dim);font-size:12px;margin-top:6px}
.status b{color:var(--text)}
.live{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--acc);box-shadow:0 0 10px var(--acc);
margin-right:6px;animation:pulse 1.6s infinite}@keyframes pulse{50%{opacity:.35}}
</style>"""

TAB_JS = """<script>
document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>{const g=b.dataset.g;
document.querySelectorAll(`.tab[data-g="${g}"]`).forEach(x=>x.classList.toggle('on',x===b));
document.querySelectorAll(`.tlv[data-g="${g}"]`).forEach(x=>x.hidden=x.dataset.k!==b.dataset.k);}));
</script>"""


def _cards(stages: list[dict], has_backtest: bool) -> str:
    cards = []
    for st in stages:
        sub = st["status"] + (" · projected teams" if st["how"] == "projected" else "")
        cards.append(
            f'<a class="card" href="{e(st["page"])}">{e(st["name"])} – Monte Carlo<small>{e(sub)}</small></a>'
        )
        if st["history"]:
            cards.append(
                f'<a class="card" href="{e(st["history_page"])}">{e(st["name"])} – prediction history'
                f"<small>{len(st['history'])} snapshot(s), baseline {e(st['history'][0]['ts'][:10])}</small></a>"
            )
    if has_backtest:
        cards.append(
            '<a class="card" href="backtest.html">Backtests<small>Budapest 2025 · Cologne 2026, point-in-time</small></a>'
        )
    return f'<div class="cards">{"".join(cards)}</div>'


def _odds_strip(st: dict) -> str:
    cur = st["current"]
    lb = cur["metrics"][-1]["label"]
    top = sorted(cur["teams"], key=lambda t: -(_metric(cur, lb, t["name"]) or 0))[:8]
    rows = "".join(
        f'<tr><td class="l"><span style="color:{t["colour"]}">■</span> {e(t["name"])}</td>'
        f'<td style="width:55%">{bar(_metric(cur, lb, t["name"]) or 0, "var(--gold)")}</td></tr>'
        for t in top
    )
    return (
        f'<div class="panel"><h2>{e(st["name"])}<span class="tl">{e(lb)} · '
        f'<a href="{e(st["page"])}">open</a></span></h2><table><tbody>{rows}</tbody></table></div>'
    )


def render_index(event: dict, stages: list[dict], current: str, status: dict, has_backtest: bool) -> str:
    cur = next((s for s in stages if s["id"] == current), stages[0])
    others = [s for s in stages if s["id"] != cur["id"]]
    live_dot = '<span class="live"></span>' if event.get("live") else ""
    st_line = (
        f'<div class="status"><span>{live_dot}<b>{e(event["phase"])}</b></span>'
        f"<span>last updated <b>{e(status['built'])}</b></span>"
        f"<span>last result change <b>{e(status.get('last_change') or '–')}</b></span>"
        f"<span>data source <b>{e(status.get('source') or 'config / manual')}</b></span></div>"
    )
    body = (
        f'<header><div><div class="kicker">cs2pickem · live dashboard</div><h1>{e(event["name"])} '
        f'<span>· Pick\'Em predictor</span></h1><div class="sub">{e(event["dates"])} · Monte Carlo Swiss + playoffs, '
        f"VRS/HLTV/Elo ratings, map veto model</div>{st_line}</div>"
        f"<nav>{''.join(f'<a href={json.dumps(s["page"])}>{e(s["name"])}</a>' for s in stages)}"
        + ('<a href="backtest.html">Backtests</a>' if has_backtest else "")
        + "</nav></header>"
        + stage_section(cur, full=False)
        + '<section style="margin-top:22px"><div class="kicker">Other stages</div>'
        f'<div class="grid" style="margin-top:10px">{"".join(_odds_strip(s) for s in others)}</div></section>'
        + f'<section style="margin-top:22px"><div class="kicker">Pages</div><div style="margin-top:10px">{_cards(stages, has_backtest)}</div></section>'
        + "<footer>Predictions are recomputed after every scraped result; the original pre-stage baseline is kept "
        "forever (append-only log in <code>data/predictions/</code>). Data: Valve VRS (GitHub), HLTV, "
        "Liquipedia (CC-BY-SA 3.0), DFRAG. Not affiliated with Valve, PGL or HLTV.</footer>"
    )
    return page(f"{event['name']} · Pick'Em dashboard", EXTRA_CSS + body, GEN, TAB_JS)


def render_stage_history(event: dict, st: dict, status: dict) -> str:
    body = (
        f'<header><div><div class="kicker">cs2pickem · prediction history</div><h1>{e(event["name"])} '
        f'<span>· {e(st["name"])}</span></h1><div class="sub">How the prediction changed as results came in '
        f"· built {e(status['built'])}</div></div>"
        f'<nav><a href="index.html">← Dashboard</a><a href="{e(st["page"])}">Monte Carlo</a></nav></header>'
        + stage_section(st, full=True)
    )
    return page(f"{event['name']} · {st['name']} · history", EXTRA_CSS + body, GEN, TAB_JS)
