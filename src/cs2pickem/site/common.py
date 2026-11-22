"""Shared dark-esports styling and tiny HTML helpers for the static pages (consistent with the animations)."""

from __future__ import annotations

import html

CSS = """
:root{--bg:#07090f;--bg2:#0d111b;--panel:#111725;--panel2:#161e2f;--line:#232d44;--line2:#2e3a57;
--text:#e6ebf5;--dim:#8a96b1;--faint:#55607a;--acc:#ff4655;--acc2:#3fa9f5;--gold:#f5b400;--win:#22d3a6;--lose:#ff4655;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
--sans:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--text);font:13px/1.45 var(--sans)}
body{background:radial-gradient(1200px 500px at 10% -10%,rgba(255,70,85,.13),transparent 60%),
radial-gradient(900px 500px at 100% 0%,rgba(63,169,245,.12),transparent 60%),var(--bg);min-height:100vh}
a{color:var(--acc2);text-decoration:none}a:hover{text-decoration:underline}
.wrap{max-width:1360px;margin:0 auto;padding:20px 20px 48px}
header{display:flex;flex-wrap:wrap;gap:12px 28px;align-items:flex-end;justify-content:space-between;margin-bottom:16px}
.kicker{font:700 11px/1 var(--mono);letter-spacing:.22em;color:var(--acc);text-transform:uppercase}
h1{margin:6px 0 2px;font-size:26px;font-weight:800}h1 span{color:var(--dim);font-weight:600}
.sub{color:var(--dim);font-size:12px}
nav a{display:inline-block;margin-left:6px;padding:6px 11px;border:1px solid var(--line2);border-radius:8px;background:var(--panel2);
color:var(--text);font-weight:700;font-size:12px}nav a:hover{border-color:var(--acc2);text-decoration:none}
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr))}
.grid.wide{grid-template-columns:1fr}
.panel{background:linear-gradient(180deg,var(--panel),var(--bg2));border:1px solid var(--line);border-radius:14px;padding:14px;min-width:0}
.panel h2{margin:0 0 10px;font:800 11px/1 var(--mono);letter-spacing:.2em;text-transform:uppercase;color:var(--dim);display:flex;gap:10px;align-items:center}
.panel h2 .tl{margin-left:auto;letter-spacing:.04em;text-transform:none;font-weight:600;color:var(--faint)}
.panel h3{margin:14px 0 6px;font-size:13px}
.kpis{display:flex;flex-wrap:wrap;gap:10px}
.kpi{flex:1 1 140px;background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:10px 12px}
.kpi .v{font:800 26px/1.05 var(--mono);font-variant-numeric:tabular-nums}.kpi .v small{font-size:13px;color:var(--dim)}
.kpi .l{font:600 10px/1.2 var(--mono);letter-spacing:.14em;color:var(--dim);text-transform:uppercase;margin-top:6px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th{font:700 10px/1 var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--faint);text-align:right;padding:6px 6px;border-bottom:1px solid var(--line2);white-space:nowrap}
td{padding:5px 6px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child,td.l,th.l{text-align:left}
tr:hover td{background:rgba(63,169,245,.05)}
.scroll{overflow-x:auto}
.ok{color:var(--win)}.bad{color:var(--lose)}.dim{color:var(--dim)}.gold{color:var(--gold)}
.up{color:var(--win)}.down{color:var(--lose)}
.pill{display:inline-block;padding:2px 8px;border-radius:99px;font:700 11px var(--mono);border:1px solid var(--line2);background:var(--panel2)}
.pill.ok{border-color:rgba(34,211,166,.5);background:rgba(34,211,166,.1)}.pill.bad{border-color:rgba(255,70,85,.5);background:rgba(255,70,85,.1)}
.pill.gold{border-color:rgba(245,180,0,.5);background:rgba(245,180,0,.1)}
.bar{position:relative;height:14px;background:var(--panel2);border-radius:4px;min-width:80px;overflow:hidden}
.bar i{position:absolute;left:0;top:0;bottom:0;border-radius:4px}
.bar b{position:absolute;right:4px;top:0;font:700 10px/14px var(--mono)}
.picks{display:flex;flex-wrap:wrap;gap:6px}
.chip{padding:4px 8px;border-radius:8px;background:var(--panel2);border:1px solid var(--line2);font-weight:700;font-size:12px}
.chip.ok{border-color:var(--win)}.chip.bad{border-color:var(--lose);opacity:.85}.chip.pend{border-style:dashed}
.chip small{color:var(--dim);font-weight:600;margin-left:4px}
.cat{font:700 10px var(--mono);letter-spacing:.14em;color:var(--faint);text-transform:uppercase;margin:8px 0 4px}
.note{color:var(--faint);font-size:11px;margin-top:8px}
footer{margin-top:22px;color:var(--faint);font-size:11px}
svg text{font-family:var(--mono)}
.cards a.card{display:block;padding:12px 14px;border:1px solid var(--line2);border-radius:12px;background:var(--panel2);color:var(--text);font-weight:700}
.cards a.card:hover{border-color:var(--acc);text-decoration:none}.cards a.card small{display:block;color:var(--dim);font-weight:500}
.cards{display:grid;gap:10px;grid-template-columns:repeat(auto-fit,minmax(200px,1fr))}
"""

e = html.escape


def pct(x: float | None, d: int = 0) -> str:
    return "–" if x is None else f"{100 * x:.{d}f}%"


def bar(p: float, colour: str = "var(--acc2)") -> str:
    return f'<div class="bar"><i style="width:{100 * p:.1f}%;background:{colour}"></i><b>{pct(p)}</b></div>'


def delta(new: float, old: float, d: int = 0, eps: float = 0.005) -> str:
    dv = new - old
    if abs(dv) < eps:
        return '<span class="dim">·</span>'
    cls, sym = ("up", "▲") if dv > 0 else ("down", "▼")
    return f'<span class="{cls}">{sym}{abs(100 * dv):.{d}f}</span>'


def page(title: str, body: str, generator: str, script: str = "") -> str:
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<meta name="generator" content="{e(generator)}"><title>{e(title)}</title>'
        f'<style>{CSS}</style></head><body><div class="wrap">{body}</div>{script}</body></html>\n'
    )


def calibration_svg(series: list[tuple[str, str, list[dict]]], w: int = 420, h: int = 320) -> str:
    """Reliability diagram: ``series`` = [(label, colour, bins)] with bins {mean_p, freq, n}."""
    pad = 38
    iw, ih = w - pad - 12, h - pad - 14
    X = lambda p: pad + p * iw  # noqa: E731
    Y = lambda p: 12 + (1 - p) * ih  # noqa: E731
    g = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="calibration plot">']
    for t in (0, 0.25, 0.5, 0.75, 1):
        g.append(f'<line x1="{X(0)}" x2="{X(1)}" y1="{Y(t)}" y2="{Y(t)}" stroke="#232d44"/>')
        g.append(f'<line y1="{Y(0)}" y2="{Y(1)}" x1="{X(t)}" x2="{X(t)}" stroke="#232d44"/>')
        g.append(
            f'<text x="{X(t)}" y="{h - 18}" fill="#55607a" font-size="10" text-anchor="middle">{t:.2g}</text>'
        )
        g.append(
            f'<text x="{pad - 6}" y="{Y(t) + 3}" fill="#55607a" font-size="10" text-anchor="end">{t:.2g}</text>'
        )
    g.append(
        f'<line x1="{X(0)}" y1="{Y(0)}" x2="{X(1)}" y2="{Y(1)}" stroke="#8a96b1" stroke-dasharray="4 4"/>'
    )
    g.append(
        f'<text x="{X(0.5)}" y="{h - 3}" fill="#8a96b1" font-size="10" text-anchor="middle">predicted probability</text>'
    )
    g.append(
        f'<text x="10" y="{Y(0.5)}" fill="#8a96b1" font-size="10" text-anchor="middle" '
        f'transform="rotate(-90 10 {Y(0.5)})">observed frequency</text>'
    )
    nmax = max((b["n"] for _, _, bins in series for b in bins), default=1)
    for i, (label, col, bins) in enumerate(series):
        pts = " ".join(f"{X(b['mean_p']):.1f},{Y(b['freq']):.1f}" for b in bins)
        g.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
        for b in bins:
            r = 3 + 6 * (b["n"] / nmax) ** 0.5
            g.append(
                f'<circle cx="{X(b["mean_p"]):.1f}" cy="{Y(b["freq"]):.1f}" r="{r:.1f}" fill="{col}" fill-opacity=".75">'
                f"<title>{e(label)}: predicted {b['mean_p']:.2f}, observed {b['freq']:.2f} (n={b['n']})</title></circle>"
            )
        g.append(f'<rect x="{pad + 8}" y="{16 + 15 * i}" width="10" height="10" rx="2" fill="{col}"/>')
        g.append(f'<text x="{pad + 23}" y="{25 + 15 * i}" fill="#e6ebf5" font-size="11">{e(label)}</text>')
    g.append("</svg>")
    return "".join(g)
