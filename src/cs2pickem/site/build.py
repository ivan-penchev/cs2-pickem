"""``cs2pickem site build``: regenerate every page of the static site and append prediction history."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from .. import history as H
from ..config import Event
from ..paths import data_dir, home
from . import dashboard as D

STATUS_FILE = "_status.json"


def status_path() -> Path:
    return data_dir() / "predictions" / STATUS_FILE


def read_status() -> dict[str, Any]:
    p = status_path()
    return json.loads(p.read_text()) if p.exists() else {}


def write_status(source: str | None, changes: list[str], ran: bool, reason: str) -> dict[str, Any]:
    """Persist the outcome of the last live update (source used, last change) for the dashboard."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    st = read_status()
    st.update({"last_check": now, "last_reason": reason, "ran": ran})
    if source:
        st["source"] = source
    if changes:
        st["last_change"] = now
        st["last_changes"] = changes[-10:]
    p = status_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(st, indent=1, sort_keys=True) + "\n")
    return st


def _phase(event: Event, today: date) -> tuple[str, str | None, bool]:
    """Human-readable event phase, the 'current' stage id, and whether the event is live."""
    cfg = event.config
    ids = list(event.stages)
    for sid in ids:
        s = event.stages[sid]
        if s.start and s.end and s.start <= today <= s.end:
            return f"LIVE – {s.name}", sid, True
    if cfg.start and today < cfg.start:
        return f"Pre-event – starts {cfg.start:%d %b %Y}", ids[0], False
    if cfg.end and today > cfg.end:
        return "Event finished", ids[-1], False
    nxt = [sid for sid in ids if event.stages[sid].start and event.stages[sid].start > today]
    return (f"Break – next: {event.stages[nxt[0]].name}", nxt[0], True) if nxt else ("–", ids[0], False)


def build(
    event: Event,
    n_sims: int | None = None,
    n_runs: int = 300,
    out_dir: Path | None = None,
    record: bool = True,
    today: date | None = None,
    note: str | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    from ..pipeline import Context
    from ..viz import build_payload, write_html
    from . import backtest as bt

    out = Path(out_dir) if out_dir else home() / "site"
    out.mkdir(parents=True, exist_ok=True)
    today = today or date.today()
    ctx = Context.load(event)
    status = read_status()
    status["built"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    stages: list[dict[str, Any]] = []
    for sid, st in event.stages.items():
        try:
            payload = build_payload(ctx, st, n_sims, n_runs=n_runs)
        except (ValueError, KeyError) as exc:
            log(f"skip {sid}: {exc}")
            continue
        write_html(payload, out / f"{sid}.html")
        entry = H.entry_from_payload(payload, status.get("source"))
        how = payload.get("teams_from") or ""
        if record and how != "projected":
            p = H.record(entry, note=note)
            if p:
                log(f"history: {p.relative_to(home()) if p.is_relative_to(home()) else p}")
        hist = H.load(sid)
        if hist and how != "projected":
            entry = {**entry, "ts": status["built"], "trigger": "current", "baseline": False}
        played = sum(1 for m in entry["known"] if m.get("winner"))
        if st.start and today < st.start:
            stat = f"upcoming · {st.start:%d %b}"
        elif st.end and today > st.end:
            stat = "finished" if played else "finished?"
        else:
            stat = "live" if (st.start and today >= st.start) else "scheduled"
        stages.append(
            {
                "id": sid,
                "name": st.name,
                "how": how,
                "status": f"{st.name} · {stat}",
                "page": f"{sid}.html",
                "history_page": f"{sid}-history.html",
                "history": hist if how != "projected" else [],
                "current": entry,
            }
        )
        log(f"built {sid}.html ({payload['n_sims']:,} sims, teams: {how})")
    if not stages:
        raise RuntimeError("no stage could be built")

    phase, cur, live = _phase(event, today)
    if cur not in {s["id"] for s in stages}:
        cur = stages[0]["id"]
    cfg = event.config
    ev = {
        "name": cfg.name,
        "dates": f"{cfg.start:%d %b} – {cfg.end:%d %b %Y}" if cfg.start and cfg.end else "",
        "phase": phase,
        "live": live,
    }
    for s in stages:
        if s["history"]:
            (out / s["history_page"]).write_text(D.render_stage_history(ev, s, status), encoding="utf-8")
    reports = bt.load_reports()
    if reports:
        bt.write(out / "backtest.html", reports)
    has_bt = (out / "backtest.html").exists()
    idx = out / "index.html"
    idx.write_text(D.render_index(ev, stages, cur, status, has_bt), encoding="utf-8")
    (out / ".nojekyll").touch()
    log(f"built {idx}")
    return idx
