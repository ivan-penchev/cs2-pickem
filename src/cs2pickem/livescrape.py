"""Live result scraping for a running Major stage (used by ``cs2pickem live update`` / GitHub Actions).

Sources are tried in order and the first one that works is used:

1. **HLTV** ``/results?event=<id>`` rendered with headless Playwright (often Cloudflare-blocked on CI).
2. **Liquipedia** MediaWiki API (``action=parse``) – bot-friendly, also provides upcoming pairings.
3. **DFRAG** event match list (plain HTML, aria-labels carry the scores).

All parsers are pure functions over text so they can be tested offline. Scraping never deletes recorded
results – :func:`cs2pickem.live.merge_results` only adds matches or fills in missing winners.
"""

from __future__ import annotations

import difflib
import html as htmllib
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .config import Event, StageConfig
from .live import MatchResult


@dataclass
class ScrapeOutcome:
    matches: list[MatchResult]
    source: str | None
    log: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.source is not None


# ---------------------------------------------------------------- name normalisation
def make_resolver(event: Event, stage: StageConfig) -> Callable[[str], str | None]:
    """Map a scraped name to a stage team (aliases first, then a fuzzy match); None if not in the stage."""
    canon = event.config.canonical
    teams = list(stage.teams)

    def norm(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())

    by_norm = {norm(t): t for t in teams}

    def resolve(name: str) -> str | None:
        c = canon(name.strip())
        if c in teams:
            return c
        n = norm(c)
        if n in by_norm:
            return by_norm[n]
        hit = difflib.get_close_matches(n, list(by_norm), n=1, cutoff=0.85)
        return by_norm[hit[0]] if hit else None

    return resolve


# ---------------------------------------------------------------- HLTV
_HLTV_BLOCK = re.compile(r'<div class="result-con[^"]*"([^>]*)>(.*?)</table>', re.S)


def _strip(s: str) -> str:
    return re.sub(r"\s+", " ", htmllib.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def parse_hltv_results(page: str) -> list[dict]:
    """Finished matches from an HLTV results page, oldest first: {a, b, winner, score, ts, id}."""
    out = []
    for attrs, body in _HLTV_BLOCK.findall(page):
        ts = re.search(r'data-zonedgrouping-entry-unix="(\d+)"', attrs)
        mid = re.search(r'href="/matches/(\d+)/', body)
        teams = re.findall(r'<div class="team( team-won)?\s*">(.*?)</div>', body, re.S)
        if len(teams) != 2:
            continue
        (wa, a), (wb, b) = teams
        if not (wa or wb):
            continue
        score = re.search(r'class="result-score">(.*?)</td>', body, re.S)
        out.append(
            {
                "a": _strip(a),
                "b": _strip(b),
                "winner": _strip(a) if wa else _strip(b),
                "score": _strip(score.group(1)) if score else "",
                "ts": int(ts.group(1)) if ts else 0,
                "id": int(mid.group(1)) if mid else None,
            }
        )
    out.sort(key=lambda m: m["ts"])
    return out


def assign_rounds(rows: list[tuple[str, str, str | None]], existing: list[MatchResult]) -> list[MatchResult]:
    """Infer rounds for chronologically ordered matches: a team's n-th match is round n.

    Holds for Swiss (every active team plays once per round) and single-elimination brackets.
    """
    played: dict[str, int] = {}
    seen = set()
    for r in existing:
        seen.add(frozenset((r.a, r.b)))
        for t in (r.a, r.b):
            played[t] = max(played.get(t, 0), r.round)
    out = []
    for a, b, w in rows:
        key = frozenset((a, b))
        rnd = max(played.get(a, 0), played.get(b, 0)) + 1
        if key in seen:
            rnd = next(r.round for r in existing if frozenset((r.a, r.b)) == key)
        else:
            played[a] = played[b] = rnd
            seen.add(key)
        out.append(MatchResult(rnd, a, b, w))
    return out


def scrape_hltv(
    event: Event, stage: StageConfig, existing: list[MatchResult], refresh: bool = True
) -> list[MatchResult]:
    from .data.hltv import get_page

    if not stage.hltv_event:
        raise RuntimeError("no hltv_event configured")
    page = get_page(f"/results?event={stage.hltv_event}", ttl_hours=0.5, refresh=refresh)
    resolve = make_resolver(event, stage)
    rows = []
    for m in parse_hltv_results(page):
        a, b, w = resolve(m["a"]), resolve(m["b"]), resolve(m["winner"])
        if a and b and w:
            rows.append((a, b, w))
    return assign_rounds(rows, existing)


# ---------------------------------------------------------------- Liquipedia
def scrape_liquipedia(event: Event, stage: StageConfig, refresh: bool = True) -> list[MatchResult]:
    from .data.liquipedia import stage_matches

    if not stage.liquipedia:
        raise RuntimeError("no liquipedia page configured")
    kind = "swiss" if stage.type == "swiss" else "bracket"
    resolve = make_resolver(event, stage)
    out = []
    for m in stage_matches(stage.liquipedia, kind, refresh=refresh, ttl_hours=0.5):
        a, b = resolve(m.a), resolve(m.b)
        if not (a and b) or a == b:
            continue
        w = resolve(m.winner) if m.winner else None
        out.append(MatchResult(m.round, a, b, w))
    return out


# ---------------------------------------------------------------- DFRAG
DFRAG = "https://dfrag.gg/counterstrike/events/{slug}/matches/"
_DFRAG_ROUNDS = {"quarter-final": 1, "semi-final": 2, "grand-final": 3, "final": 3}


def _dfrag_round(match_slug: str) -> int | None:
    m = re.search(r"(?:^|-)round-(\d+)-", match_slug)
    if m:
        return int(m.group(1))
    for k, v in _DFRAG_ROUNDS.items():
        if re.search(rf"(?:^|-){k}-", match_slug):
            return v
    return None


def parse_dfrag(page: str, slug: str) -> list[dict]:
    """Matches from a DFRAG event match list: {round, a, b, winner|None}."""
    pat = re.compile(
        rf'<a href="(/counterstrike/events/{re.escape(slug)}/matches/([^"#]+))(?:#[^"]*)?"[^>]*aria-label="([^"]*)"'
    )
    out, seen = [], set()
    for _href, mslug, label in pat.findall(page):
        mslug = mslug.strip("/")
        if mslug in seen:
            continue
        rnd = _dfrag_round(mslug)
        if rnd is None:
            continue
        label = htmllib.unescape(label)
        pend = re.match(r"Who wins\?\s*(.+?) or (.+)$", label)
        if pend:
            seen.add(mslug)
            out.append({"round": rnd, "a": pend.group(1).strip(), "b": pend.group(2).strip(), "winner": None})
            continue
        fin = re.match(r"(.+?) (\d+), (.+?) (\d+)(?:,|$)", label)
        if fin:
            seen.add(mslug)
            a, sa, b, sb = fin.group(1).strip(), int(fin.group(2)), fin.group(3).strip(), int(fin.group(4))
            w = a if sa > sb else b if sb > sa else None
            out.append({"round": rnd, "a": a, "b": b, "winner": w})
    return out


def scrape_dfrag(event: Event, stage: StageConfig, refresh: bool = True) -> list[MatchResult]:
    from .data.cache import get_text

    if not stage.dfrag:
        raise RuntimeError("no dfrag slug configured")
    page = get_text(
        DFRAG.format(slug=stage.dfrag),
        ttl_hours=0.5,
        refresh=refresh,
        namespace="dfrag",
        headers={"User-Agent": "cs2pickem/0.2 (+https://github.com/ivan-penchev/cs2-pickem)"},
    )
    resolve = make_resolver(event, stage)
    out = []
    for m in parse_dfrag(page, stage.dfrag):
        a, b = resolve(m["a"]), resolve(m["b"])
        if not (a and b) or a == b:
            continue
        out.append(MatchResult(m["round"], a, b, resolve(m["winner"]) if m["winner"] else None))
    return out


# ---------------------------------------------------------------- orchestration
def scrape_stage(
    event: Event, stage: StageConfig, existing: list[MatchResult] | None = None, refresh: bool = True
) -> ScrapeOutcome:
    """Try HLTV → Liquipedia → DFRAG; return the first successful source's matches.

    When HLTV wins, Liquipedia is still consulted (softly) for upcoming pairings and the official rounds.
    """
    existing = existing or []
    log: list[str] = []
    if not stage.teams:
        return ScrapeOutcome([], None, [f"{stage.id}: teams not known yet – nothing to scrape"])

    try:
        hl = scrape_hltv(event, stage, existing, refresh)
        log.append(f"hltv: {len(hl)} finished matches")
        try:
            lp = scrape_liquipedia(event, stage, refresh)
            lp_round = {frozenset((m.a, m.b)): m.round for m in lp}
            for m in hl:
                m.round = lp_round.get(frozenset((m.a, m.b)), m.round)
            done = {frozenset((m.a, m.b)) for m in hl}
            pending = [m for m in lp if frozenset((m.a, m.b)) not in done and m.winner is None]
            log.append(f"liquipedia: +{len(pending)} scheduled pairings")
            hl += pending
        except Exception as e:  # noqa: BLE001
            log.append(f"liquipedia (pairings) failed: {e}")
        return ScrapeOutcome(hl, "hltv", log)
    except Exception as e:  # noqa: BLE001
        log.append(f"hltv failed: {type(e).__name__}: {str(e)[:160]}")

    for name, fn in (("liquipedia", scrape_liquipedia), ("dfrag", scrape_dfrag)):
        try:
            ms = fn(event, stage, refresh)
            log.append(f"{name}: {sum(m.winner is not None for m in ms)} finished, {len(ms)} total")
            return ScrapeOutcome(ms, name, log)
        except Exception as e:  # noqa: BLE001
            log.append(f"{name} failed: {type(e).__name__}: {str(e)[:160]}")
    return ScrapeOutcome([], None, log)


# ---------------------------------------------------------------- update loop
@dataclass
class StageUpdate:
    stage: str
    source: str | None = None
    changes: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)


@dataclass
class UpdateReport:
    ran: bool
    reason: str
    stages: list[StageUpdate] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return any(s.changes for s in self.stages)

    def markdown(self) -> str:
        lines = [f"### cs2pickem live update – {self.reason}", ""]
        for s in self.stages:
            lines.append(f"**{s.stage}** – source: `{s.source or 'none'}`, {len(s.changes)} change(s)")
            lines += [f"- {c}" for c in s.changes]
            lines += [f"  - _{x}_" for x in s.log]
        if not self.stages:
            lines.append("_nothing to do_")
        return "\n".join(lines) + "\n"


def in_window(start, end, today, slack_days: int = 1) -> bool:
    from datetime import timedelta

    if start is None:
        return False
    end = end or start
    return start - timedelta(days=slack_days) <= today <= end + timedelta(days=slack_days)


def _stage_finished(event: Event, stage: StageConfig) -> bool:
    from .live import final_standings, load_results

    if not stage.teams:
        return False
    if stage.type == "swiss":
        return final_standings(event, stage) is not None
    res = load_results(event, stage)
    return any(r.round == 3 and r.winner for r in res)


def update(
    event: Event,
    stage_ids: list[str] | None = None,
    today=None,
    force: bool = False,
    refresh: bool = True,
    scraper: Callable[..., ScrapeOutcome] = scrape_stage,
) -> UpdateReport:
    """One idempotent live-update pass: scrape → merge → save; build the next stage when one finishes.

    Only runs inside the event's date window (± 1 day) unless ``force``. Never deletes results.
    """
    from datetime import date

    from .live import (
        draw_from_results,
        load_results,
        merge_results,
        resolve_teams,
        save_results,
        write_stage_round1,
        write_stage_teams,
    )

    today = today or date.today()
    cfg = event.config
    if not force and not in_window(cfg.start, cfg.end, today):
        return UpdateReport(False, f"{today} is outside the event window {cfg.start} … {cfg.end}")

    order = list(event.stages)
    if stage_ids:
        todo = [s for s in order if s in stage_ids]
    else:
        todo = [s for s in order if in_window(event.stages[s].start, event.stages[s].end, today)]
        if force and not todo:
            todo = [s for s in order if event.stages[s].teams and not _stage_finished(event, event.stages[s])]
            todo = todo[:1]
    report = UpdateReport(True, f"{today}: stages {', '.join(todo) or '–'}")

    queue = list(todo)
    done: set[str] = set()
    while queue:
        sid = queue.pop(0)
        if sid in done:
            continue
        done.add(sid)
        st = event.stages[sid]
        up = StageUpdate(sid)
        report.stages.append(up)
        if not st.teams:
            try:
                teams, how = resolve_teams(event, st)
            except (ValueError, KeyError) as e:
                up.log.append(f"teams not resolvable yet: {e}")
                continue
            write_stage_teams(event, st, teams)
            st.teams = teams
            up.changes.append(f"{sid} seeds set from {how}: " + ", ".join(teams))
        existing = load_results(event, st)
        out = scraper(event, st, existing, refresh=refresh)
        up.source, up.log = out.source, up.log + out.log
        if out.matches:
            merged, changes = merge_results(existing, out.matches)
            if changes:
                save_results(st, merged)
                up.changes += changes
            draw = draw_from_results(st, merged)
            if draw and sorted(map(sorted, draw)) != sorted(map(sorted, st.round1 or [])):
                write_stage_round1(event, st, draw)
                st.round1 = draw
                up.changes.append(f"{sid} Round 1 draw published: " + ", ".join(f"{a}–{b}" for a, b in draw))
        if _stage_finished(event, st):
            up.log.append("stage finished")
            nxt = [
                s for s in order if (event.stages[s].source and event.stages[s].source.qualifiers_from) == sid
            ]
            queue += [s for s in nxt if s not in done]
    return report
