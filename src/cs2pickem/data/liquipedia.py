"""Liquipedia (MediaWiki API) – tournament results from page wikitext.

Liquipedia asks bots to use the API with a descriptive User-Agent, gzip and at most one ``parse`` request
every ~2 s (we use 2.5 s, see ``cache.MIN_INTERVAL``). Content is CC-BY-SA 3.0 – attribute when reusing.

Wikitext shape (Swiss stage)::

    ====Round 1====
    {{Matchlist|...|matchsection=Round 1
    |{{Match |opponent1={{TeamOpponent|b8}}|opponent2={{TeamOpponent|m80}} |finished=true
        |map1={{Map|map=Mirage|finished=true|t1t=2|t1ct=9|t2t=3|t2ct=10}} }}

Playoffs use ``{{Bracket|...|R1M1={{Match ...}}|R2M1=...}}`` (R = round, M = match within the round).
Map winner = sum of the half (and overtime ``o<k>t1t``…) scores; skipped maps are ``finished=skip``.
Team templates are Liquipedia short codes (``fq``, ``imp``…) → resolved to display names via
``action=expandtemplates`` (cached), so callers only ever see full names.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from urllib.parse import quote

from .cache import get_text

API = "https://liquipedia.net/counterstrike/api.php"
UA = "cs2pickem/0.2 (CS2 Pick'Em research tool; https://github.com/ivan-penchev/cs2-pickem)"
HEADERS = {"User-Agent": UA, "Accept-Encoding": "gzip"}


@dataclass
class LpMap:
    map: str
    score_a: int
    score_b: int

    @property
    def winner(self) -> int:  # 0 = team a, 1 = team b
        return 0 if self.score_a > self.score_b else 1


@dataclass
class LpMatch:
    round: int
    a: str
    b: str
    winner: str | None
    maps: list[LpMap] = field(default_factory=list)
    bo: int = 1
    date: str | None = None
    slot: str | None = None  # bracket slot, e.g. "R1M2"
    hltv_id: int | None = None

    @property
    def finished(self) -> bool:
        return self.winner is not None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["maps"] = [asdict(m) for m in self.maps]
        return d


# ---------------------------------------------------------------- fetching
def page_url(page: str) -> str:
    return f"{API}?action=parse&page={quote(page)}&prop=wikitext&format=json&formatversion=2"


def fetch_wikitext(page: str, *, ttl_hours: float = 1.0, refresh: bool = False) -> str:
    raw = get_text(
        page_url(page), ttl_hours=ttl_hours, refresh=refresh, namespace="liquipedia", headers=HEADERS
    )
    data = json.loads(raw)
    if "error" in data:
        raise RuntimeError(f"Liquipedia: {data['error'].get('info', data['error'])} ({page})")
    wt = data["parse"]["wikitext"]
    return wt["*"] if isinstance(wt, dict) else wt


def resolve_codes(codes: list[str], *, refresh: bool = False) -> dict[str, str]:
    """Team template codes → display names (``{{Team|fq}}`` → ``FlyQuest``), via ``expandtemplates``."""
    out: dict[str, str] = {}
    uniq = sorted({c.strip().lower() for c in codes if c.strip()})
    for i in range(0, len(uniq), 40):
        chunk = uniq[i : i + 40]
        text = " ".join(f"<<{c}>>{{{{Team|{c}}}}}" for c in chunk)
        url = f"{API}?action=expandtemplates&format=json&prop=wikitext&text={quote(text)}"
        raw = get_text(url, ttl_hours=24 * 30, refresh=refresh, namespace="liquipedia", headers=HEADERS)
        wt = json.loads(raw)["expandtemplates"]["wikitext"]
        for code, body in re.findall(r"<<([^>]+)>>(.*?)(?=<<|$)", wt, flags=re.S):
            m = re.search(r'data-highlighting-class="([^"]+)"', body) or re.search(r"\[\[([^|\]]+)\|", body)
            out[code] = m.group(1) if m else code
    return out


# ---------------------------------------------------------------- parsing
def _balanced(text: str, start: int) -> int:
    """Index just past the template starting at ``text[start:]`` (which begins with ``{{``)."""
    depth, i = 0, start
    while i < len(text) - 1:
        two = text[i : i + 2]
        if two == "{{":
            depth += 1
            i += 2
            continue
        if two == "}}":
            depth -= 1
            i += 2
            if depth == 0:
                return i
            continue
        i += 1
    return len(text)


def _top_params(body: str) -> dict[str, str]:
    """Top-level ``|key=value`` params of a template body (nested templates kept intact)."""
    params: dict[str, str] = {}
    depth, cur, i = 0, [], 0
    parts: list[str] = []
    while i < len(body):
        two = body[i : i + 2]
        if two == "{{":
            depth += 1
            cur.append(two)
            i += 2
            continue
        if two == "}}":
            depth -= 1
            cur.append(two)
            i += 2
            continue
        if body[i] == "|" and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(body[i])
        i += 1
    parts.append("".join(cur))
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            params[k.strip()] = v.strip()
    return params


def _opponent(v: str) -> str | None:
    m = re.search(r"\{\{\s*TeamOpponent\s*\|\s*([^|}]+)", v)
    return m.group(1).strip().lower() if m else None


def _int(params: dict[str, str], key: str) -> int:
    try:
        return int(params.get(key, "0") or 0)
    except ValueError:
        return 0


def _map(v: str) -> LpMap | None:
    inner = v.strip()
    if not inner.startswith("{{"):
        return None
    p = _top_params(inner[2:-2])
    if p.get("finished", "").lower() != "true":
        return None
    keys = [k for k in p if re.fullmatch(r"(o\d+)?t[12](t|ct)", k)]
    a = sum(_int(p, k) for k in keys if re.fullmatch(r"(o\d+)?t1(t|ct)", k))
    b = sum(_int(p, k) for k in keys if re.fullmatch(r"(o\d+)?t2(t|ct)", k))
    if a == b and "winner" in p:
        a, b = (1, 0) if p["winner"].strip() == "1" else (0, 1)
    if a == b:
        return None
    return LpMap(p.get("map", "?").strip(), a, b)


def _parse_match(body: str, rnd: int, slot: str | None) -> LpMatch | None:
    p = _top_params(body)
    a, b = _opponent(p.get("opponent1", "")), _opponent(p.get("opponent2", ""))
    if not a or not b:
        return None
    map_keys = sorted((k for k in p if re.fullmatch(r"map\d+", k)), key=lambda k: int(k[3:]))
    maps = [m for k in map_keys if (m := _map(p[k])) is not None]
    bo = int(p["bestof"]) if p.get("bestof", "").isdigit() else max(1, len(map_keys))
    winner = None
    need = bo // 2 + 1
    wa = sum(1 for m in maps if m.winner == 0)
    wb = len(maps) - wa
    if p.get("winner", "").strip() in ("1", "2"):
        winner = a if p["winner"].strip() == "1" else b
    elif wa >= need or wb >= need:
        winner = a if wa > wb else b
    elif p.get("finished", "").lower() == "true" and wa != wb:
        winner = a if wa > wb else b
    date = re.match(r"\s*([A-Za-z]+ \d{1,2}, \d{4})", p.get("date", ""))
    hl = p.get("hltv", "")
    return LpMatch(
        rnd,
        a,
        b,
        winner,
        maps,
        bo,
        date.group(1) if date else None,
        slot,
        int(hl) if hl.isdigit() else None,
    )


def _iter_templates(text: str, name: str):
    for m in re.finditer(r"\{\{\s*" + name + r"\s*\|", text):
        end = _balanced(text, m.start())
        yield m.start(), text[m.start() + 2 : end - 2]


def parse_swiss(wikitext: str) -> list[LpMatch]:
    """All matches of a Swiss stage page, with ``round`` from the enclosing Matchlist/section."""
    out: list[LpMatch] = []
    heads = [(m.start(), int(m.group(1))) for m in re.finditer(r"=+\s*Round\s+(\d+)\s*=+", wikitext)]
    for pos, body in _iter_templates(wikitext, "Matchlist"):
        sec = re.search(r"matchsection\s*=\s*Round\s+(\d+)", body)
        if sec:
            rnd = int(sec.group(1))
        else:
            prior = [r for h, r in heads if h < pos]
            if not prior:
                continue
            rnd = prior[-1]
        for _, mb in _iter_templates(body, "Match"):
            m = _parse_match(mb, rnd, None)
            if m:
                out.append(m)
    return out


def parse_bracket(wikitext: str) -> list[LpMatch]:
    """Single-elimination bracket matches (``R<r>M<m>`` slots); third-place matches are skipped."""
    out: list[LpMatch] = []
    for _, body in _iter_templates(wikitext, "Bracket"):
        params = _top_params(body)
        for k, v in params.items():
            sm = re.fullmatch(r"R(\d+)M(\d+)", k)
            if not sm or not v.lstrip().startswith("{{"):
                continue
            inner = v.strip()
            m = _parse_match(inner[2:-2], int(sm.group(1)), k)
            if m:
                out.append(m)
    out.sort(key=lambda m: (m.round, int(m.slot[m.slot.index("M") + 1 :]) if m.slot else 0))
    return out


def rename(matches: list[LpMatch], names: dict[str, str]) -> list[LpMatch]:
    for m in matches:
        m.a = names.get(m.a, m.a)
        m.b = names.get(m.b, m.b)
        if m.winner:
            m.winner = names.get(m.winner, m.winner)
    return matches


def stage_matches(page: str, kind: str = "swiss", *, refresh: bool = False, ttl_hours: float = 1.0):
    """Fetch + parse + resolve names. ``kind`` = "swiss" | "bracket"."""
    wt = fetch_wikitext(page, refresh=refresh, ttl_hours=ttl_hours)
    ms = parse_swiss(wt) if kind == "swiss" else parse_bracket(wt)
    codes = [c for m in ms for c in (m.a, m.b)]
    return rename(ms, resolve_codes(codes)) if codes else ms
