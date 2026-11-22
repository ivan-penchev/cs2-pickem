"""HLTV scraping (ranking + team pages) via headless Playwright.

HLTV sits behind Cloudflare: plain HTTP gets a 403 and reusing one browser for many pages trips the
challenge. What works reliably: a *fresh* headless Chromium per page, ``domcontentloaded`` + short wait,
and a polite delay between pages. Rendered HTML is cached in ``.cache/hltv`` so re-runs are free.

Parsing works on the cached HTML with regexes, so it is testable without a browser.
Playwright is an optional dependency (``pip install 'cs2pickem[scrape]' && playwright install chromium``).
"""

from __future__ import annotations

import html as htmllib
import re
import time
from datetime import UTC, datetime

import pandas as pd

from .cache import cache_text, cached_text

BASE = "https://www.hltv.org"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0 Safari/537.36"
)
DELAY_S = 4.0
MAP_NAMES = {"dust2": "Dust II", "dust ii": "Dust II"}


def _render(url: str, wait_ms: int = 3000) -> str:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Playwright not installed: pip install 'cs2pickem[scrape]' && playwright install chromium"
        ) from e
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
        try:
            pg = b.new_context(user_agent=UA, locale="en-US").new_page()
            resp = pg.goto(url, wait_until="domcontentloaded", timeout=45_000)
            pg.wait_for_timeout(wait_ms)
            if resp is None or resp.status >= 400 or "Just a moment" in pg.title():
                raise RuntimeError(f"HLTV blocked {url} (status {resp.status if resp else '?'})")
            return pg.content()
        finally:
            b.close()


def get_page(path: str, ttl_hours: float = 24, refresh: bool = False) -> str:
    url = BASE + path
    if not refresh:
        txt = cached_text(url, namespace="hltv", ttl_hours=ttl_hours)
        if txt is not None:
            return txt
    try:
        txt = _render(url)
    except Exception:
        stale = cached_text(url, namespace="hltv", ttl_hours=None)
        if stale is not None:
            return stale
        raise
    cache_text(url, txt, namespace="hltv")
    time.sleep(DELAY_S)
    return txt


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", htmllib.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


# ---------------------------------------------------------------- ranking
def parse_ranking(page: str) -> pd.DataFrame:
    rows = []
    for block in page.split('class="ranked-team')[1:]:
        pos = re.search(r'class="position[^"]*"[^>]*>\s*#?(\d+)', block)
        name = re.search(r'class="name[^"]*"[^>]*>([^<]+)<', block)
        pts = re.search(r'class="points[^"]*"[^>]*>\s*\((\d+)', block) or re.search(
            r"(\d+) HLTV points", block
        )
        href = re.search(r'href="/team/(\d+)/([^"#?]+)"', block)
        if not (pos and name):
            continue
        rows.append(
            {
                "rank": int(pos.group(1)),
                "team": htmllib.unescape(name.group(1)).strip(),
                "points": int(pts.group(1)) if pts else 0,
                "hltv_id": int(href.group(1)) if href else None,
                "hltv_slug": href.group(2) if href else None,
            }
        )
    return pd.DataFrame(rows)


def ranking(date_path: str | None = None, refresh: bool = False) -> pd.DataFrame:
    """``date_path`` like ``2026/november/16``; default = latest ranking."""
    path = "/ranking/teams" + (f"/{date_path}" if date_path else "")
    return parse_ranking(get_page(path, ttl_hours=24, refresh=refresh))


# ---------------------------------------------------------------- team page
_WDL = re.compile(r"(\d+) Wins? (\d+) Draws? (\d+) Loss(?:es)?")


def parse_team_maps(page: str, team: str) -> pd.DataFrame:
    """'Map win statistics past 3 months' block → per-map W/L (+ pick/ban rates)."""
    rows = []
    for block in page.split('class="map-statistics-row"')[1:]:
        m = re.search(r'map-statistics-row-map-mapname">([^<]+)<', block)
        if not m:
            continue
        txt = _text(block[:6000])
        wdl = _WDL.search(txt)
        if not wdl:
            continue
        picks = re.search(r"Picks (\d+)% of (\d+)", txt)
        bans = re.search(r"Bans (\d+)% of (\d+)", txt)
        name = m.group(1).strip()
        rows.append(
            {
                "team": team,
                "map": MAP_NAMES.get(name.lower(), name),
                "wins": int(wdl.group(1)),
                "losses": int(wdl.group(3)),
                "pick_pct": int(picks.group(1)) if picks else None,
                "ban_pct": int(bans.group(1)) if bans else None,
            }
        )
    return pd.DataFrame(rows)


def parse_team_results(page: str, team: str) -> pd.DataFrame:
    """'Recent results' table → series results (date, opponent, maps won/lost)."""
    i = page.find("Recent results for")
    if i < 0:
        return pd.DataFrame()
    rows = []
    for tr in re.findall(r'<tr class="team-row">(.*?)</tr>', page[i:], flags=re.S):
        unix = re.search(r'data-unix="(\d+)"', tr)
        t1 = re.search(r'team-name team-1">([^<]+)<', tr)
        t2 = re.search(r'team-name team-2">([^<]+)<', tr)
        sc = re.findall(r'class="score[^"]*">\s*(\d+)\s*<', tr)
        if not (unix and t1 and t2 and len(sc) == 2):
            continue
        a, b = htmllib.unescape(t1.group(1)).strip(), htmllib.unescape(t2.group(1)).strip()
        s1, s2 = int(sc[0]), int(sc[1])
        if b == team and a != team:
            a, b, s1, s2 = b, a, s2, s1
        rows.append(
            {
                "date": datetime.fromtimestamp(int(unix.group(1)) / 1000, UTC).date().isoformat(),
                "team": a,
                "opponent": b,
                "maps_won": s1,
                "maps_lost": s2,
            }
        )
    return pd.DataFrame(rows)


def team_page(hltv_id: int, slug: str, refresh: bool = False) -> str:
    return get_page(f"/team/{hltv_id}/{slug}", ttl_hours=24, refresh=refresh)
