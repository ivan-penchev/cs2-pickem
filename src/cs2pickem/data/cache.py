"""Tiny polite HTTP layer: on-disk cache in .cache/http, per-host rate limiting, gzip + descriptive UA."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from .. import __version__
from ..paths import cache_dir

USER_AGENT = f"cs2pickem/{__version__} (+https://github.com/; CS2 Pick'Em research tool)"
_last_hit: dict[str, float] = {}
MIN_INTERVAL = {"liquipedia.net": 2.5, "www.hltv.org": 4.0}


def _cache_path(url: str, namespace: str) -> Path:
    h = hashlib.sha256(url.encode()).hexdigest()[:24]
    d = cache_dir() / namespace
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{h}.txt"


def throttle(host: str, default: float = 0.3) -> None:
    gap = MIN_INTERVAL.get(host, default)
    wait = _last_hit.get(host, 0.0) + gap - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_hit[host] = time.monotonic()


def get_text(
    url: str,
    *,
    ttl_hours: float = 24.0,
    refresh: bool = False,
    namespace: str = "http",
    headers: dict[str, str] | None = None,
) -> str:
    """GET ``url`` returning text, served from cache when fresh (or when offline and a cached copy exists)."""
    path = _cache_path(url, namespace)
    if path.exists() and not refresh and (time.time() - path.stat().st_mtime) < ttl_hours * 3600:
        return path.read_text()
    throttle(urlparse(url).netloc)
    h = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
    h.update(headers or {})
    try:
        r = httpx.get(url, headers=h, timeout=30, follow_redirects=True)
        r.raise_for_status()
    except httpx.HTTPError:
        if path.exists():
            return path.read_text()
        raise
    path.write_text(r.text)
    return r.text


def cache_text(key: str, text: str, namespace: str) -> Path:
    path = _cache_path(key, namespace)
    path.write_text(text)
    return path


def cached_text(key: str, namespace: str, ttl_hours: float | None = None) -> str | None:
    path = _cache_path(key, namespace)
    if not path.exists():
        return None
    if ttl_hours is not None and (time.time() - path.stat().st_mtime) > ttl_hours * 3600:
        return None
    return path.read_text()
