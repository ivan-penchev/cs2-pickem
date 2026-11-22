"""Versioned, committed data snapshots.

Layout::

    data/snapshots/<source>/<YYYY-MM-DD>/<table>.csv   normalized tables
    data/snapshots/<source>/<YYYY-MM-DD>/meta.json     provenance (urls, fetched_at, tool version, row counts)
    data/latest.json                                   manifest: source -> latest snapshot date

All modelling commands read snapshots (offline); only ``cs2pickem fetch`` writes them.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from . import __version__
from .paths import data_dir


def snapshot_root() -> Path:
    return data_dir() / "snapshots"


def manifest_path() -> Path:
    return data_dir() / "latest.json"


def read_manifest() -> dict:
    p = manifest_path()
    return json.loads(p.read_text()) if p.exists() else {}


def write_snapshot(
    source: str,
    day: str,
    tables: dict[str, pd.DataFrame],
    *,
    urls: list[str] | None = None,
    notes: str | None = None,
    extra: dict | None = None,
    manifest: bool = True,
) -> Path:
    """Write a dated snapshot. ``manifest=False`` keeps ``data/latest.json`` untouched (historical data)."""
    d = snapshot_root() / source / day
    d.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(d / f"{name}.csv", index=False)
    meta = {
        "source": source,
        "snapshot_date": day,
        "fetched_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "tool_version": __version__,
        "rows": {name: int(len(df)) for name, df in tables.items()},
        "urls": urls or [],
    }
    if notes:
        meta["notes"] = notes
    if extra:
        meta.update(extra)
    (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if not manifest:
        return d
    man = read_manifest()
    man[source] = {"date": day, "path": str(d.relative_to(data_dir())), "tables": sorted(tables)}
    manifest_path().write_text(json.dumps(dict(sorted(man.items())), indent=2) + "\n")
    return d


def latest_dir(source: str) -> Path | None:
    man = read_manifest()
    if source in man:
        p = data_dir() / man[source]["path"]
        if p.exists():
            return p
    base = snapshot_root() / source
    if base.exists():
        days = sorted(p for p in base.iterdir() if p.is_dir())
        if days:
            return days[-1]
    return None


def load_table(source: str, table: str, day: str | None = None) -> pd.DataFrame | None:
    """Read a snapshot table – the latest one, or the snapshot of a specific ``day`` (YYYY-MM-DD)."""
    d = snapshot_root() / source / day if day else latest_dir(source)
    if d is None or not (d / f"{table}.csv").exists():
        return None
    try:
        return pd.read_csv(d / f"{table}.csv")
    except pd.errors.EmptyDataError:
        return None
