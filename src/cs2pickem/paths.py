"""Repository-relative default locations (overridable via CS2PICKEM_HOME)."""

from __future__ import annotations

import os
from pathlib import Path


def home() -> Path:
    env = os.environ.get("CS2PICKEM_HOME")
    if env:
        return Path(env)
    # walk up from cwd until a directory with configs/ or data/ is found
    cwd = Path.cwd()
    for p in [cwd, *cwd.parents]:
        if (p / "configs").is_dir() and (p / "data").is_dir():
            return p
    return cwd


def data_dir() -> Path:
    return home() / "data"


def cache_dir() -> Path:
    d = home() / ".cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def configs_dir() -> Path:
    return home() / "configs"
