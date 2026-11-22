from __future__ import annotations

from pathlib import Path

import pytest

from cs2pickem.config import load_event

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def event():
    return load_event(ROOT / "configs" / "singapore2026")
