from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cs2pickem.inputs import load_inputs
from cs2pickem.rating.blend import compute_ratings
from cs2pickem.rating.bt import fit_bt


def _synthetic(n_teams=6, n_games=4000, seed=0):
    rng = np.random.default_rng(seed)
    true = np.linspace(-1.5, 1.5, n_teams)
    i = rng.integers(0, n_teams, n_games)
    j = (i + rng.integers(1, n_teams, n_games)) % n_teams
    y = (rng.random(n_games) < 1 / (1 + np.exp(-(true[i] - true[j])))).astype(float)
    return true, i, j, y


def test_bt_recovers_strengths():
    true, i, j, y = _synthetic()
    th = fit_bt(i, j, y, np.ones(len(y)), np.zeros(len(true)), l2=0.01)
    assert list(np.argsort(th)) == list(range(len(true)))
    assert np.allclose(th - th.mean(), true - true.mean(), atol=0.2)


def test_bt_symmetry_and_prior():
    # one game each way between two teams -> equal; no data -> prior mean
    th = fit_bt(np.array([0, 1]), np.array([1, 0]), np.array([1.0, 1.0]), np.ones(2), np.zeros(3), l2=1.0)
    assert th[0] == pytest.approx(th[1])
    assert th[2] == pytest.approx(0.0)
    mu = np.array([0.3, -0.3])
    assert np.allclose(fit_bt(np.array([0]), np.array([1]), np.array([1.0]), np.zeros(1), mu, l2=2.0), mu)


def test_bt_prior_pulls_towards_vrs():
    th_weak = fit_bt(
        np.array([0]), np.array([1]), np.array([0.0]), np.ones(1), np.array([1.0, -1.0]), l2=100.0
    )
    th_free = fit_bt(
        np.array([0]), np.array([1]), np.array([0.0]), np.ones(1), np.array([1.0, -1.0]), l2=0.01
    )
    assert th_weak[0] > th_weak[1] and th_free[0] < th_free[1]


def test_method_switch_and_temperature(event):
    cfg = event.config
    inp = load_inputs(cfg, cutoff=event.stage("stage1").start.isoformat())
    out = {}
    for name, upd in {
        "blend": {"method": "blend", "temperature": 1.0},
        "blend_t": {"method": "blend", "temperature": 0.5},
        "bt": {"method": "bt", "temperature": 1.0},
    }.items():
        c = cfg.model_copy(update={"rating": cfg.rating.model_copy(update=upd)})
        out[name] = compute_ratings(c, inp).table
    assert "bt" in out["bt"].columns and "bt" not in out["blend"].columns
    a, b = out["blend"], out["blend_t"].reindex(out["blend"].index)
    spread = (a["rating"] - a["override"] - 1500).abs().sum()
    assert spread > 0
    assert np.allclose(b["rating"] - b["override"] - 1500, 0.5 * (a["rating"] - a["override"] - 1500))
    assert pd.Series(out["bt"]["rating"]).notna().all()
