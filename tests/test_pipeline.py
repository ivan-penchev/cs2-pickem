from __future__ import annotations

import numpy as np
import pytest

from cs2pickem.pipeline import Context, run_swiss
from cs2pickem.rating.elo import expected, roster_weight


def test_elo_helpers():
    assert expected(1500, 1500) == pytest.approx(0.5)
    assert expected(1600, 1500) == pytest.approx(1 - expected(1500, 1600))
    cur = {"a", "b", "c", "d", "e"}
    assert roster_weight("a,b,c,x,y", cur, 3, 0.2) == 1.0
    assert roster_weight("a,b,x,y,z", cur, 3, 0.2) == 0.2
    assert roster_weight(None, cur, 3, 0.2) == 1.0


@pytest.fixture(scope="module")
def ctx(event):
    return Context.load(event)


def test_ratings_cover_stage1(ctx, event):
    r = ctx.ratings.as_dict()
    for t in event.stage("stage1").teams:
        assert t in r and np.isfinite(r[t])


def test_match_model_is_consistent(ctx, event):
    teams = list(event.stage("stage1").teams)
    m = ctx.model(teams)
    for bo in (1, 3):
        M = m.matrix(bo)
        assert M.shape == (16, 16)
        assert ((M >= 0) & (M <= 1)).all()
        off = ~np.eye(16, dtype=bool)
        # P(A beats B) + P(B beats A) ≈ 1 (veto side asymmetry is small)
        assert np.abs((M + M.T)[off] - 1).max() < 0.05


def test_stage1_end_to_end(ctx, event):
    res = run_swiss(ctx, event.stage("stage1"), n_sims=300, use_results=False)
    s = res.summary()
    assert s["qualified"].sum() == pytest.approx(8)
    assert s["perfect"].sum() == pytest.approx(2)
