from __future__ import annotations

import numpy as np
import pytest

from cs2pickem.swiss.simulator import SwissFormat, play_out, simulate_swiss
from helpers import logistic


def _mats(M):
    return {1: M, 3: M}


@pytest.fixture(scope="module")
def strong_res():
    teams = [f"T{i + 1}" for i in range(16)]
    M = logistic([1800 - 40 * i for i in range(16)])
    return simulate_swiss(teams, _mats(M), n_sims=3000, seed=1)


def test_per_sim_category_counts(strong_res):
    m = strong_res.outcome_masks()
    assert (m["perfect"].sum(1) == 2).all()
    assert (m["winless"].sum(1) == 2).all()
    assert (m["qualified"].sum(1) == 8).all()
    assert (m["eliminated"].sum(1) == 8).all()
    assert (m["qualified"] ^ m["eliminated"]).all()
    assert (np.sort(strong_res.rank, axis=1) == np.arange(16)).all()


def test_probabilities_sum(strong_res):
    s = strong_res.summary()
    assert s["perfect"].sum() == pytest.approx(2)
    assert s["winless"].sum() == pytest.approx(2)
    assert s["qualified"].sum() == pytest.approx(8)
    rd = strong_res.record_distribution()
    assert set(rd) == {"3-0", "3-1", "3-2", "2-3", "1-3", "0-3"}
    assert np.allclose(sum(rd.values()), 1)


def test_stronger_teams_do_better(strong_res):
    q = strong_res.summary()["qualified"]
    assert q[0] > 0.9 and q[-1] < 0.1
    assert q[:4].mean() > q[-4:].mean()


def test_equal_strength_symmetry():
    teams = [f"T{i}" for i in range(16)]
    res = simulate_swiss(teams, _mats(np.full((16, 16), 0.5)), n_sims=8000, seed=3)
    s = res.summary()
    assert np.allclose(s["qualified"], 0.5, atol=0.04)
    assert np.allclose(s["perfect"], 2 / 16, atol=0.03)
    assert np.allclose(s["winless"], 2 / 16, atol=0.03)


def test_known_results_are_forced():
    teams = [f"T{i}" for i in range(16)]
    known = {1: [(f"T{i}", f"T{i + 8}", f"T{i + 8}") for i in range(8)]}
    res = simulate_swiss(teams, _mats(np.full((16, 16), 0.5)), n_sims=500, seed=4, known=known)
    assert (res.wins[:, 8:] >= 1).all()
    assert (res.losses[:, :8] >= 1).all()
    assert res.summary()["perfect"][:8].sum() == 0


def test_bo3_matrix_used_for_deciders():
    teams = [f"T{i}" for i in range(16)]
    # Bo1 is a coin flip, Bo3 always won by the better seed -> seed 0..7 can never be eliminated at 2-2 by worse
    M1 = np.full((16, 16), 0.5)
    M3 = (np.arange(16)[:, None] < np.arange(16)[None, :]).astype(float)
    res = simulate_swiss(teams, {1: M1, 3: M3}, n_sims=500, seed=5)
    # T0 wins every Bo3; it can only lose Bo1 matches (rounds 1-2) → it always qualifies
    assert (res.wins[:, 0] == 3).all()


def test_play_out_partial_round():
    teams = [f"T{i}" for i in range(16)]
    fmt = SwissFormat()
    known = {1: [(f"T{i}", f"T{i + 8}", f"T{i}") for i in range(8)]}
    known[2] = [("T0", "T7", "T0")]  # only one round-2 match finished
    wins, losses, opps, nxt, rnd = play_out(teams, fmt, None, known)
    assert rnd == 2
    assert wins[0] == 2 and losses[7] == 1
    assert nxt is not None and (0, 7) in [tuple(sorted(p)) for p in nxt]
    assert len(nxt) == 8
