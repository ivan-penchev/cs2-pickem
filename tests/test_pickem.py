from __future__ import annotations

import itertools

import numpy as np
import pytest

from cs2pickem.pickem.bracket_opt import optimise_bracket
from cs2pickem.pickem.swiss_opt import optimise_swiss
from cs2pickem.playoffs import simulate_bracket
from cs2pickem.swiss.simulator import simulate_swiss
from helpers import logistic

BRACKET = [[1, 8], [4, 5], [2, 7], [3, 6]]


@pytest.fixture(scope="module")
def swiss_res():
    teams = [f"T{i + 1}" for i in range(16)]
    M = logistic([1800 - 35 * i for i in range(16)])
    return simulate_swiss(teams, {1: M, 3: M}, n_sims=4000, seed=11)


def _score(res, picks, thr=5):
    m = res.outcome_masks()
    correct = sum(m[c][:, t].astype(int) for c, ts in picks.items() for t in ts)
    return (correct >= thr).mean(), correct.mean()


def test_swiss_picks_are_valid_and_scored(swiss_res):
    sp = optimise_swiss(swiss_res, restarts=3)
    flat = [t for ts in sp.picks.values() for t in ts]
    assert len(flat) == len(set(flat)) == 10
    assert [len(sp.picks[c]) for c in ("perfect", "winless", "advance")] == [2, 2, 6]
    p, e = _score(swiss_res, sp.picks)
    assert sp.p_threshold == pytest.approx(p)
    assert sp.expected == pytest.approx(e)
    assert sp.distribution.sum() == pytest.approx(1)
    # strongest teams in 3-0, weakest in 0-3
    assert set(sp.picks["perfect"]) <= set(range(5))
    assert set(sp.picks["winless"]) <= set(range(10, 16))


def test_swiss_beats_naive_and_random(swiss_res):
    sp = optimise_swiss(swiss_res, restarts=3)
    naive = {"perfect": [0, 1], "winless": [14, 15], "advance": [2, 3, 4, 5, 6, 7]}
    assert sp.p_threshold >= _score(swiss_res, naive)[0] - 1e-12
    rng = np.random.default_rng(0)
    for _ in range(30):
        perm = rng.permutation(16).tolist()
        rnd = {"perfect": perm[:2], "winless": perm[2:4], "advance": perm[4:10]}
        assert sp.p_threshold >= _score(swiss_res, rnd)[0]


def test_swiss_locked_respected(swiss_res):
    sp = optimise_swiss(swiss_res, restarts=2, locked={"perfect": ["T10"]})
    assert 9 in sp.picks["perfect"]


def test_bracket_dp_equals_brute_force():
    teams = [f"S{i + 1}" for i in range(8)]
    M = logistic([1700, 1690, 1650, 1640, 1600, 1580, 1560, 1500])
    res = simulate_bracket(teams, {3: M, 5: M}, BRACKET, n_sims=4000, seed=2)
    bp = optimise_bracket(res)
    T = len(teams)
    p = [[np.bincount(w, minlength=T) / w.shape[0] for w in rnd] for rnd in res.rounds]
    qf = [(BRACKET[m][0] - 1, BRACKET[m][1] - 1) for m in range(4)]
    best = -1.0
    for q in itertools.product(*qf):
        for s in itertools.product((q[0], q[1]), (q[2], q[3])):
            for f in s:
                v = sum(p[0][m][q[m]] for m in range(4)) + p[1][0][s[0]] + p[1][1][s[1]] + p[2][0][f]
                best = max(best, v)
    assert bp.expected == pytest.approx(best)
    assert bp.picks[2][0] in bp.picks[1] and set(bp.picks[1]) <= set(bp.picks[0])


def test_bracket_known_and_equal():
    teams = [f"S{i + 1}" for i in range(8)]
    M = np.full((8, 8), 0.5)
    res = simulate_bracket(teams, {3: M, 5: M}, BRACKET, n_sims=8000, seed=3, known={(0, 0): "S8"})
    reach = res.reach()
    assert reach[res.labels[0]][7] == pytest.approx(1) and reach[res.labels[0]][0] == 0
    assert reach["Final"].sum() == pytest.approx(1)
    assert np.allclose(reach["Final"][1:7], 1 / 8, atol=0.02)
    assert reach["Final"][7] == pytest.approx(1 / 4, abs=0.02)  # S8 already won its QF


def test_swiss_expected_objective(swiss_res):
    se = optimise_swiss(swiss_res, restarts=3, objective="expected")
    st = optimise_swiss(swiss_res, restarts=3)
    assert se.expected >= st.expected - 1e-12
    assert se.p_threshold == pytest.approx(_score(swiss_res, se.picks)[0])
    with pytest.raises(ValueError):
        optimise_swiss(swiss_res, objective="nope")


def test_swiss_seed_anchor_within_tolerance(swiss_res):
    base = optimise_swiss(swiss_res, restarts=3)
    # reversed seed order: anchoring must not cost more than the tolerance
    rank = list(range(15, -1, -1))
    anc = optimise_swiss(swiss_res, restarts=3, seed_rank=rank, tie_tol=0.01)
    assert anc.p_threshold >= base.p_threshold - 0.01 - 1e-12
    flat = [t for ts in anc.picks.values() for t in ts]
    assert len(set(flat)) == 10


def test_swiss_seed_anchor_breaks_exact_ties():
    # identical teams: every pick set ties, so the anchor follows the seed order
    teams = [f"T{i + 1}" for i in range(16)]
    M = logistic([1500] * 16)
    res = simulate_swiss(teams, {1: M, 3: M}, n_sims=2000, seed=3)
    sp = optimise_swiss(res, restarts=2, seed_rank=list(range(16)), tie_tol=1.0)
    assert set(sp.picks["perfect"]) == {0, 1}
    assert set(sp.picks["winless"]) == {14, 15}
