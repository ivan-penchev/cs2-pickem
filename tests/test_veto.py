from __future__ import annotations

import numpy as np
import pytest

from cs2pickem.maps.veto import BAN, PICK, series_win_prob, solve_veto, veto_sequence


@pytest.mark.parametrize("bo", [1, 3, 5])
def test_sequence_leaves_bo_maps(bo):
    seq = veto_sequence(7, bo)
    assert len(seq) == 6  # one map is always left as decider
    assert sum(a == PICK for _, a in seq) == bo - 1


def test_bo1_and_bo3_orders():
    assert [a for _, a in veto_sequence(7, 1)] == [BAN] * 6
    assert [a for _, a in veto_sequence(7, 3)] == [BAN, BAN, PICK, PICK, BAN, BAN]


def test_series_win_prob():
    assert series_win_prob([0.5], 1) == pytest.approx(0.5)
    assert series_win_prob([0.6, 0.6, 0.6], 3) == pytest.approx(0.6**3 + 3 * 0.6**2 * 0.4)
    assert series_win_prob([1, 0, 0.3], 3) == pytest.approx(0.3)
    assert series_win_prob([0.5] * 5, 5) == pytest.approx(0.5)


def test_uniform_maps_give_coinflip():
    for bo in (1, 3, 5):
        assert solve_veto(np.full(7, 0.5), bo).win_prob == pytest.approx(0.5)


def test_bo1_veto_is_minimax_value():
    rng = np.random.default_rng(1)
    for _ in range(20):
        p = rng.random(7)
        res = solve_veto(p, 1)

        # Brute-force alternating min/max over the ban order A A B B B A
        def val(rem, step, p=p):
            if step == 6:
                return p[rem[0]]
            actor = [0, 0, 1, 1, 1, 0][step]
            vals = [val(rem[:k] + rem[k + 1 :], step + 1, p) for k in range(len(rem))]
            # A bans to maximise its win prob; B to minimise it
            return max(vals) if actor == 0 else min(vals)

        assert res.win_prob == pytest.approx(val(list(range(7)), 0))
        assert len(res.maps) == 1


def test_dominant_maps_and_bounds():
    p = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1])
    r = solve_veto(p, 1)
    assert r.win_prob == pytest.approx(0.9)
    assert 6 not in r.maps
    r3 = solve_veto(p, 3)
    assert r3.map_probs.sum() == pytest.approx(3)
    assert len(r3.maps) == 3
    assert 0.1 < r3.win_prob < 1


def test_veto_monotonic():
    rng = np.random.default_rng(7)
    for bo in (1, 3):
        p = rng.random(7)
        # A's first-ban advantage aside, a strictly better team on every map wins more than a weaker one
        assert solve_veto(np.clip(p + 0.1, 0, 1), bo).win_prob >= solve_veto(p, bo).win_prob - 1e-12


def test_bo3_brute_force_matches():
    p = np.array([0.7, 0.3, 0.55, 0.45, 0.6, 0.4, 0.5])
    seq = veto_sequence(7, 3)

    def val(rem, picks, step):
        if step == len(seq):
            return series_win_prob([p[i] for i in (*picks, *rem)], 3)
        actor, action = seq[step]
        vals = [
            val(rem[:k] + rem[k + 1 :], (*picks, rem[k]) if action == PICK else picks, step + 1)
            for k in range(len(rem))
        ]
        return max(vals) if actor == 0 else min(vals)

    assert solve_veto(p, 3).win_prob == pytest.approx(val(tuple(range(7)), (), 0))
