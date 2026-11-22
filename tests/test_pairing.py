from __future__ import annotations

import random

import pytest

from cs2pickem.swiss.pairing import SIX_TEAM_PRIORITY, pair_group, pair_round, round1_pairs, standings_order
from cs2pickem.swiss.simulator import SwissFormat


def test_round1_default_is_i_vs_i_plus_half():
    assert round1_pairs(16) == [(i, i + 8) for i in range(8)]


def test_round1_explicit_orders_better_seed_first():
    assert round1_pairs(4, [(3, 0), (1, 2)]) == [(0, 3), (1, 2)]


def test_top_vs_bottom_and_rematch_avoidance():
    g = [0, 1, 2, 3]
    assert pair_group(g, 2, set()) == [(0, 3), (1, 2)]
    # 0 already played 3 -> 0 plays the lowest non-rematch (2)
    assert pair_group(g, 2, {frozenset((0, 3))}) == [(0, 2), (1, 3)]


def test_six_team_table_first_valid_row():
    g = [10, 11, 12, 13, 14, 15]
    assert pair_group(g, 4, set()) == [(10, 15), (11, 14), (12, 13)]
    played = {frozenset((10, 15))}
    row = next(r for r in SIX_TEAM_PRIORITY if (1, 6) not in r)
    assert pair_group(g, 4, played) == [(g[i - 1], g[j - 1]) for i, j in row]


def test_bo3_only_for_deciders():
    f = SwissFormat()
    assert f.best_of(0, 0, 0, 0) == 1
    assert f.best_of(1, 1, 1, 1) == 1
    assert f.best_of(2, 0, 2, 0) == 3  # advancement match
    assert f.best_of(0, 2, 0, 2) == 3  # elimination match
    assert f.best_of(2, 2, 2, 2) == 3
    assert f.best_of(2, 1, 2, 1) == 3
    assert f.best_of(1, 2, 1, 2) == 3


def test_standings_order_buchholz_then_seed():
    wins, losses = [1, 1, 0, 0], [0, 0, 1, 1]
    opps = [[2], [3], [0], [1]]
    assert standings_order(range(4), wins, losses, opps) == [0, 1, 2, 3]


def _random_swiss(seed: int, fmt: SwissFormat):
    rng = random.Random(seed)
    T = fmt.n_teams
    wins, losses = [0] * T, [0] * T
    opps: list[list[int]] = [[] for _ in range(T)]
    played: set[frozenset[int]] = set()
    rounds = []
    for rnd in range(1, fmt.max_rounds + 1):
        active = [t for t in range(T) if wins[t] < 3 and losses[t] < 3]
        if not active:
            break
        pairs = round1_pairs(T) if rnd == 1 else pair_round(rnd, active, wins, losses, opps, played)
        rounds.append([])
        for a, b in pairs:
            assert (wins[a], losses[a]) == (wins[b], losses[b]), "teams must share a record"
            bo = fmt.best_of(wins[a], losses[a], wins[b], losses[b])
            w = a if rng.random() < 0.5 else b
            rounds[-1].append((a, b, bo))
            assert frozenset((a, b)) not in played, "rematch"
            played.add(frozenset((a, b)))
            opps[a].append(b)
            opps[b].append(a)
            wins[w] += 1
            losses[b if w == a else a] += 1
        # every active team plays exactly once per round
        seen = [t for a, b in pairs for t in (a, b)]
        assert sorted(seen) == sorted(active)
    return wins, losses, rounds


@pytest.mark.parametrize("seed", range(300))
def test_full_swiss_invariants(seed):
    fmt = SwissFormat()
    wins, losses, rounds = _random_swiss(seed, fmt)
    assert len(rounds) == 5
    assert sum(w == 3 for w in wins) == 8 and sum(lo == 3 for lo in losses) == 8
    assert all(w == 3 or lo == 3 for w, lo in zip(wins, losses, strict=True))
    assert sum(w == 3 and lo == 0 for w, lo in zip(wins, losses, strict=True)) == 2
    assert sum(w == 0 and lo == 3 for w, lo in zip(wins, losses, strict=True)) == 2
    # Bo1 in rounds 1-2, Bo3 for every match in round 4-5 (all teams at 2W or 2L)
    assert all(bo == 1 for r in rounds[:2] for *_, bo in r)
    assert [bo for *_, bo in rounds[2]].count(3) == 4  # 2-0 and 0-2 matches in round 3
    assert all(bo == 3 for r in rounds[3:] for *_, bo in r)
