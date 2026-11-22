"""Map veto as a two-player zero-sum game, solved exactly by memoised minimax.

Valve Major veto formats (7-map pool, Team A = higher seed / the team that chose to be A):

* **Bo1**: A bans 2, B bans 3, A bans 1 → the remaining map is played.
  The higher seed chooses whether to be Team A or Team B (``bo1_side_choice``).
* **Bo3**: A ban, B ban, A pick, B pick, B ban, A ban → remaining map is the decider.
* **Bo5**: A ban, B ban, A pick, B pick, A pick, B pick → remaining map is the decider.

Each team wants to maximise its own probability of winning the *series*. With independent maps the
series result only depends on the multiset of played maps (not their order), so a state is
``(remaining maps, picked maps)``. ``temperature > 0`` replaces the max/min with a softmax over the
child values (quantal response) to model imperfect or unpredictable vetoes; it is still exact.

For pools that are not 7 maps a generic sequence is used: alternate bans down to ``bo + ...`` maps,
alternate picks for ``bo - 1`` maps, the last one is the decider.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

A, B = 0, 1
BAN, PICK = "ban", "pick"

VETO_7 = {
    1: [(A, BAN), (A, BAN), (B, BAN), (B, BAN), (B, BAN), (A, BAN)],
    3: [(A, BAN), (B, BAN), (A, PICK), (B, PICK), (B, BAN), (A, BAN)],
    5: [(A, BAN), (B, BAN), (A, PICK), (B, PICK), (A, PICK), (B, PICK)],
}


def veto_sequence(n_maps: int, bo: int) -> list[tuple[int, str]]:
    if n_maps == 7 and bo in VETO_7:
        return list(VETO_7[bo])
    if bo > n_maps:
        raise ValueError(f"best-of-{bo} needs at least {bo} maps, pool has {n_maps}")
    n_ban = n_maps - bo
    seq = [((i % 2), BAN) for i in range(n_ban)]
    seq += [((i % 2), PICK) for i in range(bo - 1)]
    return seq


def series_win_prob(p: np.ndarray | list[float], bo: int) -> float:
    """P(team wins majority of ``bo`` independent maps with win probs ``p``)."""
    dist = np.array([1.0])
    for q in p:
        dist = np.concatenate([dist * (1 - q), [0.0]]) + np.concatenate([[0.0], dist * q])
    return float(dist[bo // 2 + 1 :].sum())


@dataclass(frozen=True)
class VetoResult:
    win_prob: float  # P(Team A wins the series)
    maps: tuple[int, ...]  # principal (most likely) line: played map indices, picks first, decider last
    map_probs: np.ndarray  # P(each map is played) under the (possibly stochastic) veto policy


def solve_veto(p: np.ndarray, bo: int, temperature: float = 0.0) -> VetoResult:
    """Solve the veto for Team A with per-map win probabilities ``p`` (length = pool size)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if bo == 1 and n == 1:
        return VetoResult(float(p[0]), (0,), np.ones(1))
    seq = veto_sequence(n, bo)
    full = (1 << n) - 1
    t = float(temperature)

    @cache
    def value(remaining: int, picks: tuple[int, ...]) -> float:
        step = n - bin(remaining).count("1")
        if step == len(seq):
            rest = [i for i in range(n) if remaining >> i & 1]
            return series_win_prob([p[i] for i in (*picks, *rest)], bo)
        actor, _ = seq[step]
        vals = np.array([_child(remaining, picks, i) for i in range(n) if remaining >> i & 1])
        return float(vals @ _policy(vals, actor, t))

    def _child(remaining: int, picks: tuple[int, ...], i: int) -> float:
        step = n - bin(remaining).count("1")
        action = seq[step][1]
        new_picks = tuple(sorted((*picks, i))) if action == PICK else picks
        return value(remaining & ~(1 << i), new_picks)

    # Walk the tree once more for the principal line and the map-play distribution.
    map_probs = np.zeros(n)

    def walk(remaining: int, picks: tuple[int, ...], w: float, line: list[int]) -> list[int] | None:
        step = n - bin(remaining).count("1")
        if step == len(seq):
            rest = [i for i in range(n) if remaining >> i & 1]
            for i in (*picks, *rest):
                map_probs[i] += w
            return [*line, *rest]
        actor, action = seq[step]
        avail = [i for i in range(n) if remaining >> i & 1]
        vals = np.array([_child(remaining, picks, i) for i in avail])
        pol = _policy(vals, actor, t)
        best = int(np.argmax(pol))
        principal = None
        for k, i in enumerate(avail):
            if pol[k] < 1e-9:
                continue
            new_picks = tuple(sorted((*picks, i))) if action == PICK else picks
            new_line = [*line, i] if action == PICK else line
            res = walk(remaining & ~(1 << i), new_picks, w * pol[k], new_line)
            if k == best:
                principal = res
        return principal

    line = walk(full, (), 1.0, [])
    return VetoResult(value(full, ()), tuple(line or ()), map_probs)


def _policy(vals: np.ndarray, actor: int, t: float) -> np.ndarray:
    """Probability of choosing each child. A maximises ``vals``, B minimises."""
    signed = vals if actor == A else -vals
    if t <= 0:
        pol = np.zeros(len(vals))
        pol[int(np.argmax(signed))] = 1.0
        return pol
    z = (signed - signed.max()) / t
    e = np.exp(z)
    return e / e.sum()
