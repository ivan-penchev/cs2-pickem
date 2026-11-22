"""Pick'Em optimiser for the playoff bracket.

Picks must form a consistent bracket (a semi-final pick must be one of your quarter-final winners, etc.).
Expected correct picks is additive over matches, so the optimum is found exactly with a DP over the
bracket tree: ``best(node, t)`` = P(t wins node) + best(child containing t, t) + max_u best(other child, u).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..playoffs import BracketResult


@dataclass
class BracketPicks:
    teams: list[str]
    picks: list[list[int]]  # picks[r][m] = team index picked to win match m of round r
    labels: list[str]
    probs: list[list[float]]  # marginal P(pick correct)
    expected: float
    p_all: float
    distribution: np.ndarray

    def named(self) -> list[list[str]]:
        return [[self.teams[t] for t in r] for r in self.picks]


def optimise_bracket(res: BracketResult) -> BracketPicks:
    T = len(res.teams)
    # p[r][m][t] = P(team t wins match m of round r)
    p = [[np.bincount(w, minlength=T) / w.shape[0] for w in rnd] for rnd in res.rounds]
    n_rounds = len(p)
    # value[r][m] = array over teams of best expected score of the subtree rooted at (r, m) given t wins it
    value: list[list[np.ndarray]] = [[pm.copy() for pm in p[0]]]
    for r in range(1, n_rounds):
        row = []
        for m in range(len(p[r])):
            left, right = value[r - 1][2 * m], value[r - 1][2 * m + 1]
            reach_l = p[r - 1][2 * m] > 0
            reach_r = p[r - 1][2 * m + 1] > 0
            v = np.full(T, -np.inf)
            v[reach_l] = p[r][m][reach_l] + left[reach_l] + right[reach_r].max(initial=0.0)
            v[reach_r] = p[r][m][reach_r] + right[reach_r] + left[reach_l].max(initial=0.0)
            row.append(v)
        value.append(row)

    picks: list[list[int]] = [[0] * len(p[r]) for r in range(n_rounds)]

    def assign(r: int, m: int, t: int) -> None:
        picks[r][m] = t
        if r == 0:
            return
        lc, rc = 2 * m, 2 * m + 1
        if p[r - 1][lc][t] > 0:
            assign(r - 1, lc, t)
            assign(r - 1, rc, int(np.argmax(value[r - 1][rc])))
        else:
            assign(r - 1, rc, t)
            assign(r - 1, lc, int(np.argmax(value[r - 1][lc])))

    root = int(np.argmax(value[-1][0]))
    assign(n_rounds - 1, 0, root)
    probs = [[float(p[r][m][picks[r][m]]) for m in range(len(p[r]))] for r in range(n_rounds)]
    score = sum(
        (res.rounds[r][m] == picks[r][m]).astype(np.int8) for r in range(n_rounds) for m in range(len(p[r]))
    )
    n = sum(len(x) for x in p)
    dist = np.bincount(score, minlength=n + 1)[: n + 1] / score.shape[0]
    return BracketPicks(
        list(res.teams), picks, res.labels, probs, float(value[-1][0][root]), float(dist[-1]), dist
    )
