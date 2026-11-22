"""Single-elimination playoff bracket simulator.

The bracket is given as first-round seed pairs (1-based), e.g. ``[[1, 8], [4, 5], [2, 7], [3, 6]]``;
winners of adjacent matches meet in the next round. All matches are ``bo`` except the final (``final_bo``).
Known winners (live mode) are forced, everything else is simulated, so the output is conditional on them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BracketResult:
    teams: list[str]  # seed order
    rounds: list[list[np.ndarray]]  # rounds[r][m] = (n_sims,) index of the winner of match m in round r
    labels: list[str]  # round names

    @property
    def n_sims(self) -> int:
        return self.rounds[0][0].shape[0]

    def reach(self) -> dict[str, np.ndarray]:
        """P(team wins a match in round r) for every round, keyed by round label."""
        T = len(self.teams)
        out = {}
        for lab, rnd in zip(self.labels, self.rounds, strict=True):
            p = np.zeros(T)
            for w in rnd:
                p += np.bincount(w, minlength=T) / w.shape[0]
            out[lab] = p
        return out

    def match_probs(self, r: int, m: int) -> dict[str, float]:
        w = self.rounds[r][m]
        c = np.bincount(w, minlength=len(self.teams)) / w.shape[0]
        return {self.teams[i]: float(c[i]) for i in np.nonzero(c)[0]}


def round_labels(n_rounds: int) -> list[str]:
    names = {1: "Final", 2: "Semi-finals", 3: "Quarter-finals", 4: "Round of 16"}
    return [names.get(n_rounds - r, f"Round {r + 1}") for r in range(n_rounds)]


def simulate_bracket(
    teams: list[str],
    matrices: dict[int, np.ndarray],
    bracket: list[list[int]],
    bo: int = 3,
    final_bo: int = 5,
    n_sims: int = 100_000,
    seed: int = 2026,
    known: dict[tuple[int, int], str] | None = None,
) -> BracketResult:
    """``matrices[bo][i, j]`` = P(seed-index i beats j). ``known[(round, match)] = winner``."""
    n_first = len(bracket)
    if n_first & (n_first - 1):
        raise ValueError("bracket must have a power-of-two number of first-round matches")
    idx = {t: i for i, t in enumerate(teams)}
    rng = np.random.default_rng(seed)
    n_rounds = int(np.log2(n_first)) + 1
    known = known or {}
    current = [(np.full(n_sims, a - 1), np.full(n_sims, b - 1)) for a, b in bracket]
    rounds: list[list[np.ndarray]] = []
    for r in range(n_rounds):
        is_final = r == n_rounds - 1
        P = np.asarray(matrices[final_bo if is_final else bo])
        winners = []
        for m, (a, b) in enumerate(current):
            # Higher seed (lower index) is team A for the veto model.
            hi, lo = np.minimum(a, b), np.maximum(a, b)
            if (r, m) in known:
                w = np.full(n_sims, idx[known[(r, m)]])
            else:
                w = np.where(rng.random(n_sims) < P[hi, lo], hi, lo)
            winners.append(w)
        rounds.append(winners)
        current = [(winners[2 * k], winners[2 * k + 1]) for k in range(len(winners) // 2)]
    return BracketResult(list(teams), rounds, round_labels(n_rounds))
