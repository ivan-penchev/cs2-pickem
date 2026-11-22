"""Pick'Em optimiser for a Swiss stage.

Valve scoring: ``n_30`` teams that go 3-0, ``n_03`` teams that go 0-3 and ``n_advance`` teams that
advance with 3-1/3-2. A team can be picked only once, and the coin needs ``threshold`` correct picks.

Outcomes are strongly correlated (16 teams share 8 R1 matches, exactly two teams go 3-0, ...), so the
objective is evaluated on the joint simulation samples rather than on marginals:

    maximise  P(#correct >= threshold)   (tie-break: E[#correct])

Search: start from the best marginal assignment (Hungarian-style greedy), then steepest-ascent local
search over "replace one pick" and "swap two picks' categories" moves, plus a few random restarts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..swiss.simulator import SwissResult

CATS = ("perfect", "winless", "advance")
CAT_LABEL = {"perfect": "3-0", "winless": "0-3", "advance": "3-1/3-2"}


@dataclass
class SwissPicks:
    teams: list[str]
    picks: dict[str, list[int]]  # category -> team indices
    marginals: dict[str, list[float]]
    p_threshold: float
    expected: float
    distribution: np.ndarray  # P(k correct), k = 0..n
    threshold: int

    def named(self) -> dict[str, list[str]]:
        return {c: [self.teams[t] for t in ts] for c, ts in self.picks.items()}


class _Evaluator:
    def __init__(self, masks: dict[str, np.ndarray], threshold: int, eps: float = 1e-4) -> None:
        self.cols = {c: masks[c].astype(np.int8) for c in CATS}
        self.threshold = threshold
        self.eps = eps

    def score(self, assign: list[tuple[str, int]]) -> np.ndarray:
        s = np.zeros(next(iter(self.cols.values())).shape[0], np.int8)
        for c, t in assign:
            s += self.cols[c][:, t]
        return s

    def objective(self, s: np.ndarray) -> float:
        return float((s >= self.threshold).mean()) + self.eps * float(s.mean())


def _initial(marg: dict[str, np.ndarray], quota: dict[str, int], rng: np.random.Generator | None) -> list:
    T = len(marg["perfect"])
    noise = (lambda: rng.normal(0, 0.05, T)) if rng is not None else (lambda: 0.0)
    cand = sorted(
        (
            (float(marg[c][t] + (noise()[t] if rng is not None else 0.0)), c, t)
            for c in CATS
            for t in range(T)
        ),
        reverse=True,
    )
    used: set[int] = set()
    left = dict(quota)
    out = []
    for _, c, t in cand:
        if left[c] and t not in used:
            out.append((c, t))
            used.add(t)
            left[c] -= 1
    return out


def _local_search(
    ev: _Evaluator,
    assign: list[tuple[str, int]],
    T: int,
    fixed: list[tuple[str, int]] = (),  # type: ignore[assignment]
    max_iter: int = 200,
) -> list:
    """Steepest ascent over the free picks ``assign``; ``fixed`` picks count towards the score."""
    assign = list(assign)
    s = ev.score(list(fixed) + assign)
    best = ev.objective(s)
    for _ in range(max_iter):
        used = {t for _, t in assign} | {t for _, t in fixed}
        move, move_val = None, best
        # Replace pick k with an unused team in the same category.
        for k, (c, t) in enumerate(assign):
            base = s - ev.cols[c][:, t]
            for u in range(T):
                if u in used:
                    continue
                v = ev.objective(base + ev.cols[c][:, u])
                if v > move_val + 1e-12:
                    move, move_val = ("rep", k, u), v
        # Swap the teams of two picks in different categories.
        for a in range(len(assign)):
            for b in range(a + 1, len(assign)):
                (ca, ta), (cb, tb) = assign[a], assign[b]
                if ca == cb:
                    continue
                ns = s - ev.cols[ca][:, ta] - ev.cols[cb][:, tb] + ev.cols[ca][:, tb] + ev.cols[cb][:, ta]
                v = ev.objective(ns)
                if v > move_val + 1e-12:
                    move, move_val = ("swap", a, b), v
        if move is None:
            break
        if move[0] == "rep":
            _, k, u = move
            c, t = assign[k]
            s = s - ev.cols[c][:, t] + ev.cols[c][:, u]
            assign[k] = (c, u)
        else:
            _, a, b = move
            (ca, ta), (cb, tb) = assign[a], assign[b]
            s = s - ev.cols[ca][:, ta] - ev.cols[cb][:, tb] + ev.cols[ca][:, tb] + ev.cols[cb][:, ta]
            assign[a], assign[b] = (ca, tb), (cb, ta)
        best = move_val
    return assign


def optimise_swiss(
    res: SwissResult,
    n_30: int = 2,
    n_03: int = 2,
    n_advance: int = 6,
    threshold: int = 5,
    restarts: int = 8,
    seed: int = 0,
    locked: dict[str, list[str]] | None = None,
) -> SwissPicks:
    """Return the pick set maximising P(correct >= threshold).

    ``locked`` optionally fixes some picks (``{"perfect": ["NiP"]}``) – they are kept in every candidate.
    """
    masks = res.outcome_masks()
    T = len(res.teams)
    quota = {"perfect": n_30, "winless": n_03, "advance": n_advance}
    if n_30 + n_03 + n_advance > T:
        raise ValueError("more picks than teams")
    marg = {c: masks[c].mean(axis=0) for c in CATS}
    ev = _Evaluator(masks, threshold)
    idx = {t: i for i, t in enumerate(res.teams)}
    fixed = [(c, idx[t]) for c, ts in (locked or {}).items() for t in ts]
    for c, _ in fixed:
        quota[c] -= 1
    if any(v < 0 for v in quota.values()):
        raise ValueError("too many locked picks for a category")

    rng = np.random.default_rng(seed)
    best_assign, best_val = None, -1.0
    for r in range(restarts + 1):
        free_marg = {c: marg[c].copy() for c in CATS}
        for _, t in fixed:
            for c in CATS:
                free_marg[c][t] = -1.0
        start = _initial(free_marg, quota, None if r == 0 else rng)
        assign = fixed + _local_search(ev, start, T, fixed)
        val = ev.objective(ev.score(assign))
        if val > best_val + 1e-12:
            best_assign, best_val = assign, val

    assert best_assign is not None
    s = ev.score(best_assign)
    n = n_30 + n_03 + n_advance
    picks = {c: sorted((t for cc, t in best_assign if cc == c), key=lambda t: -marg[c][t]) for c in CATS}
    return SwissPicks(
        teams=list(res.teams),
        picks=picks,
        marginals={c: [float(marg[c][t]) for t in picks[c]] for c in CATS},
        p_threshold=float((s >= threshold).mean()),
        expected=float(s.mean()),
        distribution=np.bincount(s, minlength=n + 1)[: n + 1] / s.shape[0],
        threshold=threshold,
    )
