"""Monte Carlo simulator for a Valve Major Swiss stage (any even field size, N wins / N losses).

Each simulation plays the stage round by round with the exact pairing rules from :mod:`.pairing`.
Series win probabilities come from pre-computed Bo1/Bo3/Bo5 matrices, so a simulation is just pairing
plus a uniform draw per match. Pairings for early rounds are memoised on the result history (there are
only 2^8 distinct round-2 states) (~30k sims/s on one core).

Known results (live mode) are honoured: if a round has fixed pairs they are used as the official pairing
for that round, and fixed winners are forced, so the remaining stage is simulated *conditional* on what
already happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pairing import Pair, buchholz, pair_round, round1_pairs, standings_order


@dataclass(frozen=True)
class SwissFormat:
    n_teams: int = 16
    wins_to_advance: int = 3
    losses_to_eliminate: int = 3
    default_bo: int = 1
    decider_bo: int = 3

    @property
    def max_rounds(self) -> int:
        return self.wins_to_advance + self.losses_to_eliminate - 1

    def best_of(self, wa: int, la: int, wb: int, lb: int) -> int:
        w, lo = self.wins_to_advance - 1, self.losses_to_eliminate - 1
        decider = wa == w or wb == w or la == lo or lb == lo
        return self.decider_bo if decider else self.default_bo


@dataclass
class KnownRound:
    """Fixed pairs (higher seed first is not required) and optional winners for one round."""

    pairs: list[Pair] = field(default_factory=list)
    winners: dict[frozenset[int], int] = field(default_factory=dict)


@dataclass
class SwissResult:
    teams: list[str]
    fmt: SwissFormat
    wins: np.ndarray  # (n_sims, n_teams) int8
    losses: np.ndarray  # (n_sims, n_teams) int8
    rank: np.ndarray  # (n_sims, n_teams) int8 – final standing (0 = 1st)

    @property
    def n_sims(self) -> int:
        return self.wins.shape[0]

    def outcome_masks(self) -> dict[str, np.ndarray]:
        """Boolean (n_sims, n_teams) masks for the Pick'Em categories."""
        W, L = self.fmt.wins_to_advance, self.fmt.losses_to_eliminate
        adv = self.wins == W
        return {
            "perfect": adv & (self.losses == 0),  # 3-0
            "advance": adv & (self.losses > 0),  # 3-1 / 3-2
            "winless": (self.losses == L) & (self.wins == 0),  # 0-3
            "qualified": adv,
            "eliminated": self.losses == L,
        }

    def summary(self) -> dict[str, np.ndarray]:
        m = self.outcome_masks()
        return {k: v.mean(axis=0) for k, v in m.items()}

    def record_distribution(self) -> dict[str, np.ndarray]:
        """``{"3-0": p[team], "3-1": ..., "0-3": ...}`` over all terminal records."""
        out = {}
        W, L = self.fmt.wins_to_advance, self.fmt.losses_to_eliminate
        for w in range(W, -1, -1):
            for lo in range(L + 1):
                if (w == W) == (lo == L) or (w < W and lo < L):
                    continue
                out[f"{w}-{lo}"] = ((self.wins == w) & (self.losses == lo)).mean(axis=0)
        return dict(sorted(out.items(), key=lambda kv: (-int(kv[0][0]), int(kv[0][2]))))


def _simulate_chunk(
    fmt: SwissFormat,
    probs: dict[int, np.ndarray],
    r1: list[Pair],
    known: dict[int, KnownRound],
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    T = fmt.n_teams
    W, L = fmt.wins_to_advance, fmt.losses_to_eliminate
    max_matches = T // 2 * fmt.max_rounds
    U = rng.random((n, max_matches))
    out_w = np.zeros((n, T), np.int8)
    out_l = np.zeros((n, T), np.int8)
    out_r = np.zeros((n, T), np.int8)
    pair_cache: dict[tuple, list[Pair]] = {}
    P = {bo: m.tolist() for bo, m in probs.items()}

    for s in range(n):
        wins = [0] * T
        losses = [0] * T
        opps: list[list[int]] = [[] for _ in range(T)]
        played: set[frozenset[int]] = set()
        history: list[int] = []
        u = U[s]
        k = 0
        for rnd in range(1, fmt.max_rounds + 1):
            active = [t for t in range(T) if wins[t] < W and losses[t] < L]
            if not active:
                break
            kr = known.get(rnd)
            if rnd == 1:
                pairs = list(kr.pairs) if kr and kr.pairs else r1
            elif kr and kr.pairs:
                fixed = {t for p in kr.pairs for t in p}
                rest = [t for t in active if t not in fixed]
                pairs = list(kr.pairs) + (pair_round(rnd, rest, wins, losses, opps, played) if rest else [])
            elif rnd <= 3:
                key = tuple(history)
                pairs = pair_cache.get(key)
                if pairs is None:
                    pairs = pair_round(rnd, active, wins, losses, opps, played)
                    pair_cache[key] = pairs
            else:
                pairs = pair_round(rnd, active, wins, losses, opps, played)
            results = []
            for a, b in pairs:
                bo = fmt.best_of(wins[a], losses[a], wins[b], losses[b])
                forced = kr.winners.get(frozenset((a, b))) if kr else None
                if forced is not None:
                    winner = forced
                else:
                    winner = a if u[k] < P[bo][a][b] else b
                k += 1
                loser = b if winner == a else a
                wins[winner] += 1
                losses[loser] += 1
                opps[a].append(b)
                opps[b].append(a)
                played.add(frozenset((a, b)))
                results.append(winner)
            history.extend(results)
        out_w[s] = wins
        out_l[s] = losses
        order = standings_order(range(T), wins, losses, opps)
        for pos, t in enumerate(order):
            out_r[s, t] = pos
    return out_w, out_l, out_r


def simulate_swiss(
    teams: list[str],
    matrices: dict[int, np.ndarray],
    fmt: SwissFormat | None = None,
    n_sims: int = 100_000,
    seed: int = 2026,
    round1: list[tuple[str, str]] | None = None,
    known: dict[int, list[tuple[str, str, str | None]]] | None = None,
) -> SwissResult:
    """Simulate a Swiss stage.

    ``teams`` is in seed order. ``matrices[bo][i, j]`` = P(i beats j) in a best-of-``bo``.
    ``known`` maps round number → list of ``(team_a, team_b, winner_or_None)``.
    """
    fmt = fmt or SwissFormat(n_teams=len(teams))
    idx = {t: i for i, t in enumerate(teams)}
    r1 = round1_pairs(len(teams), [(idx[a], idx[b]) for a, b in round1] if round1 else None)
    kn: dict[int, KnownRound] = {}
    for rnd, matches in (known or {}).items():
        kr = KnownRound()
        for a, b, w in matches:
            ia, ib = idx[a], idx[b]
            kr.pairs.append((min(ia, ib), max(ia, ib)))
            if w is not None:
                kr.winners[frozenset((ia, ib))] = idx[w]
        kn[int(rnd)] = kr
    # Round 1 known pairs must cover the whole field; otherwise keep the configured draw.
    if 1 in kn and len(kn[1].pairs) < len(teams) // 2:
        kn[1] = KnownRound(pairs=[], winners=kn[1].winners)
    needed = {fmt.default_bo, fmt.decider_bo}
    probs = {bo: np.asarray(matrices[bo]) for bo in needed}

    w, lo, r = _simulate_chunk(fmt, probs, r1, kn, n_sims, seed)
    return SwissResult(list(teams), fmt, w, lo, r)


def play_out(
    teams: list[str],
    fmt: SwissFormat,
    round1: list[tuple[str, str]] | None,
    known: dict[int, list[tuple[str, str, str]]],
) -> tuple[list[int], list[int], list[list[int]], list[Pair] | None, int]:
    """Replay known results deterministically.

    Returns wins, losses, opponents (including finished matches of a partially played round), the
    pairs of the round currently being played (None once the stage is finished) and that round number.
    """
    T = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    wins, losses = [0] * T, [0] * T
    opps: list[list[int]] = [[] for _ in range(T)]
    played: set[frozenset[int]] = set()
    r1 = round1_pairs(T, [(idx[a], idx[b]) for a, b in round1] if round1 else None)
    for rnd in range(1, fmt.max_rounds + 1):
        active = [
            t for t in range(T) if wins[t] < fmt.wins_to_advance and losses[t] < fmt.losses_to_eliminate
        ]
        if not active:
            return wins, losses, opps, None, rnd
        given = known.get(rnd, [])
        if len(given) == len(active) // 2:
            pairs = [(idx[a], idx[b]) for a, b, _ in given]  # official draw wins over our pairing
        else:
            pairs = r1 if rnd == 1 else pair_round(rnd, active, wins, losses, opps, played)
        done = {frozenset((idx[a], idx[b])): idx[w] for a, b, w in given if w}
        complete = len(done) >= len(pairs)
        for a, b, w in given:
            if not w:
                continue
            ia, ib, iw = idx[a], idx[b], idx[w]
            wins[iw] += 1
            losses[ib if iw == ia else ia] += 1
            opps[ia].append(ib)
            opps[ib].append(ia)
            played.add(frozenset((ia, ib)))
        if not complete:  # standings include the finished matches of the round in progress
            return wins, losses, opps, pairs, rnd
    return wins, losses, opps, None, fmt.max_rounds + 1


__all__ = ["KnownRound", "SwissFormat", "SwissResult", "buchholz", "play_out", "simulate_swiss"]
