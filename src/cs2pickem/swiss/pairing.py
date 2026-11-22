"""Valve Major Swiss pairing rules (Major Supplemental Rulebook, "Swiss System").

* **Round 1**: the official draw if given, otherwise seed *i* vs seed *i + n/2* (1v9, 2v10, …).
* **Later rounds**: teams with the same W-L record form a group, ordered by

  1. Buchholz "difficulty score" – sum of current (wins − losses) of every opponent faced so far,
  2. initial seed.

* **Rounds 2–3**: the highest team in a group plays the lowest team that is not a rematch, then the
  next highest plays the lowest remaining, … (backtracking so a valid full pairing is always found).
* **Rounds 4–5**: 6-team groups use the first row of :data:`SIX_TEAM_PRIORITY` that has no rematch.
  Other group sizes fall back to the round 2–3 rule.

In every match the team listed first (better group position) is the *higher seed*: Team A in the veto
and the side-chooser in Bo1. A group with an odd number of teams (only possible in non-standard
formats) floats its lowest team down into the next group. If a group cannot avoid a rematch at all,
rematches are allowed for that group (as a last resort).
"""

from __future__ import annotations

from collections.abc import Sequence

SIX_TEAM_PRIORITY: list[list[tuple[int, int]]] = [
    [(1, 6), (2, 5), (3, 4)],
    [(1, 6), (2, 4), (3, 5)],
    [(1, 5), (2, 6), (3, 4)],
    [(1, 5), (2, 4), (3, 6)],
    [(1, 4), (2, 6), (3, 5)],
    [(1, 4), (2, 5), (3, 6)],
    [(1, 6), (2, 3), (4, 5)],
    [(1, 5), (2, 3), (4, 6)],
    [(1, 3), (2, 6), (4, 5)],
    [(1, 3), (2, 5), (4, 6)],
    [(1, 4), (2, 3), (5, 6)],
    [(1, 3), (2, 4), (5, 6)],
    [(1, 2), (3, 6), (4, 5)],
    [(1, 2), (3, 5), (4, 6)],
    [(1, 2), (3, 4), (5, 6)],
]

Pair = tuple[int, int]


def round1_pairs(n: int, explicit: Sequence[Pair] | None = None) -> list[Pair]:
    """Round-1 pairs as team indices (index = initial seed - 1); the better seed first."""
    if explicit:
        return [(min(a, b), max(a, b)) for a, b in explicit]
    h = n // 2
    return [(i, i + h) for i in range(h)]


def buchholz(
    team: int, wins: Sequence[int], losses: Sequence[int], opponents: Sequence[Sequence[int]]
) -> int:
    return sum(wins[o] - losses[o] for o in opponents[team])


def _top_vs_bottom(group: list[int], played: set[frozenset[int]], allow_rematch: bool) -> list[Pair] | None:
    if not group:
        return []
    a, rest = group[0], group[1:]
    for k in range(len(rest) - 1, -1, -1):
        b = rest[k]
        if not allow_rematch and frozenset((a, b)) in played:
            continue
        sub = _top_vs_bottom(rest[:k] + rest[k + 1 :], played, allow_rematch)
        if sub is not None:
            return [(a, b), *sub]
    return None


def _six_table(group: list[int], played: set[frozenset[int]]) -> list[Pair] | None:
    for row in SIX_TEAM_PRIORITY:
        pairs = [(group[i - 1], group[j - 1]) for i, j in row]
        if all(frozenset(p) not in played for p in pairs):
            return pairs
    return None


def pair_group(group: list[int], rnd: int, played: set[frozenset[int]]) -> list[Pair]:
    """Pair an already-ordered group (best first) for round ``rnd`` (1-based)."""
    pairs = None
    if rnd >= 4 and len(group) == 6:
        pairs = _six_table(group, played)
    if pairs is None:
        pairs = _top_vs_bottom(group, played, allow_rematch=False)
    if pairs is None:
        pairs = _top_vs_bottom(group, played, allow_rematch=True)
    assert pairs is not None
    return pairs


def pair_round(
    rnd: int,
    active: Sequence[int],
    wins: Sequence[int],
    losses: Sequence[int],
    opponents: Sequence[Sequence[int]],
    played: set[frozenset[int]],
) -> list[Pair]:
    """Pair round ``rnd`` (≥ 2) for the ``active`` (not yet finished) teams.

    Team indices are initial seeds - 1, so "initial seed" tie-break is the index itself.
    """
    groups: dict[tuple[int, int], list[int]] = {}
    for t in active:
        groups.setdefault((wins[t], losses[t]), []).append(t)
    pairs: list[Pair] = []
    carry: list[int] = []
    # Best records first (most wins, then fewest losses).
    for key in sorted(groups, key=lambda k: (-k[0], k[1])):
        g = sorted(groups[key], key=lambda t: (-buchholz(t, wins, losses, opponents), t))
        g = carry + g
        carry = []
        if len(g) % 2:
            carry = [g.pop()]
        pairs.extend(pair_group(g, rnd, played))
    if carry:
        raise ValueError("odd number of active teams")
    return pairs


def standings_order(
    teams: Sequence[int], wins: Sequence[int], losses: Sequence[int], opponents: Sequence[Sequence[int]]
) -> list[int]:
    """Final ordering: wins desc, losses asc, Buchholz desc, initial seed asc."""
    return sorted(teams, key=lambda t: (-wins[t], losses[t], -buchholz(t, wins, losses, opponents), t))
