"""Pairwise series win-probability matrices (Bo1 / Bo3 / Bo5) from ratings + map offsets + veto.

Map win probability for *i* vs *j* on map *m*::

    logit P = ln(10) * (R_i - R_j) / 400 + off_i[m] - off_j[m]

The veto is solved per pairing (see :mod:`.veto`). The team listed earlier in ``teams`` (better seed)
is the higher seed: it is Team A in Bo3/Bo5 and chooses its side in Bo1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import MapParams
from .veto import series_win_prob, solve_veto

LN10_400 = np.log(10) / 400


@dataclass
class MatchModel:
    teams: list[str]  # seed order
    pool: list[str]
    ratings: np.ndarray  # Elo per team (seed order)
    offsets: np.ndarray  # teams x pool logit offsets
    params: MapParams = field(default_factory=MapParams)
    use_veto: bool = True
    _cache: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    @classmethod
    def build(
        cls,
        teams: list[str],
        ratings: dict[str, float],
        offsets: pd.DataFrame | None,
        pool: list[str],
        params: MapParams | None = None,
        use_veto: bool = True,
    ) -> MatchModel:
        r = np.array([ratings[t] for t in teams], dtype=float)
        if offsets is None:
            off = np.zeros((len(teams), len(pool)))
        else:
            off = offsets.reindex(index=teams, columns=pool).fillna(0.0).to_numpy(float)
        return cls(list(teams), list(pool), r, off, params or MapParams(), use_veto)

    def index(self, team: str) -> int:
        return self.teams.index(team)

    def map_probs(self, i: int, j: int) -> np.ndarray:
        logit = LN10_400 * (self.ratings[i] - self.ratings[j]) + self.offsets[i] - self.offsets[j]
        return 1.0 / (1.0 + np.exp(-logit))

    def series(self, i: int, j: int, bo: int) -> float:
        """P(team i beats team j) in a best-of-``bo``. ``i < j`` means *i* is the higher seed."""
        if i == j:
            return 0.5
        hi, lo = (i, j) if i < j else (j, i)
        p = self.map_probs(hi, lo)
        if not self.use_veto:
            q = 1.0 / (1.0 + np.exp(-LN10_400 * (self.ratings[hi] - self.ratings[lo])))
            v = series_win_prob([q] * bo, bo)
        else:
            t = self.params.veto_temperature
            v = solve_veto(p, bo, t).win_prob
            if bo == 1:
                v_as_b = 1.0 - solve_veto(1.0 - p, bo, t).win_prob
                v = max(v, v_as_b) if self.params.bo1_side_choice == "optimal" else 0.5 * (v + v_as_b)
        return v if i == hi else 1.0 - v

    def matrix(self, bo: int) -> np.ndarray:
        """``M[i, j] = P(i beats j)`` in a best-of-``bo`` (cached)."""
        if bo not in self._cache:
            n = len(self.teams)
            m = np.full((n, n), 0.5)
            for i in range(n):
                for j in range(i + 1, n):
                    m[i, j] = self.series(i, j, bo)
                    m[j, i] = 1.0 - m[i, j]
            self._cache[bo] = m
        return self._cache[bo]

    def veto_line(self, i: int, j: int, bo: int) -> list[str]:
        """Most likely played maps (higher seed as Team A), for display."""
        hi, lo = (i, j) if i < j else (j, i)
        res = solve_veto(self.map_probs(hi, lo), bo, self.params.veto_temperature)
        return [self.pool[k] for k in res.maps]
