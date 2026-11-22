"""Time-decayed Bradley-Terry ratings with a VRS prior (MAP estimate, Newton's method).

Every series in the window (event teams *and* their opponents) is a weighted logistic observation::

    P(i beats j) = sigmoid(theta_i - theta_j)
    weight       = 0.5 ** (age_days / half_life) * roster_weight

Each team's prior is ``N(mu_i, 1 / l2)`` with ``mu_i = prior_weight * (VRS_i - median VRS)`` in logit
units (VRS points are Elo-like, so ``ln(10)/400`` per point). Unlike the sequential Elo, the fit is
order-independent and uses the whole schedule at once, so strength of schedule propagates fully.
Thetas are returned on the Elo scale (``1500 + 400/ln(10) * theta``).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..config import EventConfig
from .elo import current_rosters, roster_weight

LOGIT_PER_ELO = np.log(10) / 400


def fit_bt(
    i: np.ndarray, j: np.ndarray, y: np.ndarray, w: np.ndarray, mu: np.ndarray, l2: float, iters: int = 50
) -> np.ndarray:
    """MAP Bradley-Terry thetas for ``n = len(mu)`` teams (``i`` beat ``j`` with prob ``sigmoid(t_i - t_j)``)."""
    n = len(mu)
    th = mu.astype(float).copy()
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(th[i] - th[j])))
        r = w * (p - y)
        g = l2 * (th - mu)
        np.add.at(g, i, r)
        np.add.at(g, j, -r)
        h = w * p * (1 - p)
        hess = l2 * np.eye(n)
        np.add.at(hess, (i, i), h)
        np.add.at(hess, (j, j), h)
        np.add.at(hess, (i, j), -h)
        np.add.at(hess, (j, i), -h)
        step = np.linalg.solve(hess, g)
        th -= step
        if np.abs(step).max() < 1e-9:
            break
    return th


def bt_ratings(
    cfg: EventConfig, matches: pd.DataFrame, vrs: pd.Series, as_of: date
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (Elo-scale rating, effective number of series) for every team seen or in ``vrs``."""
    p = cfg.rating
    base = float(vrs.median()) if len(vrs) else 0.0
    floor = float(vrs.min()) - 100 if len(vrs) else base
    m = matches.copy() if matches is not None else pd.DataFrame()
    if not m.empty:
        m["d"] = pd.to_datetime(m["date"]).dt.date
        lo = as_of.fromordinal(as_of.toordinal() - p.bt_window_days)
        m = m[(m["d"] < as_of) & (m["d"] > lo)]
        a = np.where(m.team < m.opponent, m.team, m.opponent)
        b = np.where(m.team < m.opponent, m.opponent, m.team)
        m = m.assign(_a=a, _b=b).drop_duplicates(["_a", "_b", "date"])
    teams = sorted(set(vrs.index) | (set(m.team) | set(m.opponent) if not m.empty else set()))
    ix = {t: k for k, t in enumerate(teams)}
    mu = np.array([p.bt_prior_weight * (float(vrs.get(t, floor)) - base) * LOGIT_PER_ELO for t in teams])
    n_eff: dict[str, float] = {}
    if m.empty:
        th = mu
    else:
        rosters = current_rosters(cfg, matches)
        changed = {t.name: date.fromisoformat(t.roster_changed) for t in cfg.teams if t.roster_changed}
        w = np.empty(len(m))
        for k, r in enumerate(m.itertuples()):
            mw = getattr(r, "weight", 1.0)
            wt = 0.5 ** ((as_of - r.d).days / p.bt_half_life_days) * (1.0 if pd.isna(mw) else float(mw))
            wt *= roster_weight(
                r.roster, rosters.get(r.team, set()), p.min_roster_overlap, p.old_roster_weight
            )
            if r.team in changed and r.d < changed[r.team]:
                wt *= p.old_roster_weight
            w[k] = wt
            n_eff[r.team] = n_eff.get(r.team, 0.0) + wt
            n_eff[r.opponent] = n_eff.get(r.opponent, 0.0) + wt
        th = fit_bt(
            m.team.map(ix).to_numpy(),
            m.opponent.map(ix).to_numpy(),
            m.result.to_numpy(float),
            w,
            mu,
            p.bt_l2,
        )
    return {t: 1500 + th[ix[t]] / LOGIT_PER_ELO for t in teams}, n_eff
