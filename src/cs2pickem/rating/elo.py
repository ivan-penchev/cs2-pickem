"""Time-decayed, roster-aware Elo ("form") over recent series results.

Every team starts at its VRS points (VRS is itself an Elo-like scale, so unknown opponents still get a
sensible prior). Series inside the window are replayed chronologically; each update is scaled by

* exponential time decay  ``0.5 ** (age_days / half_life)``
* roster continuity: matches played with < ``min_roster_overlap`` of the current five, or before the
  team's ``roster_changed`` date, count only ``old_roster_weight``.

Only the event teams' ratings are reported; opponents float too so strength of schedule propagates.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..config import EventConfig


def expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def roster_weight(match_roster: str | None, current: set[str], min_overlap: int, old_w: float) -> float:
    if not current or not isinstance(match_roster, str) or not match_roster.strip():
        return 1.0
    players = {p.strip().lower() for p in match_roster.split(",")}
    return 1.0 if len(players & current) >= min_overlap else old_w


def current_rosters(cfg: EventConfig, matches: pd.DataFrame) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for t in cfg.teams:
        if t.roster:
            out[t.name] = {p.lower() for p in t.roster}
    if not matches.empty and "roster" in matches:
        last = matches.dropna(subset=["roster"]).sort_values("date").groupby("team").tail(1)
        for _, r in last.iterrows():
            out.setdefault(r.team, {p.strip().lower() for p in str(r.roster).split(",")})
    return out


def form_elo(
    cfg: EventConfig,
    matches: pd.DataFrame,
    prior: dict[str, float],
    as_of: date,
    default_prior: float = 1200.0,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (rating, effective_n) per team."""
    p = cfg.rating
    if matches.empty:
        return dict(prior), {}
    m = matches.copy()
    m["d"] = pd.to_datetime(m["date"]).dt.date
    m = m[(m["d"] <= as_of) & (m["d"] > as_of.fromordinal(as_of.toordinal() - p.elo_window_days))]
    # each series is reported from both sides when both teams are event teams -> keep one orientation
    a = np.where(m.team < m.opponent, m.team, m.opponent)
    b = np.where(m.team < m.opponent, m.opponent, m.team)
    m = m.assign(_a=a, _b=b).drop_duplicates(["_a", "_b", "date"])
    m = m.sort_values("date")
    rosters = current_rosters(cfg, matches)
    changed = {t.name: date.fromisoformat(t.roster_changed) for t in cfg.teams if t.roster_changed}
    R: dict[str, float] = dict(prior)
    n_eff: dict[str, float] = {}
    for r in m.itertuples():
        t, o = r.team, r.opponent
        ra, rb = R.setdefault(t, prior.get(t, default_prior)), R.setdefault(o, prior.get(o, default_prior))
        age = (as_of - r.d).days
        w = 0.5 ** (age / p.elo_half_life_days) * float(r.weight)
        w *= roster_weight(r.roster, rosters.get(t, set()), p.min_roster_overlap, p.old_roster_weight)
        if t in changed and r.d < changed[t]:
            w *= p.old_roster_weight
        delta = p.elo_k * w * (r.result - expected(ra, rb))
        R[t] = ra + delta
        R[o] = rb - delta
        n_eff[t] = n_eff.get(t, 0.0) + w
        n_eff[o] = n_eff.get(o, 0.0) + w
    return R, n_eff
