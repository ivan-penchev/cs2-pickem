"""Per-team, per-map strength offsets (logit scale), shrunk towards the team's overall map record.

For team *t* with overall map record ``W/N`` (all pool maps) and record ``w/n`` on map *m*::

    base   = (W + 1) / (N + 2)                          # team's average map win rate
    rate_m = (w + k * base) / (n + k)                   # Bayesian shrinkage, k = prior_maps
    off_m  = clamp(logit(rate_m) - logit(base), ±max_offset)

Offsets are relative to the team's own average, so the overall strength stays in the blended rating and
the offsets only redistribute it across maps (e.g. a team that is great on Nuke but never plays Anubis).
Maps with no games get ``unplayed_offset`` (usually a permaban – mildly negative).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import MapParams


def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def map_offsets(maps: pd.DataFrame, teams: list[str], pool: list[str], params: MapParams) -> pd.DataFrame:
    """Return a ``teams x pool`` DataFrame of logit offsets."""
    out = pd.DataFrame(0.0, index=pd.Index(teams, name="team"), columns=pool)
    if maps is None or maps.empty:
        return out
    df = maps[maps["map"].isin(pool)]
    k = params.prior_maps
    for t in teams:
        rows = df[df["team"] == t]
        if rows.empty:
            continue
        w_tot, n_tot = float(rows["wins"].sum()), float((rows["wins"] + rows["losses"]).sum())
        base = (w_tot + 1) / (n_tot + 2)
        rec = rows.groupby("map")[["wins", "losses"]].sum()
        for m in pool:
            if m not in rec.index or rec.loc[m].sum() == 0:
                out.loc[t, m] = params.unplayed_offset
                continue
            w, n = float(rec.loc[m, "wins"]), float(rec.loc[m].sum())
            rate = (w + k * base) / (n + k)
            out.loc[t, m] = float(np.clip(_logit(rate) - _logit(base), -params.max_offset, params.max_offset))
    return out
