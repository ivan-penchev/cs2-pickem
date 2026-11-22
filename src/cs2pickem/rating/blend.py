"""Blend VRS points, HLTV points and form Elo into one Elo-scale strength per team.

Each component is z-scored across the teams being rated (HLTV on log scale – its points are very
top-heavy). Missing components are dropped and the remaining weights renormalised per team. The blended
z is mapped to Elo with ``rating_scale`` points per 1 sd, centred at 1500, then manual ``delta`` from
``data/ratings_override.csv`` is added.

With ``rating.method: bt`` the strength is instead a time-decayed Bradley-Terry fit with a VRS prior
(see :mod:`.bt`); the blend components are still reported for reference. Either way the spread around
1500 is multiplied by ``rating.temperature`` (backtests show the raw spread is overconfident).

Elo difference → map win probability: ``1 / (1 + 10^(-diff/400))``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from ..config import EventConfig
from ..inputs import Inputs
from .bt import bt_ratings
from .elo import form_elo


@dataclass
class Ratings:
    table: pd.DataFrame  # index team; columns vrs, hltv, elo, z_*, override, rating

    def rating(self, team: str) -> float:
        return float(self.table.loc[team, "rating"])

    def as_dict(self) -> dict[str, float]:
        return self.table["rating"].to_dict()


def _z(s: pd.Series) -> pd.Series:
    v = s.dropna()
    if len(v) < 2 or v.std(ddof=0) == 0:
        return s * 0
    return (s - v.mean()) / v.std(ddof=0)


def compute_ratings(
    cfg: EventConfig, inp: Inputs, teams: list[str] | None = None, as_of: date | None = None
) -> Ratings:
    """Rate all event teams (so the scale is identical across stages) plus any extra ``teams``."""
    p = cfg.rating
    teams = list(dict.fromkeys([t.name for t in cfg.teams] + list(teams or [])))
    as_of = as_of or cfg.start or date.today()
    vrs = inp.vrs_standings.set_index("team")["points"].astype(float)
    hltv = inp.hltv_ranking.set_index("team")["points"].astype(float)

    # Elo prior: VRS points for everyone in the standings (opponents included), fallback = low.
    floor = float(vrs.quantile(0.5)) if len(vrs) else 1200.0
    prior = {t: float(v) for t, v in vrs.items()}
    elo, n_eff = form_elo(cfg, inp.matches, prior, as_of, default_prior=floor)

    df = pd.DataFrame(index=pd.Index(teams, name="team"))
    df["vrs"] = vrs.reindex(teams)
    df["hltv"] = hltv.reindex(teams)
    df["elo"] = pd.Series({t: elo.get(t, np.nan) for t in teams})
    df["n_matches"] = pd.Series({t: round(n_eff.get(t, 0.0), 2) for t in teams})
    df["z_vrs"] = _z(df["vrs"])
    df["z_hltv"] = _z(np.log1p(df["hltv"]))
    df["z_elo"] = _z(df["elo"])
    w = {"z_vrs": p.w_vrs, "z_hltv": p.w_hltv, "z_elo": p.w_elo}
    num = sum(df[c].fillna(0) * wt for c, wt in w.items())
    den = sum(df[c].notna() * wt for c, wt in w.items()).replace(0, np.nan)
    df["z"] = (num / den).fillna(df[list(w)].mean(axis=1)).fillna(-1.0)
    if p.method == "bt":
        bt, bt_n = bt_ratings(cfg, inp.matches, vrs, as_of)
        df["bt"] = pd.Series({t: bt.get(t, 1300.0) for t in teams})
        df["n_matches"] = pd.Series({t: round(bt_n.get(t, 0.0), 2) for t in teams})
        raw = df["bt"]
    else:
        raw = 1500 + p.rating_scale * df["z"]
    ov = inp.overrides.groupby("team")["delta"].sum() if not inp.overrides.empty else pd.Series(dtype=float)
    df["override"] = ov.reindex(teams).fillna(0.0)
    df["rating"] = 1500 + p.temperature * (raw - 1500) + df["override"]
    return Ratings(df.sort_values("rating", ascending=False))
