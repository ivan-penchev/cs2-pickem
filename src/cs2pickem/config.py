"""Event / stage configuration (YAML) loaded into pydantic models.

An *event* directory (e.g. ``configs/singapore2026``) contains:

* ``event.yaml``  – event metadata, map pool, team registry (aliases, HLTV ids, rosters), model params
* ``<stage>.yaml`` – one file per stage (Swiss stages and the playoff bracket)

Everything event-specific lives in those files so a new Major only needs new YAML.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

DEFAULT_MAP_POOL = ["Ancient", "Anubis", "Cache", "Dust II", "Inferno", "Mirage", "Nuke"]


class TeamInfo(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    tag: str | None = None  # short display tag (badges in the HTML visualisation)
    vrs_name: str | None = None
    hltv_id: int | None = None
    hltv_slug: str | None = None
    roster: list[str] = Field(default_factory=list)
    roster_changed: str | None = None  # ISO date of last roster change (older matches down-weighted)
    region: str | None = None

    def all_names(self) -> list[str]:
        names = [self.name, *self.aliases]
        if self.vrs_name:
            names.append(self.vrs_name)
        return names


class RatingParams(BaseModel):
    """Weights of the strength blend (z-scored components) and Elo settings."""

    w_vrs: float = 0.40
    w_hltv: float = 0.25
    w_elo: float = 0.35
    elo_k: float = 32.0
    elo_half_life_days: float = 30.0
    elo_window_days: int = 90
    min_roster_overlap: int = 4  # players of current roster that must have played for a match to fully count
    old_roster_weight: float = 0.35
    rating_scale: float = 200.0  # Elo points per 1.0 blended z-score
    lan_bonus: float = 0.0


class MapParams(BaseModel):
    prior_maps: float = 8.0  # Bayesian shrinkage strength (pseudo-maps at team-average)
    max_offset: float = 0.8  # clamp of per-map logit offset
    # 0 = both teams veto optimally (minimax); >0 = quantal-response (softmax over win-prob, in
    # probability units) – models imperfect/unpredictable vetoes. Computed exactly, no sampling.
    veto_temperature: float = 0.0
    bo1_side_choice: Literal["optimal", "random"] = "optimal"  # higher seed picks Team A/B in Bo1
    unplayed_offset: float = -0.25  # logit offset for maps a team never plays (usually its permaban)


class SimParams(BaseModel):
    n_sims: int = 100_000
    seed: int = 2026


class PickemRules(BaseModel):
    n_30: int = 2
    n_03: int = 2
    n_advance: int = 6
    threshold: int = 5


class EventConfig(BaseModel):
    id: str
    name: str
    start: date | None = None
    map_pool: list[str] = Field(default_factory=lambda: list(DEFAULT_MAP_POOL))
    vrs_date: str | None = None  # VRS standings file date, e.g. 2026_11_02
    teams: list[TeamInfo] = Field(default_factory=list)
    rating: RatingParams = Field(default_factory=RatingParams)
    maps: MapParams = Field(default_factory=MapParams)
    sim: SimParams = Field(default_factory=SimParams)
    pickem: PickemRules = Field(default_factory=PickemRules)
    stages: list[str] = Field(default_factory=list)

    # ---- name resolution -------------------------------------------------
    def canonical(self, name: str) -> str:
        key = _norm(name)
        for t in self.teams:
            if any(_norm(n) == key for n in t.all_names()):
                return t.name
        return name

    def team(self, name: str) -> TeamInfo:
        c = self.canonical(name)
        for t in self.teams:
            if t.name == c:
                return t
        return TeamInfo(name=c)


class StageSource(BaseModel):
    """How a stage's 16 teams are assembled when not listed explicitly."""

    qualifiers_from: str | None = None  # previous stage id
    invites: list[str] = Field(default_factory=list)  # seeded 1..k by VRS (list order = seed order)


class StageConfig(BaseModel):
    id: str
    name: str
    type: Literal["swiss", "playoffs"] = "swiss"
    start: date | None = None
    teams: list[str] = Field(default_factory=list)  # seed order (seed 1 first)
    source: StageSource | None = None
    # Swiss
    wins_to_advance: int = 3
    losses_to_eliminate: int = 3
    decider_bo: int = 3  # best-of for advancement/elimination matches
    default_bo: int = 1  # best-of for everything else
    # Playoffs
    playoff_bo: int = 3
    final_bo: int = 5
    bracket: list[list[int]] = Field(default_factory=lambda: [[1, 8], [4, 5], [2, 7], [3, 6]])
    results_file: str | None = None
    # Optional explicit Round-1 pairs (official draw). Default: seed i vs seed i+n/2.
    round1: list[list[str]] | None = None

    @model_validator(mode="after")
    def _check(self) -> StageConfig:
        if self.type == "swiss" and self.teams and len(self.teams) % 2:
            raise ValueError("Swiss stage needs an even number of teams")
        if self.round1 and any(len(p) != 2 for p in self.round1):
            raise ValueError("round1 entries must be pairs")
        return self


class Event(BaseModel):
    """Loaded event (config + stages + root directory)."""

    root: Path
    config: EventConfig
    stages: dict[str, StageConfig]

    def stage(self, stage_id: str) -> StageConfig:
        try:
            return self.stages[stage_id]
        except KeyError as e:
            raise KeyError(f"unknown stage {stage_id!r}; known: {list(self.stages)}") from e


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def load_event(path: str | Path) -> Event:
    root = Path(path)
    if root.is_file():
        root = root.parent
    cfg = EventConfig.model_validate(yaml.safe_load((root / "event.yaml").read_text()))
    stage_ids = cfg.stages or sorted(p.stem for p in root.glob("*.yaml") if p.stem != "event")
    stages: dict[str, StageConfig] = {}
    for sid in stage_ids:
        raw = yaml.safe_load((root / f"{sid}.yaml").read_text())
        raw.setdefault("id", sid)
        st = StageConfig.model_validate(raw)
        st.teams = [cfg.canonical(t) for t in st.teams]
        if st.round1:
            st.round1 = [[cfg.canonical(a), cfg.canonical(b)] for a, b in st.round1]
            flat = [t for pair in st.round1 for t in pair]
            if st.teams and sorted(flat) != sorted(st.teams):
                raise ValueError(f"{sid}: round1 must pair every stage team exactly once")
        if st.source:
            st.source.invites = [cfg.canonical(t) for t in st.source.invites]
        stages[sid] = st
    return Event(root=root, config=cfg, stages=stages)
