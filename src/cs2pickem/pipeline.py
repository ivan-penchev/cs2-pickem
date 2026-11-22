"""High-level glue: event → ratings → match model → stage simulation → Pick'Em."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from .config import Event, StageConfig
from .inputs import Inputs, load_inputs
from .live import known_by_round, load_results, resolve_teams, swiss_format
from .maps import MatchModel, map_offsets
from .playoffs import BracketResult, simulate_bracket
from .rating.blend import Ratings, compute_ratings
from .swiss.simulator import SwissResult, simulate_swiss


@dataclass
class Context:
    event: Event
    inputs: Inputs
    ratings: Ratings

    @classmethod
    def load(cls, event: Event, as_of: date | None = None, extra_matches=None) -> Context:
        """``as_of`` = point-in-time cut-off: only matches strictly before that day are used."""
        cutoff = as_of.isoformat() if as_of else None
        inp = load_inputs(event.config, cutoff=cutoff, extra_matches=extra_matches)
        return cls(event, inp, compute_ratings(event.config, inp, as_of=as_of))

    def model(self, teams: list[str], use_veto: bool = True) -> MatchModel:
        cfg = self.event.config
        off = map_offsets(self.inputs.maps, teams, cfg.map_pool, cfg.maps)
        return MatchModel.build(teams, self.ratings.as_dict(), off, cfg.map_pool, cfg.maps, use_veto)


def stage_teams(ctx: Context, stage: StageConfig) -> tuple[list[str], str]:
    """Seed order for a stage; if the feeder stage isn't finished, project its qualifiers by simulation."""
    try:
        return resolve_teams(ctx.event, stage)
    except ValueError:
        if not stage.source or not stage.source.qualifiers_from:
            raise
    prev = ctx.event.stage(stage.source.qualifiers_from)
    prev_teams, prev_how = stage_teams(ctx, prev)
    if prev_how != "projected":
        try:
            return resolve_teams(ctx.event, stage, prev_teams=prev_teams)
        except ValueError:
            pass
    prev = prev.model_copy(update={"teams": prev_teams})
    n = min(ctx.event.config.sim.n_sims, 20_000)
    if prev.type == "swiss":
        res = run_swiss(ctx, prev, n_sims=n)
        q = res.summary()["qualified"]
        mean_rank = res.rank.mean(axis=0)
        k = len(prev_teams) // 2
        top = np.argsort(-q, kind="stable")[:k]
        projected = [prev_teams[i] for i in sorted(top, key=lambda i: mean_rank[i])]
    else:
        raise ValueError(f"cannot project qualifiers from a {prev.type} stage")
    return resolve_teams(ctx.event, stage, projected, prev_teams=prev_teams)


def run_swiss(
    ctx: Context,
    stage: StageConfig,
    n_sims: int | None = None,
    use_results: bool = True,
    use_veto: bool = True,
    record_runs: int = 0,
) -> SwissResult:
    teams = stage.teams or stage_teams(ctx, stage)[0]
    stage = stage.model_copy(update={"teams": teams})
    mm = ctx.model(teams, use_veto)
    fmt = swiss_format(stage)
    mats = {bo: mm.matrix(bo) for bo in {fmt.default_bo, fmt.decider_bo}}
    known = known_by_round(load_results(ctx.event, stage)) if use_results else None
    sim = ctx.event.config.sim
    return simulate_swiss(
        teams,
        mats,
        fmt,
        n_sims=n_sims or sim.n_sims,
        seed=sim.seed,
        round1=[tuple(p) for p in stage.round1] if stage.round1 else None,  # type: ignore[misc]
        known=known,
        record_runs=record_runs,
    )


def playoff_known(stage: StageConfig, teams: list[str], results) -> dict[tuple[int, int], str]:
    """Map recorded playoff results to (round_index, match_index) slots of the bracket."""
    slots: list[set[str]] = [{teams[a - 1], teams[b - 1]} for a, b in stage.bracket]
    known: dict[tuple[int, int], str] = {}
    n_rounds = int(np.log2(len(stage.bracket))) + 1
    for r in range(n_rounds):
        size = 2**r
        for m in range(len(stage.bracket) // size):
            members = set().union(*slots[m * size : (m + 1) * size])
            for res in results:
                if res.round == r + 1 and res.winner and {res.a, res.b} <= members:
                    known[(r, m)] = res.winner
    return known


def run_playoffs(ctx: Context, stage: StageConfig, n_sims: int | None = None) -> tuple[BracketResult, str]:
    teams, how = stage_teams(ctx, stage)
    mm = ctx.model(teams)
    known = playoff_known(stage, teams, load_results(ctx.event, stage))
    sim = ctx.event.config.sim
    res = simulate_bracket(
        teams,
        {stage.playoff_bo: mm.matrix(stage.playoff_bo), stage.final_bo: mm.matrix(stage.final_bo)},
        stage.bracket,
        bo=stage.playoff_bo,
        final_bo=stage.final_bo,
        n_sims=n_sims or sim.n_sims,
        seed=sim.seed,
        known=known,
    )
    return res, how
