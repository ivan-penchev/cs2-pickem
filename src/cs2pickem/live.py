"""Live mode: completed match results and stage-to-stage hand-over.

Results live in ``data/results/<stage>.yaml`` (path configurable via ``results_file``)::

    matches:
      - {round: 1, a: Ninjas in Pyjamas, b: M80, winner: Ninjas in Pyjamas}
      - {round: 1, a: Liquid, b: GOSH, winner: GOSH}

For playoffs ``round`` is 1 = quarter-finals, 2 = semi-finals, 3 = final. Team names may be aliases.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import Event, StageConfig
from .paths import home
from .swiss.pairing import standings_order
from .swiss.simulator import SwissFormat, play_out


@dataclass
class MatchResult:
    round: int
    a: str
    b: str
    winner: str | None = None

    def as_dict(self) -> dict:
        return {"round": self.round, "a": self.a, "b": self.b, "winner": self.winner}


def results_path(stage: StageConfig) -> Path:
    rel = stage.results_file or f"data/results/{stage.id}.yaml"
    p = Path(rel)
    return p if p.is_absolute() else home() / p


def load_results(event: Event, stage: StageConfig) -> list[MatchResult]:
    p = results_path(stage)
    if not p.exists():
        return []
    raw = yaml.safe_load(p.read_text()) or {}
    out = []
    canon = event.config.canonical
    for m in raw.get("matches", []) or []:
        w = m.get("winner")
        r = MatchResult(int(m["round"]), canon(m["a"]), canon(m["b"]), canon(w) if w else None)
        if r.winner and r.winner not in (r.a, r.b):
            raise ValueError(f"{p}: winner {r.winner!r} is not in match {r.a} vs {r.b}")
        out.append(r)
    return out


def save_results(stage: StageConfig, results: list[MatchResult]) -> Path:
    p = results_path(stage)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = {"matches": [r.as_dict() for r in sorted(results, key=lambda r: r.round)]}
    p.write_text(
        f"# {stage.name} results – edit by hand or use `cs2pickem result add`\n"
        + yaml.safe_dump(body, sort_keys=False, allow_unicode=True)
    )
    return p


def add_result(
    event: Event, stage: StageConfig, a: str, b: str, winner: str, rnd: int | None = None
) -> MatchResult:
    """Record ``winner`` of ``a`` vs ``b``. The round is inferred from the current standings if omitted."""
    canon = event.config.canonical
    a, b, winner = canon(a), canon(b), canon(winner)
    if winner not in (a, b):
        raise ValueError(f"winner {winner!r} must be {a!r} or {b!r}")
    results = load_results(event, stage)
    for r in results:
        if {r.a, r.b} == {a, b}:
            raise ValueError(f"{a} vs {b} already recorded (round {r.round})")
    if rnd is None:
        if stage.type == "swiss":
            fmt = swiss_format(stage)
            known = known_by_round(results)
            *_, rnd = play_out(stage.teams, fmt, stage.round1, known)  # type: ignore[arg-type]
        else:
            rnd = 1 + sum(1 for r in results if a in (r.a, r.b) or b in (r.a, r.b)) // 2
    for t in (a, b):
        if t not in stage.teams:
            raise ValueError(f"{t!r} is not in {stage.id}")
    res = MatchResult(rnd, a, b, winner)
    save_results(stage, [*results, res])
    return res


def known_by_round(results: list[MatchResult]) -> dict[int, list[tuple[str, str, str | None]]]:
    out: dict[int, list[tuple[str, str, str | None]]] = {}
    for r in results:
        out.setdefault(r.round, []).append((r.a, r.b, r.winner))
    return out


def swiss_format(stage: StageConfig) -> SwissFormat:
    return SwissFormat(
        n_teams=len(stage.teams),
        wins_to_advance=stage.wins_to_advance,
        losses_to_eliminate=stage.losses_to_eliminate,
        default_bo=stage.default_bo,
        decider_bo=stage.decider_bo,
    )


def final_standings(event: Event, stage: StageConfig) -> list[str] | None:
    """Final order of a *completed* Swiss stage (None if still running)."""
    fmt = swiss_format(stage)
    known = known_by_round(load_results(event, stage))
    wins, losses, opps, nxt, _ = play_out(stage.teams, fmt, stage.round1, known)  # type: ignore[arg-type]
    if nxt is not None or any(
        w < fmt.wins_to_advance and lo < fmt.losses_to_eliminate for w, lo in zip(wins, losses, strict=True)
    ):
        return None
    order = standings_order(range(len(stage.teams)), wins, losses, opps)
    return [stage.teams[i] for i in order]


def qualifiers(event: Event, stage: StageConfig) -> list[str] | None:
    order = final_standings(event, stage)
    if order is None:
        return None
    return order[: len(order) // 2]


def resolve_teams(
    event: Event,
    stage: StageConfig,
    projected: list[str] | None = None,
    prev_teams: list[str] | None = None,
) -> tuple[list[str], str]:
    """Seed order for ``stage``. Returns (teams, how) where how ∈ {"config", "results", "projected"}.

    Stage N+1 = direct invites (seeds 1..k) + Stage N qualifiers (seeds k+1.., ordered by final standing).
    ``projected`` – qualifiers in projected order, used when the previous stage is still running.
    ``prev_teams`` – already-resolved seed order of the previous stage (may itself be projected).
    """
    if stage.teams:
        return list(stage.teams), "config"
    if not stage.source or not stage.source.qualifiers_from:
        raise ValueError(f"{stage.id}: no teams and no source configured")
    prev = event.stage(stage.source.qualifiers_from)
    if prev_teams is None:
        prev_teams, _ = resolve_teams(event, prev)
    prev = prev.model_copy(update={"teams": prev_teams})
    q = qualifiers(event, prev) if prev.type == "swiss" else None
    how = "results"
    if q is None:
        if projected is None:
            raise ValueError(
                f"{stage.id}: {prev.id} is not finished – enter its results (`cs2pickem result add`) "
                "or use projected qualifiers"
            )
        q, how = projected, "projected"
    return list(stage.source.invites) + list(q), how


def write_stage_teams(event: Event, stage: StageConfig, teams: list[str]) -> Path:
    """Persist the resolved seed order into the stage YAML (`next-stage`)."""
    p = event.root / f"{stage.id}.yaml"
    text = p.read_text()
    block = "teams:\n" + "".join(f"  - {t}\n" for t in teams)
    lines = text.splitlines(keepends=True)
    out, skip = [], False
    for ln in lines:
        if ln.startswith("teams:"):
            out.append(block)
            skip = not ln.strip().endswith("[]")
            continue
        if skip and (ln.startswith("  - ") or ln.startswith("- ")):
            continue
        skip = False
        out.append(ln)
    if not any(ln.startswith("teams:") for ln in lines):
        out.append(block)
    p.write_text("".join(out))
    return p


def draw_from_results(stage: StageConfig, results: list[MatchResult]) -> list[list[str]] | None:
    """The published Round 1 draw, if the (scraped) round-1 matches pair every stage team exactly once."""
    pairs = [[r.a, r.b] for r in results if r.round == 1]
    flat = [t for p in pairs for t in p]
    if stage.type != "swiss" or not stage.teams or sorted(flat) != sorted(stage.teams):
        return None
    return pairs


def write_stage_round1(event: Event, stage: StageConfig, pairs: list[list[str]]) -> Path:
    """Pin the official Round 1 draw into the stage YAML (replacing any previous ``round1`` block)."""
    p = event.root / f"{stage.id}.yaml"
    out, skip = [], False
    for ln in p.read_text().splitlines(keepends=True):
        if ln.startswith("round1:"):
            skip = not ln.split("#")[0].strip().endswith("[]")
            continue
        if skip and ln.startswith(("  - ", "- ")):
            continue
        skip = False
        out.append(ln)
    text = "".join(out)
    if text and not text.endswith("\n"):
        text += "\n"
    text += "round1:  # official draw (scraped)\n" + "".join(f"  - [{a}, {b}]\n" for a, b in pairs)
    p.write_text(text)
    return p


def merge_results(existing: list[MatchResult], new: list[MatchResult]) -> tuple[list[MatchResult], list[str]]:
    """Merge scraped results into recorded ones without ever deleting or overriding a known winner.

    Matches are keyed by the pair of teams. New pairs are appended, missing winners are filled in.
    Returns ``(merged, changes)`` where ``changes`` are human-readable descriptions.
    """
    merged = [MatchResult(r.round, r.a, r.b, r.winner) for r in existing]
    index = {frozenset((r.a, r.b)): r for r in merged}
    changes: list[str] = []
    for n in new:
        key = frozenset((n.a, n.b))
        cur = index.get(key)
        if cur is None:
            m = MatchResult(n.round, n.a, n.b, n.winner)
            merged.append(m)
            index[key] = m
            changes.append(f"R{n.round} {n.a} vs {n.b}: " + (f"{n.winner} won" if n.winner else "scheduled"))
        elif cur.winner is None and n.winner is not None:
            cur.winner = n.winner
            changes.append(f"R{cur.round} {cur.a} vs {cur.b}: {n.winner} won")
    return merged, changes
