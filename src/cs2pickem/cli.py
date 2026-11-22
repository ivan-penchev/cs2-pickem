"""``cs2pickem`` command line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from .config import Event, StageConfig, load_event
from .paths import configs_dir

app = typer.Typer(
    help="CS2 Major Pick'Em predictor (Swiss + playoff Monte Carlo, Pick'Em optimiser).", no_args_is_help=True
)
result_app = typer.Typer(help="Record completed matches (live mode).", no_args_is_help=True)
app.add_typer(result_app, name="result")
console = Console(width=max(Console().width, 110))

EventOpt = Annotated[
    str | None,
    typer.Option(
        "--event", "-e", help="Event config dir or name under configs/ (default: the only/first one)."
    ),
]
StageOpt = Annotated[str, typer.Option("--stage", "-s", help="Stage id from the event config, e.g. stage1.")]
SimsOpt = Annotated[int | None, typer.Option("--sims", "-n", help="Number of Monte Carlo simulations.")]
JsonOpt = Annotated[Path | None, typer.Option("--json", help="Also write machine-readable output here.")]


def _event(name: str | None) -> Event:
    if name:
        p = Path(name)
        if not p.exists():
            p = configs_dir() / name
        return load_event(p)
    cands = (
        sorted(d for d in configs_dir().iterdir() if (d / "event.yaml").exists())
        if configs_dir().exists()
        else []
    )
    if not cands:
        raise typer.BadParameter("no event config found – pass --event")
    return load_event(cands[-1])


def _ctx(event: Event):
    from .pipeline import Context

    ctx = Context.load(event)
    for n in ctx.inputs.notes:
        console.print(f"[yellow]note:[/] {n}")
    return ctx


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _stage(event: Event, stage: str) -> StageConfig:
    try:
        return event.stage(stage)
    except KeyError as e:
        raise typer.BadParameter(str(e)) from e


# --------------------------------------------------------------------------------------------- data
@app.command()
def fetch(
    event: EventOpt = None,
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Re-scrape sources and write new snapshots.")
    ] = False,
    source: Annotated[
        list[str] | None, typer.Option("--source", help="vrs and/or hltv (default both).")
    ] = None,
) -> None:
    """Show snapshot status, or re-scrape (``--refresh``) VRS (GitHub) and HLTV (Playwright)."""
    from .data.fetch import fetch_all
    from .snapshots import read_manifest

    ev = _event(event)
    if refresh:
        fetch_all(ev, source or ["vrs", "hltv"], refresh=True, log=lambda m: console.print(m))
    man = read_manifest()
    t = Table(title="Data snapshots (data/snapshots, offline by default)")
    for c in ("source", "date", "path", "tables"):
        t.add_column(c)
    for src, info in sorted(man.items()):
        if isinstance(info, dict):
            t.add_row(
                src, str(info.get("date", "")), str(info.get("path", "")), ", ".join(info.get("tables", []))
            )
    console.print(t)
    if not refresh:
        console.print("Using committed snapshots. Run [bold]cs2pickem fetch --refresh[/] to re-scrape.")


@app.command()
def rate(
    event: EventOpt = None,
    stage: Annotated[str | None, typer.Option("--stage", "-s", help="Only teams of this stage.")] = None,
) -> None:
    """Blended team ratings (VRS + HLTV + recent-form Elo + manual overrides)."""
    ev = _event(event)
    ctx = _ctx(ev)
    tab = ctx.ratings.table
    if stage:
        from .pipeline import stage_teams

        teams, _ = stage_teams(ctx, _stage(ev, stage))
        tab = tab.loc[[t for t in tab.index if t in teams]]
    t = Table(title=f"{ev.config.name} – team ratings")
    for c in ("#", "team", "rating", "VRS pts", "HLTV pts", "form Elo", "eff. matches", "override"):
        t.add_column(c, justify="right" if c != "team" else "left")
    for i, (team, r) in enumerate(tab.iterrows(), 1):
        f = lambda v, d=0: "–" if v != v else f"{v:.{d}f}"  # noqa: E731
        t.add_row(
            str(i),
            team,
            f(r["rating"]),
            f(r["vrs"]),
            f(r["hltv"]),
            f(r["elo"]),
            f(r["n_matches"], 1),
            f(r["override"]),
        )
    console.print(t)


@app.command()
def matchup(
    a: str,
    b: str,
    event: EventOpt = None,
    bo: Annotated[int, typer.Option(help="Best-of.")] = 3,
) -> None:
    """Head-to-head: series win probability, per-map probabilities and predicted veto (A = higher seed)."""
    ev = _event(event)
    ctx = _ctx(ev)
    a, b = ev.config.canonical(a), ev.config.canonical(b)
    mm = ctx.model([a, b])
    mp = mm.map_probs(0, 1)
    console.print(f"[bold]{a}[/] vs [bold]{b}[/]  Bo{bo}: P({a}) = {_pct(mm.series(0, 1, bo))}")
    t = Table(title="Map win probability for " + a)
    for m in mm.pool:
        t.add_column(m, justify="right")
    t.add_row(*[_pct(p) for p in mp])
    console.print(t)
    console.print("Predicted veto: " + " → ".join(mm.veto_line(0, 1, bo)))


# --------------------------------------------------------------------------------------------- swiss
def _swiss_table(res, title: str) -> Table:
    s = res.summary()
    rec = res.record_distribution()
    t = Table(title=title)
    t.add_column("seed", justify="right")
    t.add_column("team", no_wrap=True)
    for c in ("3-0", "advance", "0-3"):
        t.add_column(f"P({c})" if c != "advance" else "P(qualify)", justify="right")
    rec = {k: v for k, v in rec.items() if k not in ("3-0", "0-3")}
    for k in rec:
        t.add_column(k, justify="right", style="dim")
    order = np.argsort(-(s["qualified"] + s["perfect"] - s["winless"]), kind="stable")
    for i in order:
        t.add_row(
            str(i + 1),
            res.teams[i],
            _pct(s["perfect"][i]),
            _pct(s["qualified"][i]),
            _pct(s["winless"][i]),
            *[_pct(v[i]) for v in rec.values()],
        )
    return t


@app.command()
def simulate(
    stage: StageOpt = "stage1",
    event: EventOpt = None,
    sims: SimsOpt = None,
    no_results: Annotated[
        bool, typer.Option("--no-results", help="Ignore recorded results (pre-stage view).")
    ] = False,
    flat: Annotated[bool, typer.Option("--flat", help="Disable map/veto model (plain Elo).")] = False,
    json_out: JsonOpt = None,
) -> None:
    """Monte Carlo simulation of a stage (Swiss or playoffs), conditional on recorded results."""
    ev = _event(event)
    st = _stage(ev, stage)
    ctx = _ctx(ev)
    if st.type == "playoffs":
        _playoffs(ctx, st, sims, json_out)
        return
    from .pipeline import run_swiss, stage_teams

    teams, how = stage_teams(ctx, st)
    if how == "projected":
        console.print(f"[yellow]{st.id} teams are projected from simulations of the previous stage[/]")
    res = run_swiss(
        ctx, st.model_copy(update={"teams": teams}), sims, use_results=not no_results, use_veto=not flat
    )
    console.print(_swiss_table(res, f"{ev.config.name} – {st.name} ({res.n_sims:,} sims)"))
    if json_out:
        s = res.summary()
        out = {
            t: {k: float(v[i]) for k, v in {**s, **res.record_distribution()}.items()}
            for i, t in enumerate(res.teams)
        }
        json_out.write_text(json.dumps(out, indent=2))


@app.command()
def pickem(
    stage: StageOpt = "stage1",
    event: EventOpt = None,
    sims: SimsOpt = None,
    lock: Annotated[
        list[str] | None,
        typer.Option(
            help="Force a pick, e.g. --lock '3-0=Ninjas in Pyjamas' --lock 'adv=FaZe' --lock '0-3=GOSH'."
        ),
    ] = None,
    no_results: Annotated[bool, typer.Option("--no-results")] = False,
    json_out: JsonOpt = None,
) -> None:
    """Optimal Pick'Em: Swiss (maximise P(≥5 correct)) or playoff bracket (maximise expected correct)."""
    ev = _event(event)
    st = _stage(ev, stage)
    ctx = _ctx(ev)
    if st.type == "playoffs":
        from .pickem import optimise_bracket
        from .pipeline import run_playoffs

        res, _ = run_playoffs(ctx, st, sims)
        bp = optimise_bracket(res)
        t = Table(
            title=f"{st.name} bracket Pick'Em – E\\[correct] = {bp.expected:.2f} / {sum(map(len, bp.picks))}"
        )
        t.add_column("round")
        t.add_column("pick")
        t.add_column("P(correct)", justify="right")
        for lab, row, pr in zip(bp.labels, bp.named(), bp.probs, strict=True):
            for team, p in zip(row, pr, strict=True):
                t.add_row(lab, team, _pct(p))
        console.print(t)
        if json_out:
            json_out.write_text(
                json.dumps(
                    {"picks": dict(zip(bp.labels, bp.named(), strict=True)), "expected": bp.expected},
                    indent=2,
                )
            )
        return

    from .pickem import CAT_LABEL, optimise_swiss
    from .pipeline import run_swiss, stage_teams

    teams, how = stage_teams(ctx, st)
    if how == "projected":
        console.print(f"[yellow]{st.id} teams are projected from simulations of the previous stage[/]")
    res = run_swiss(ctx, st.model_copy(update={"teams": teams}), sims, use_results=not no_results)
    locked: dict[str, list[str]] = {}
    alias = {
        "3-0": "perfect",
        "30": "perfect",
        "0-3": "winless",
        "03": "winless",
        "adv": "advance",
        "advance": "advance",
    }
    for item in lock or []:
        k, _, v = item.partition("=")
        if k.strip() not in alias:
            raise typer.BadParameter(f"--lock {item!r}: use 3-0=TEAM, 0-3=TEAM or adv=TEAM")
        locked.setdefault(alias[k.strip()], []).append(ev.config.canonical(v.strip()))
    r = ev.config.pickem
    pk = optimise_swiss(res, r.n_30, r.n_03, r.n_advance, r.threshold, locked=locked)
    s = res.summary()
    t = Table(title=f"{ev.config.name} – {st.name} Pick'Em ({res.n_sims:,} sims)")
    for c in ("pick", "team", "P(correct)", "P(3-0)", "P(qualify)", "P(0-3)"):
        t.add_column(c, justify="left" if c in ("pick", "team") else "right")
    idx = {x: i for i, x in enumerate(res.teams)}
    for cat in ("perfect", "advance", "winless"):
        for team, p in zip(pk.named()[cat], pk.marginals[cat], strict=True):
            i = idx[team]
            t.add_row(
                CAT_LABEL[cat],
                team,
                _pct(p),
                _pct(s["perfect"][i]),
                _pct(s["qualified"][i]),
                _pct(s["winless"][i]),
            )
    console.print(t)
    console.print(
        f"P(≥{pk.threshold} correct) = [bold green]{_pct(pk.p_threshold)}[/]   E\\[correct] = [bold]{pk.expected:.2f}[/]"
    )
    console.print(
        "P(k correct): " + "  ".join(f"{k}:{_pct(p).strip()}" for k, p in enumerate(pk.distribution))
    )
    if json_out:
        json_out.write_text(
            json.dumps(
                {
                    "picks": {CAT_LABEL[c]: v for c, v in pk.named().items()},
                    "p_threshold": pk.p_threshold,
                    "expected": pk.expected,
                    "distribution": pk.distribution.tolist(),
                },
                indent=2,
            )
        )


# --------------------------------------------------------------------------------------------- playoffs
def _playoffs(ctx, st: StageConfig, sims: int | None, json_out: Path | None) -> None:
    from .pipeline import run_playoffs

    res, how = run_playoffs(ctx, st, sims)
    if how == "projected":
        console.print("[yellow]playoff teams are projected from simulations of the previous stage[/]")
    reach = res.reach()
    t = Table(title=f"{st.name} ({res.n_sims:,} sims)")
    t.add_column("seed", justify="right")
    t.add_column("team", no_wrap=True)
    labels = {"Quarter-finals": "P(SF)", "Semi-finals": "P(Final)", "Final": "P(Champion)"}
    for lab in reach:
        t.add_column(labels.get(lab, f"P(win {lab})"), justify="right")
    for i in np.argsort(-reach[res.labels[-1]], kind="stable"):
        t.add_row(str(i + 1), res.teams[i], *[_pct(v[i]) for v in reach.values()])
    console.print(t)
    if json_out:
        json_out.write_text(
            json.dumps(
                {t: {k: float(v[i]) for k, v in reach.items()} for i, t in enumerate(res.teams)}, indent=2
            )
        )


@app.command()
def playoffs(
    stage: StageOpt = "playoffs",
    event: EventOpt = None,
    sims: SimsOpt = None,
    json_out: JsonOpt = None,
) -> None:
    """Simulate the playoff bracket (teams from config, Stage 3 results, or projected)."""
    ev = _event(event)
    _playoffs(_ctx(ev), _stage(ev, stage), sims, json_out)


# --------------------------------------------------------------------------------------------- live
@result_app.command("add")
def result_add(
    a: Annotated[str, typer.Argument(help="Team A")],
    b: Annotated[str, typer.Argument(help="Team B")],
    winner: Annotated[str, typer.Argument(help="Winner (team name or alias)")],
    stage: StageOpt = "stage1",
    event: EventOpt = None,
    round_: Annotated[int | None, typer.Option("--round", "-r", help="Round (inferred if omitted).")] = None,
) -> None:
    """Record a completed match, e.g. ``cs2pickem result add NiP M80 NiP``."""
    from .live import add_result, results_path

    ev = _event(event)
    st = _stage(ev, stage)
    if not st.teams:
        raise typer.BadParameter(
            f"{st.id} has no teams yet – run `cs2pickem next-stage --stage {st.id}` first"
        )
    try:
        r = add_result(ev, st, a, b, winner, round_)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    console.print(f"Round {r.round}: {r.a} vs {r.b} → [bold]{r.winner}[/]  (saved to {results_path(st)})")


@result_app.command("list")
def result_list(stage: StageOpt = "stage1", event: EventOpt = None) -> None:
    """Show recorded results and current Swiss standings."""
    from .live import known_by_round, load_results, swiss_format
    from .swiss.pairing import buchholz
    from .swiss.simulator import play_out

    ev = _event(event)
    st = _stage(ev, stage)
    res = load_results(ev, st)
    t = Table(title=f"{st.name} results")
    for c in ("round", "team A", "team B", "winner"):
        t.add_column(c)
    for r in res:
        t.add_row(str(r.round), r.a, r.b, r.winner or "–")
    console.print(t)
    if st.type == "swiss" and st.teams:
        wins, losses, opps, nxt, rnd = play_out(
            st.teams,
            swiss_format(st),
            [tuple(p) for p in st.round1] if st.round1 else None,
            known_by_round(res),  # type: ignore[misc]
        )
        s = Table(title="Standings")
        for c in ("team", "W-L", "Buchholz"):
            s.add_column(c)
        for i in sorted(
            range(len(st.teams)), key=lambda i: (-wins[i], losses[i], -buchholz(i, wins, losses, opps), i)
        ):
            s.add_row(st.teams[i], f"{wins[i]}-{losses[i]}", str(buchholz(i, wins, losses, opps)))
        console.print(s)
        if nxt:
            console.print(
                f"Round {rnd} pairings: " + ", ".join(f"{st.teams[x]} vs {st.teams[y]}" for x, y in nxt)
            )


@result_app.command("remove")
def result_remove(a: str, b: str, stage: StageOpt = "stage1", event: EventOpt = None) -> None:
    """Delete a recorded match."""
    from .live import load_results, save_results

    ev = _event(event)
    st = _stage(ev, stage)
    a, b = ev.config.canonical(a), ev.config.canonical(b)
    res = load_results(ev, st)
    keep = [r for r in res if {r.a, r.b} != {a, b}]
    if len(keep) == len(res):
        raise typer.BadParameter(f"no result {a} vs {b}")
    save_results(st, keep)
    console.print(f"removed {a} vs {b}")


@app.command("next-stage")
def next_stage(
    stage: StageOpt,
    event: EventOpt = None,
    write: Annotated[bool, typer.Option("--write", help="Write the seed order into the stage YAML.")] = False,
) -> None:
    """Build a stage's seed order: direct invites (1..8) + previous-stage qualifiers (by final standing)."""
    from .live import resolve_teams, write_stage_teams
    from .pipeline import stage_teams

    ev = _event(event)
    st = _stage(ev, stage).model_copy(update={"teams": []})
    try:
        teams, how = resolve_teams(ev, st)
    except ValueError as e:
        console.print(f"[yellow]{e}[/]\nShowing the most likely seeding from simulations instead.")
        teams, how = stage_teams(_ctx(ev), st)
    t = Table(title=f"{st.name} seeds ({how})")
    t.add_column("seed", justify="right")
    t.add_column("team", no_wrap=True)
    for i, x in enumerate(teams, 1):
        t.add_row(str(i), x)
    console.print(t)
    if write and how == "projected":
        raise typer.BadParameter("refusing to --write a projected seeding; finish the previous stage first")
    if write:
        console.print(f"wrote {write_stage_teams(ev, st, teams)}")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
