from __future__ import annotations

import json
import re

import numpy as np
import pytest
from typer.testing import CliRunner

from cs2pickem.cli import app
from cs2pickem.live import add_result, swiss_format
from cs2pickem.pickem import optimise_swiss
from cs2pickem.pipeline import Context, run_swiss
from cs2pickem.viz import build_payload, build_playoff_payload, build_swiss_payload, render_html
from cs2pickem.viz.export import _cum_at, checkpoints, team_colour, team_tag

N = 3000
RUNS = 60


@pytest.fixture(scope="module")
def ctx(event):
    return Context.load(event)


@pytest.fixture
def iso_event(event, tmp_path, monkeypatch):
    stages = {
        k: v.model_copy(update={"results_file": str(tmp_path / f"{k}.yaml")}) for k, v in event.stages.items()
    }
    monkeypatch.setattr(event, "stages", stages)
    return event


@pytest.fixture(scope="module")
def swiss(ctx):
    st = ctx.event.stage("stage1")
    return st, build_swiss_payload(ctx, st, N, n_runs=RUNS, use_results=False)


# --------------------------------------------------------------------------------------------- helpers
def test_checkpoints():
    assert checkpoints(0) == []
    assert checkpoints(1) == [1]
    pts = checkpoints(100_000)
    assert pts[0] == 1 and pts[-1] == 100_000
    assert all(b > a for a, b in zip(pts, pts[1:], strict=False))
    assert 100 < len(pts) <= 241


def test_cum_at_matches_cumsum():
    x = np.random.default_rng(1).integers(0, 3, size=(97, 4))
    pts = checkpoints(97, 20)
    np.testing.assert_array_equal(_cum_at(x, pts), np.cumsum(x, axis=0)[np.array(pts) - 1])


def test_team_tag_and_colour():
    assert team_tag("Anything", tag="XYZ") == "XYZ"
    assert team_tag("Ninjas in Pyjamas", ["NIP", "Ninjas"]) == "NIP"
    assert team_tag("Chinggis Warriors") == "CW"
    assert team_colour("a", 0) != team_colour("b", 1)
    assert re.fullmatch(r"#[0-9a-f]{6}", team_colour("Some Team", 99))


# --------------------------------------------------------------------------------------------- swiss
def test_swiss_schema(swiss):
    st, p = swiss
    for k in (
        "schema", "kind", "event", "stage", "n_sims", "format", "teams", "pool", "prob", "maps",
        "known", "cats", "cat_style", "bars", "runs", "ck", "final", "pickem",
    ):  # fmt: skip
        assert k in p, k
    T = len(st.teams)
    assert p["kind"] == "swiss" and p["n_sims"] == N
    assert p["cats"] == ["3-0", "3-1", "3-2", "2-3", "1-3", "0-3"]
    assert p["cat_style"] == ["adv"] * 3 + ["elim"] * 3
    assert len(p["teams"]) == T and all({"name", "tag", "colour", "seed"} <= set(t) for t in p["teams"])
    assert len(p["pool"]) == 7
    for m in p["prob"].values():
        a = np.array(m)
        assert a.shape == (T, T)
        np.testing.assert_allclose(a + a.T - np.diag(np.diag(a + a.T)), 1 - np.eye(T), atol=2e-4)
    assert len(p["runs"]) == RUNS
    K = len(p["ck"]["n"])
    assert np.array(p["ck"]["cat"]).shape == (K, T, len(p["cats"]))
    assert np.array(p["ck"]["hist"]).shape == (K, p["pickem"]["n_picks"] + 1)
    groups = [x["cat"] for x in p["pickem"]["picks"]]
    assert groups.count("perfect") == 2 and groups.count("winless") == 2 and groups.count("advance") == 6
    assert len({x["team"] for x in p["pickem"]["picks"]}) == 10


def test_swiss_tallies_match_simulator(ctx, swiss):
    st, p = swiss
    res = run_swiss(ctx, st, N, use_results=False)
    ck = p["ck"]
    assert ck["n"][-1] == N
    cat = np.array(ck["cat"])
    assert (np.diff(cat, axis=0) >= 0).all()  # cumulative
    assert (cat.sum(axis=2) == np.array(ck["n"])[:, None]).all()  # one category per team per sim
    dist = res.record_distribution()
    for k, c in enumerate(p["cats"]):
        np.testing.assert_allclose(cat[-1, :, k] / N, dist[c], atol=1e-12)
        np.testing.assert_allclose(p["final"][c], dist[c], atol=1e-5)
    hist = np.array(ck["hist"])
    assert (hist.sum(axis=1) == np.array(ck["n"])).all()
    r = ctx.event.config.pickem
    pk = optimise_swiss(res, r.n_30, r.n_03, r.n_advance, r.threshold)
    np.testing.assert_allclose(hist[-1] / N, pk.distribution, atol=1e-12)
    assert hist[-1][r.threshold :].sum() / N == pytest.approx(pk.p_threshold, abs=1e-5)
    assert ck["upsets"][-1] <= ck["matches"][-1]
    assert ck["matches"][-1] > 0


def test_swiss_runs_are_valid(ctx, swiss):
    st, p = swiss
    fmt = swiss_format(st)
    W, L = fmt.wins_to_advance, fmt.losses_to_eliminate
    res = run_swiss(ctx, st, N, use_results=False)
    T = len(st.teams)
    for s, run in enumerate(p["runs"]):
        w, lo = np.zeros(T, int), np.zeros(T, int)
        seen = set()
        for rnd in sorted({m[0] for m in run["m"]}):
            ms = [m for m in run["m"] if m[0] == rnd]
            assert [m[1] for m in ms] == list(range(len(ms)))
            playing = [t for m in ms for t in m[2:4]]
            assert len(playing) == len(set(playing))
            for _, _, a, b, bo, win, _ in ms:
                assert w[a] < W and lo[a] < L and w[b] < W and lo[b] < L  # 3W / 3L terminate
                assert (w[a], lo[a]) == (w[b], lo[b])  # same record bucket
                decider = w[a] == W - 1 or lo[a] == L - 1
                assert bo == (fmt.decider_bo if decider else fmt.default_bo)
                assert frozenset((a, b)) not in seen  # no rematches
                seen.add(frozenset((a, b)))
                assert win in (a, b)
                loser = b if win == a else a
                w[win] += 1
                lo[loser] += 1
        np.testing.assert_array_equal(w, res.wins[s])
        np.testing.assert_array_equal(lo, res.losses[s])
        cats = [f"{x}-{y}" for x, y in zip(w, lo, strict=True)]
        assert [p["cats"][k] for k in run["fin"]] == cats
        for _, _, a, b, bo, _, _ in run["m"]:
            assert f"{min(a, b)}-{max(a, b)}-{bo}" in p["maps"]


def test_record_runs_does_not_change_results(ctx):
    st = ctx.event.stage("stage1")
    a = run_swiss(ctx, st, 1500, use_results=False)
    b = run_swiss(ctx, st, 1500, use_results=False, record_runs=50)
    np.testing.assert_array_equal(a.wins, b.wins)
    np.testing.assert_array_equal(a.losses, b.losses)
    assert len(b.runs) == 50 and not a.runs


def test_live_results_are_forced(iso_event):
    ctx = Context.load(iso_event)
    st = iso_event.stage("stage1")
    a, b = st.round1[0]
    add_result(iso_event, st, a, b, b, 1)
    p = build_swiss_payload(ctx, st, 1000, n_runs=30)
    ia, ib = st.teams.index(a), st.teams.index(b)
    assert len(p["known"]) == 1
    for run in p["runs"]:
        m = next(m for m in run["m"] if m[0] == 1 and {m[2], m[3]} == {ia, ib})
        assert m[5] == ib and m[6] == 1
        assert sum(x[6] for x in run["m"]) == 1


# --------------------------------------------------------------------------------------------- playoffs
def test_playoff_payload(ctx):
    st = ctx.event.stage("playoffs")
    p = build_playoff_payload(ctx, st, 2000, n_runs=25)
    R = p["format"]["rounds"]
    assert p["kind"] == "playoffs" and R == 3
    assert p["cats"][-1] == "Champion" and len(p["cats"]) == R + 1
    assert p["format"]["bo"][-1] == st.final_bo
    cat = np.array(p["ck"]["cat"])
    assert (cat[-1].sum(axis=1) == 2000).all()
    assert cat[-1, :, R].sum() == 2000  # exactly one champion per sim
    assert np.isclose(sum(p["final"]["Champion"]), 1)
    assert len(p["pickem"]["picks"]) == 7
    for run in p["runs"]:
        assert len(run["m"]) == 7
        champ = run["m"][-1][5]
        assert run["fin"][champ] == R
        assert sorted(run["fin"]) == [0, 0, 0, 0, 1, 1, 2, 3]


# --------------------------------------------------------------------------------------------- html
def test_render_html_self_contained(swiss):
    _, p = swiss
    h = render_html(p)
    assert "__DATA__" not in h and "__TITLE__" not in h
    assert not re.search(r"""(src|href)\s*=\s*["']?https?:""", h)  # works offline
    m = re.search(r'<script type="application/json" id="data">(.*?)</script>', h, re.S)
    assert m and json.loads(m.group(1)) == json.loads(json.dumps(p))


def test_render_html_escapes_script_close(swiss):
    _, p = swiss
    q = {**p, "stage_name": "</script><b>x"}
    h = render_html(q)
    assert h.count("</script>") == render_html(p).count("</script>")
    m = re.search(r'<script type="application/json" id="data">(.*?)</script>', h, re.S)
    assert json.loads(m.group(1))["stage_name"] == "</script><b>x"


def test_cli_animate(tmp_path):
    out = tmp_path / "s.html"
    r = CliRunner().invoke(
        app, ["animate", "--stage", "stage1", "-n", "500", "--runs", "10", "--out", str(out)]
    )
    assert r.exit_code == 0, r.output
    assert out.exists() and "wrote" in r.output
    assert build_payload  # re-export sanity
