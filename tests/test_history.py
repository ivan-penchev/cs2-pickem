from __future__ import annotations

import json
from datetime import UTC, date, datetime

from cs2pickem import history as H

TEAMS = ["A", "B", "C", "D"]


def entry(known, picks=(("3-0", "A"), ("0-3", "D")), stage="s1", teams=TEAMS):
    return {
        "stage": stage,
        "stage_name": "Stage",
        "type": "swiss",
        "teams": [{"name": t, "tag": t, "colour": "#fff", "seed": i + 1} for i, t in enumerate(teams)],
        "metrics": [{"label": "P(advance)", "colour": "#fff", "values": [0.5] * len(teams)}],
        "picks": [{"group": g, "team": t, "cats": [0]} for g, t in picks],
        "cats": ["3-0"],
        "threshold": 5,
        "p_threshold": 0.5,
        "expected": 4.9,
        "n_picks": 10,
        "known": [dict(m) for m in known],
        "source": "test",
    }


def m(r, a, b, w=None):
    return {"round": r, "a": a, "b": b, "winner": w}


T0 = datetime(2026, 11, 24, 9, tzinfo=UTC)


def test_baseline_then_append_only(tmp_path):
    pending = [m(1, "A", "B"), m(1, "C", "D")]
    p0 = H.record(entry(pending), tmp_path, T0)
    assert p0 and json.loads(p0.read_text())["baseline"] is True
    assert json.loads(p0.read_text())["trigger"] == "pre-stage baseline"
    # nothing finished changed (even if the scheduled list grows) → no entry
    assert H.record(entry(pending), tmp_path, T0.replace(hour=10)) is None
    base_txt = p0.read_text()
    p1 = H.record(entry([m(1, "A", "B", "A"), m(1, "C", "D")]), tmp_path, T0.replace(hour=11))
    e1 = json.loads(p1.read_text())
    assert e1["baseline"] is False and e1["trigger"] == "R1: A beat B"
    # same timestamp never overwrites
    p2 = H.record(entry([m(1, "A", "B", "A"), m(1, "C", "D", "D")]), tmp_path, T0.replace(hour=11))
    assert p2 != p1 and p1.exists() and p0.read_text() == base_txt
    hist = H.load("s1", tmp_path)
    assert [h["baseline"] for h in hist] == [True, False, False]
    assert sum(h["baseline"] for h in hist) == 1
    assert H.stages(tmp_path) == ["s1"]


def test_baseline_with_known_results(tmp_path):
    p = H.record(entry([m(1, "A", "B", "B")]), tmp_path, T0)
    assert json.loads(p.read_text())["trigger"].startswith("baseline (1 results")


def test_grade_swiss_picks():
    p30, p03, adv = (
        {"group": "3-0", "team": "A"},
        {"group": "0-3", "team": "A"},
        {"group": "Advance", "team": "A"},
    )
    assert H.grade_pick(p30, "swiss", []) is None
    k = [m(1, "A", "B", "A"), m(2, "A", "C", "A")]
    assert H.grade_pick(p30, "swiss", k) is None
    assert H.grade_pick(p03, "swiss", k) is False
    k3 = [*k, m(3, "A", "D", "A")]
    assert H.grade_pick(p30, "swiss", k3) is True and H.grade_pick(adv, "swiss", k3) is True
    kl = [m(1, "A", "B", "B"), m(2, "A", "C", "C"), m(3, "A", "D", "D")]
    assert H.grade_pick(p30, "swiss", kl) is False
    assert H.grade_pick(p03, "swiss", kl) is True
    assert H.grade_pick(adv, "swiss", kl) is False


def test_grade_playoff_picks():
    k = [m(1, "A", "B", "A"), m(1, "C", "D", "C"), m(2, "A", "C", "C")]
    assert H.grade_pick({"group": "QF 1", "team": "A", "cats": [1, 2, 3]}, "playoffs", k) is True
    assert H.grade_pick({"group": "SF 1", "team": "A", "cats": [2, 3]}, "playoffs", k) is False
    assert H.grade_pick({"group": "SF 2", "team": "C", "cats": [2, 3]}, "playoffs", k) is True
    assert H.grade_pick({"group": "Final", "team": "C", "cats": [3]}, "playoffs", k) is None


def test_pick_changes():
    a = {**entry([]), "ts": "t0", "trigger": "pre-stage baseline"}
    b = {**entry([], picks=(("3-0", "B"), ("0-3", "D"))), "ts": "t1", "trigger": "R1: B beat A"}
    ch = H.pick_changes([a, a, b])
    assert len(ch) == 1 and ch[0]["added"] == [("3-0", "B")] and ch[0]["removed"] == [("3-0", "A")]
    assert ch[0]["trigger"] == "R1: B beat A"


def test_records():
    assert H.records([m(1, "A", "B", "A"), m(1, "C", "D")]) == {"A": (1, 0), "B": (0, 1)}


def test_dashboard_render(tmp_path):
    from cs2pickem.site import dashboard as D

    b = {
        **entry([m(1, "A", "B")]),
        "ts": "2026-11-24T09:00:00Z",
        "trigger": "pre-stage baseline",
        "baseline": True,
    }
    c = {
        **entry([m(1, "A", "B", "B")], picks=(("3-0", "B"), ("0-3", "D"))),
        "ts": "x",
        "trigger": "R1: B beat A",
    }
    c["metrics"][0]["values"] = [0.3, 0.7, 0.5, 0.5]
    st = {
        "id": "s1",
        "name": "Stage 1",
        "how": "config",
        "status": "Stage 1 · live",
        "page": "s1.html",
        "history_page": "s1-history.html",
        "history": [b, c],
        "current": c,
    }
    ev = {"name": "Test Major", "dates": "", "phase": "LIVE", "live": True}
    html = D.render_index(ev, [st], "s1", {"built": "now", "source": "liquipedia"}, False)
    assert "Original (baseline) picks" in html and "liquipedia" in html
    assert "▲20" in html and "▼20" in html  # deltas vs baseline
    assert "R1: B beat A" in html  # timeline annotation / picks changed
    hist = D.render_stage_history(ev, st, {"built": "now"})
    assert "Prediction log" in hist and "baseline" in hist


def test_site_build_records_baseline_once(event, tmp_path, monkeypatch):
    from cs2pickem.site import build as B

    monkeypatch.setattr(H, "data_dir", lambda: tmp_path / "data")
    monkeypatch.setattr(B, "data_dir", lambda: tmp_path / "data")
    stages = {
        k: v.model_copy(update={"results_file": str(tmp_path / f"{k}.yaml")}) for k, v in event.stages.items()
    }
    monkeypatch.setattr(event, "stages", {"stage1": stages["stage1"]})
    out = tmp_path / "site"
    B.build(event, n_sims=1000, n_runs=5, out_dir=out, today=date(2026, 11, 22), log=lambda _: None)
    B.build(event, n_sims=1000, n_runs=5, out_dir=out, today=date(2026, 11, 22), log=lambda _: None)
    hist = H.load("stage1")
    assert len(hist) == 1 and hist[0]["baseline"]
    assert len(hist[0]["picks"]) == 10 and len(hist[0]["teams"]) == 16
    assert (
        (out / "index.html").exists()
        and (out / "stage1.html").exists()
        and (out / "stage1-history.html").exists()
    )
