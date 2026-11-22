from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from cs2pickem.live import MatchResult, load_results, merge_results
from cs2pickem.livescrape import (
    ScrapeOutcome,
    assign_rounds,
    in_window,
    make_resolver,
    parse_dfrag,
    parse_hltv_results,
    update,
)

FX = Path(__file__).parent / "fixtures"


def test_parse_dfrag_pending_and_finished():
    pend = parse_dfrag((FX / "dfrag_pending.html").read_text(), "pgl-major-singapore-2026-stage-1")
    assert len(pend) == 8
    assert all(m["round"] == 1 and m["winner"] is None for m in pend)
    assert {"9z", "Lynn Vision"} == {pend[0]["a"], pend[0]["b"]}
    done = parse_dfrag((FX / "dfrag_done.html").read_text(), "iem-cologne-major-2026-stage-1")
    assert len(done) == 33  # 8 + 8 + 8 + 6 + 3 Swiss matches
    assert {m["round"] for m in done} == {1, 2, 3, 4, 5}
    assert all(m["winner"] in (m["a"], m["b"]) for m in done)


def test_parse_hltv_results():
    rows = parse_hltv_results((FX / "hltv_results.html").read_text())
    assert len(rows) == 4
    assert [r["ts"] for r in rows] == sorted(r["ts"] for r in rows)
    for r in rows:
        assert r["winner"] in (r["a"], r["b"]) and r["id"]


def test_resolver_aliases_and_fuzzy(event):
    res = make_resolver(event, event.stage("stage1"))
    assert res("Ninjas in Pyjamas") == "Ninjas in Pyjamas"
    assert res("NIP") == "Ninjas in Pyjamas"
    assert res("BC.Game Esports") == "GOSH"
    assert res("Team Liquid") == "Liquid"
    assert res("Lynn  Vision") == "Lynn Vision"
    assert res("The MongolZ") is None  # not a stage 1 team


def test_assign_rounds_by_match_count():
    rows = [("A", "B", "A"), ("C", "D", "D"), ("A", "D", "A"), ("B", "C", "C")]
    out = assign_rounds(rows, [])
    assert [m.round for m in out] == [1, 1, 2, 2]
    out2 = assign_rounds([("A", "C", "C")], out)
    assert out2[0].round == 3
    # already known pair keeps its round
    assert assign_rounds([("D", "A", "A")], out)[0].round == 2


def test_merge_is_idempotent_and_never_deletes():
    ex = [MatchResult(1, "A", "B", "A"), MatchResult(1, "C", "D", None)]
    new = [MatchResult(1, "D", "C", "D"), MatchResult(2, "A", "D", None), MatchResult(1, "B", "A", "B")]
    merged, ch = merge_results(ex, new)
    assert len(merged) == 3 and len(ch) == 2
    by = {frozenset((m.a, m.b)): m for m in merged}
    assert by[frozenset("AB")].winner == "A"  # never overridden
    assert by[frozenset("CD")].winner == "D"  # filled in
    merged2, ch2 = merge_results(merged, new)
    assert ch2 == [] and len(merged2) == 3
    merged3, _ = merge_results(merged, [])  # a failed scrape returns nothing → nothing removed
    assert len(merged3) == 3


def test_in_window():
    s, e = date(2026, 11, 24), date(2026, 12, 13)
    assert not in_window(s, e, date(2026, 11, 22))
    assert in_window(s, e, date(2026, 11, 23))  # 1 day slack
    assert in_window(s, e, date(2026, 12, 1))
    assert not in_window(None, None, date(2026, 12, 1))


@pytest.fixture
def iso_event(event, tmp_path, monkeypatch):
    stages = {
        k: v.model_copy(update={"results_file": str(tmp_path / f"{k}.yaml")}) for k, v in event.stages.items()
    }
    monkeypatch.setattr(event, "stages", stages)
    return event


def test_update_outside_window_is_noop(iso_event):
    rep = update(iso_event, today=date(2026, 1, 1), scraper=lambda *a, **k: pytest.fail("scraped"))
    assert not rep.ran and not rep.changed


def test_update_merges_and_is_idempotent(iso_event):
    s1 = iso_event.stage("stage1")
    a, b = s1.round1[0]

    def scraper(ev, st, existing, refresh=True):
        return ScrapeOutcome([MatchResult(1, a, b, a)], "fake", ["ok"])

    rep = update(iso_event, today=date(2026, 11, 24), scraper=scraper)
    assert rep.ran and rep.changed and rep.stages[0].source == "fake"
    assert load_results(iso_event, s1)[0].winner == a
    rep2 = update(iso_event, today=date(2026, 11, 24), scraper=scraper)
    assert not rep2.changed

    def failing(ev, st, existing, refresh=True):
        return ScrapeOutcome([], None, ["all sources failed"])

    rep3 = update(iso_event, today=date(2026, 11, 24), scraper=failing)
    assert not rep3.changed and len(load_results(iso_event, s1)) == 1
    assert "all sources failed" in rep3.markdown()
