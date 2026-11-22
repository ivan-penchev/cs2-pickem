from __future__ import annotations

import pytest

from cs2pickem.live import (
    add_result,
    final_standings,
    known_by_round,
    load_results,
    qualifiers,
    resolve_teams,
    swiss_format,
)
from cs2pickem.swiss.simulator import play_out


def test_event_config(event):
    s1 = event.stage("stage1")
    assert len(s1.teams) == 16 and len(s1.round1) == 8
    assert {t for p in s1.round1 for t in p} == set(s1.teams)
    for t in s1.teams:
        assert event.config.canonical(t) == t
    po = event.stage("playoffs")
    assert po.type == "playoffs" and po.final_bo == 5


@pytest.fixture
def iso_event(event, tmp_path, monkeypatch):
    """Event whose results files live in tmp_path (isolated from any local data/results)."""
    stages = {
        k: v.model_copy(update={"results_file": str(tmp_path / f"{k}.yaml")}) for k, v in event.stages.items()
    }
    monkeypatch.setattr(event, "stages", stages)
    return event


def test_resolve_projected_puts_invites_first(iso_event):
    s1, s2 = iso_event.stage("stage1"), iso_event.stage("stage2")
    with pytest.raises(ValueError):
        resolve_teams(iso_event, s2)  # stage 1 not finished, nothing projected
    proj = list(s1.teams[:8])
    teams, how = resolve_teams(iso_event, s2, projected=proj)
    assert how == "projected"
    assert teams[:8] == list(s2.source.invites)
    assert teams[8:] == proj


def test_results_roundtrip_and_qualifiers(iso_event):
    event = iso_event
    s1 = event.stage("stage1")
    teams = list(s1.teams)
    # better seed wins every match until the stage ends
    for _ in range(40):
        known = known_by_round(load_results(event, s1))
        _, _, _, nxt, rnd = play_out(teams, swiss_format(s1), s1.round1, known)
        if nxt is None:
            break
        done = {frozenset((teams.index(a), teams.index(b))) for a, b, _ in known.get(rnd, [])}
        a, b = next(p for p in nxt if frozenset(p) not in done)
        add_result(event, s1, teams[a], teams[b], teams[min(a, b)], rnd)
    order = final_standings(event, s1)
    assert order is not None
    assert len(qualifiers(event, s1)) == 8
    assert set(qualifiers(event, s1)) == set(teams[:8])


def test_add_result_rejects_bad_winner(iso_event):
    event = iso_event
    s1 = event.stage("stage1")
    a, b = s1.round1[0]
    with pytest.raises(ValueError):
        add_result(event, s1, a, b, "Nobody", 1)


def test_stage2_from_finished_stage1(iso_event):
    s1, s2 = iso_event.stage("stage1"), iso_event.stage("stage2")
    test_results_roundtrip_and_qualifiers(iso_event)
    teams, how = resolve_teams(iso_event, s2)
    assert how == "results"
    assert teams[:8] == list(s2.source.invites) and set(teams[8:]) == set(s1.teams[:8])
