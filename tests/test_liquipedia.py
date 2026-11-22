from __future__ import annotations

from pathlib import Path

from cs2pickem.data import liquipedia as lp

FIX = Path(__file__).parent / "fixtures"


def test_parse_swiss_stage():
    ms = lp.parse_swiss((FIX / "lp_swiss.wiki").read_text())
    assert len(ms) == 33
    per_round = [sum(m.round == r for m in ms) for r in range(1, 6)]
    assert per_round == [8, 8, 8, 6, 3]
    assert all(m.finished for m in ms)
    assert sum(m.bo == 3 for m in ms) == 13
    for m in ms:
        need = m.bo // 2 + 1
        wins = sum(1 for x in m.maps if (x.winner == 0) == (m.winner == m.a))
        assert wins == need
    r1 = [(m.a, m.b) for m in ms if m.round == 1]
    assert r1[0] == ("b8", "m80")
    teams = {t for m in ms for t in (m.a, m.b)}
    assert len(teams) == 16


def test_parse_bracket_final_bo5():
    ms = lp.parse_bracket((FIX / "lp_bracket.wiki").read_text())
    assert [m.round for m in ms] == [1, 1, 1, 1, 2, 2, 3]
    assert [m.bo for m in ms] == [3] * 6 + [5]
    final = ms[-1]
    assert {final.a, final.b} == {"vit", "faze"} and final.winner == "vit"


def test_map_scores_include_overtime():
    m = lp._map("{{Map|map=Nuke|finished=true|t1t=6|t1ct=6|t2t=6|t2ct=6|o1t1t=3|o1t1ct=1|o1t2t=2|o1t2ct=1}}")
    assert m is not None and (m.score_a, m.score_b) == (16, 15) and m.winner == 0
    assert lp._map("{{Map|map=Nuke|finished=skip}}") is None


def test_rename():
    ms = [lp.LpMatch(1, "fq", "imp", "imp")]
    lp.rename(ms, {"fq": "FlyQuest", "imp": "Imperial Esports"})
    assert (ms[0].a, ms[0].b, ms[0].winner) == ("FlyQuest", "Imperial Esports", "Imperial Esports")
