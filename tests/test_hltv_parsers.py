from cs2pickem.data.hltv import parse_match_maps, parse_results_list

RESULT = (
    '<div class="result-con " data-zonedgrouping-entry-unix="{unix}">'
    '<a href="/matches/{mid}/a-vs-b-event" class="a-reset"><div class="result"><table><tr>'
    '<td><div class="team ">{t1}</div></td><td class="result-score">'
    '<span class="score-{c1}">{s1}</span> - <span class="score-{c2}">{s2}</span></td>'
    '<td><div class="team team-won">{t2}</div></td>'
    '<td><div class="map map-text">{fmt}</div></td></tr></table></div></a></div>'
)

MAP = (
    '<div class="mapholder"><div class="mapname">{name}</div>'
    '<div class="results-teamname text-ellipsis">{t1}</div><div class="results-team-score">{s1}</div>'
    '<div class="results-teamname text-ellipsis">{t2}</div><div class="results-team-score">{s2}</div></div>'
)


def test_parse_results_list():
    page = "".join(
        [
            RESULT.format(
                unix=1759690267000, mid=1, t1="FaZe", t2="3DMAX", c1="lost", s1=0, c2="won", s2=2, fmt="bo3"
            ),
            RESULT.format(
                unix=1759600000000, mid=2, t1="NiP", t2="M80", c1="won", s1=13, c2="lost", s2=7, fmt="anc"
            ),
            RESULT.format(
                unix=1759500000000, mid=3, t1="A&amp;B", t2="C", c1="won", s1=1, c2="lost", s2=0, fmt="def"
            ),
        ]
    )
    rows = parse_results_list(page)
    assert [r["match_id"] for r in rows] == [1, 2, 3]
    assert rows[0] == {
        "match_id": 1,
        "slug": "a-vs-b-event",
        "date": "2025-10-05",
        "team1": "FaZe",
        "team2": "3DMAX",
        "score1": 0,
        "score2": 2,
        "fmt": "bo3",
    }
    assert rows[1]["fmt"] == "Ancient" and rows[1]["score1"] == 13
    assert rows[2]["fmt"] == "def" and rows[2]["team1"] == "A&B"


def test_parse_match_maps_skips_unplayed():
    page = "".join(
        [
            MAP.format(name="Train", t1="FaZe", s1=9, t2="3DMAX", s2=13),
            MAP.format(name="Dust2", t1="FaZe", s1=11, t2="3DMAX", s2=13),
            MAP.format(name="Nuke", t1="FaZe", s1="-", t2="3DMAX", s2="-"),
            MAP.format(name="Default", t1="FaZe", s1=0, t2="3DMAX", s2=0),
        ]
    )
    maps = parse_match_maps(page)
    assert [m["map"] for m in maps] == ["Train", "Dust II"]
    assert maps[0]["score1"] == 9 and maps[0]["score2"] == 13
