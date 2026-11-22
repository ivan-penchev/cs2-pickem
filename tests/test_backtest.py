import json

import numpy as np
import pytest

from cs2pickem.backtest import MAJORS
from cs2pickem.backtest.run import _category, _match_scores, _records, reliability
from cs2pickem.paths import home
from cs2pickem.site import backtest as bt_html


def test_match_scores_basic():
    s = _match_scores([0.5, 0.5], [1, 0])
    assert s["brier"] == pytest.approx(0.25)
    assert s["log_loss"] == pytest.approx(np.log(2))
    assert s["accuracy"] == pytest.approx(0.5)
    s = _match_scores([0.9, 0.2], [1, 0])
    assert s["brier"] == pytest.approx((0.01 + 0.04) / 2)
    assert s["accuracy"] == 1.0
    assert _match_scores([], [])["n"] == 0


def test_reliability_bins_partition():
    rng = np.random.default_rng(0)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(int)
    bins = reliability(p.tolist(), y.tolist())
    assert sum(b["n"] for b in bins) == 500
    for b in bins:
        assert b["lo"] <= b["mean_p"] <= b["hi"]
    assert reliability([1.0], [1])[-1]["n"] == 1  # p == 1 lands in the last bin


def test_records_and_categories():
    ms = [{"a": "A", "b": "B", "winner": "A"}, {"a": "A", "b": "C", "winner": "C"}]
    assert _records(ms, ["A", "B", "C"]) == {"A": (1, 1), "B": (0, 1), "C": (1, 0)}
    assert _category(3, 0, 3, 3) == "perfect"
    assert _category(3, 2, 3, 3) == "advance"
    assert _category(0, 3, 3, 3) == "winless"
    assert _category(2, 3, 3, 3) is None


@pytest.mark.parametrize("major", list(MAJORS))
def test_committed_report_schema(major):
    path = home() / "data" / "backtests" / major / "report.json"
    if not path.exists():
        pytest.skip("backtest not run")
    rep = json.loads(path.read_text())
    assert set(rep["summary"]) >= {"model", "legacy", "vrs_only", "coin_flip", "outcome_calibration"}
    assert rep["summary"]["coin_flip"]["match_scores"]["brier"] == pytest.approx(0.25)
    for st in rep["stages"].values():
        for key in ("model", "legacy", "vrs_only", "coin_flip"):
            m = st[key]
            assert all(0 <= x["p_a"] <= 1 for x in m["matches"])
            if st["type"] == "swiss":
                assert len(m["matches"]) == 33
                assert sum(len(v) for v in m["picks"].values()) == 10
                assert m["n_correct"] == sum(p["correct"] for v in m["picks"].values() for p in v)
                assert m["coin"] == (m["n_correct"] >= 5)
                assert sum(t["p_adv"] for t in m["teams"]) == pytest.approx(8, abs=1e-6)
            else:
                assert m["n_correct"] == sum(p["correct"] for p in m["picks"])
    html = bt_html.render([rep])
    assert rep["name"] in html and "<svg" in html


def test_majors_cover_all_swiss_majors():
    from cs2pickem.backtest.majors import MAJORS as M

    assert len(M) == 5
    assert {"copenhagen2024", "shanghai2024", "austin2025", "budapest2025", "cologne2026"} <= set(M)


def test_bootstrap_ci_deterministic():
    from cs2pickem.backtest.run import bootstrap_ci

    x = np.array([3, 5, 4, 6, 2, 5, 4], float)
    a, b = bootstrap_ci(x), bootstrap_ci(x)
    assert a == b
    assert a[0] <= x.mean() <= a[1]


def test_summarise_majors_schema():
    from cs2pickem.backtest.run import STRATEGIES, summarise_majors

    reps = bt_html.load_reports()
    summ = summarise_majors(reps)
    n = len(summ["stages"])
    assert n == sum(1 for r in reps for st in r["stages"].values() if st["type"] == "swiss")
    assert set(summ["strategies"]) == set(STRATEGIES)
    for v in summ["strategies"].values():
        assert len(v["per_stage"]) == n
        assert v["mean_correct"] == pytest.approx(np.mean(v["per_stage"]))
        assert v["ci"][0] <= v["mean_correct"] <= v["ci"][1]
        assert v["coins"] == sum(c >= 5 for c in v["per_stage"])
    assert summ["strategies"]["vrs_only"]["diff_vs_vrs_only"]["mean"] == 0
    html = bt_html.render(reps, summ)
    assert 'id="strategies"' in html
