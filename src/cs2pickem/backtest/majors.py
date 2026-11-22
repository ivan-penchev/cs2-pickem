"""Static facts about past Majors used for backtests (verified against Liquipedia, VRS repo, HLTV)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StageSpec:
    id: str
    name: str
    page: str  # Liquipedia sub-page
    kind: str  # "swiss" | "playoffs"
    start: str
    end: str


@dataclass(frozen=True)
class MajorSpec:
    id: str
    name: str
    liquipedia: str
    start: str
    end: str
    map_pool: list[str]
    vrs_date: str  # VRS standings file published just before the event (YYYY_MM_DD)
    hltv_ranking: str  # HLTV ranking archive path (YYYY/month/D) dated just before the event
    stages: list[StageSpec] = field(default_factory=list)

    @property
    def vrs_day(self) -> str:
        return self.vrs_date.replace("_", "-")

    @property
    def hltv_day(self) -> str:
        from datetime import datetime

        return datetime.strptime(self.hltv_ranking, "%Y/%B/%d").date().isoformat()


def _stages(dates: list[tuple[str, str]]) -> list[StageSpec]:
    names = [
        ("stage1", "Stage 1", "Stage_1"),
        ("stage2", "Stage 2", "Stage_2"),
        ("stage3", "Stage 3", "Stage_3"),
    ]
    out = [StageSpec(i, n, p, "swiss", s, e) for (i, n, p), (s, e) in zip(names, dates[:3], strict=True)]
    out.append(StageSpec("playoffs", "Playoffs", "Playoffs", "playoffs", *dates[3]))
    return out


MAJORS: dict[str, MajorSpec] = {
    "budapest2025": MajorSpec(
        id="budapest2025",
        name="StarLadder Budapest Major 2025",
        liquipedia="StarLadder/2025/Major",
        start="2025-11-24",
        end="2025-12-14",
        map_pool=["Ancient", "Dust II", "Inferno", "Mirage", "Nuke", "Overpass", "Train"],
        vrs_date="2025_11_03",
        hltv_ranking="2025/november/17",
        stages=_stages(
            [
                ("2025-11-24", "2025-11-27"),
                ("2025-11-29", "2025-12-02"),
                ("2025-12-04", "2025-12-07"),
                ("2025-12-11", "2025-12-14"),
            ]
        ),
    ),
    "cologne2026": MajorSpec(
        id="cologne2026",
        name="IEM Cologne Major 2026",
        liquipedia="Intel_Extreme_Masters/2026/Cologne",
        start="2026-06-02",
        end="2026-06-21",
        map_pool=["Ancient", "Anubis", "Dust II", "Inferno", "Mirage", "Nuke", "Overpass"],
        vrs_date="2026_06_01",
        hltv_ranking="2026/june/1",
        stages=_stages(
            [
                ("2026-06-02", "2026-06-05"),
                ("2026-06-06", "2026-06-09"),
                ("2026-06-11", "2026-06-15"),
                ("2026-06-18", "2026-06-21"),
            ]
        ),
    ),
}

# Liquipedia display name -> VRS team name (HLTV uses the VRS spelling up to case, e.g. "BETBOOM").
LP_TO_VRS: dict[str, str] = {
    "Aurora Gaming": "Aurora",
    "Team Falcons": "Falcons",
    "FaZe Clan": "FaZe",
    "G2 Esports": "G2",
    "The Huns Esports": "The Huns",
    "Imperial Esports": "Imperial",
    "Lynn Vision Gaming": "Lynn Vision",
    "paiN Gaming": "paiN",
    "Team Spirit": "Spirit",
    "Team Liquid": "Liquid",
    "Team Vitality": "Vitality",
    "9z Team": "9z",
    "BetBoom Team": "BetBoom",
    "FUT Esports": "FUT",
    "DENDELE CS": "Sharks",  # VRS/HLTV still listed the roster under its previous organisation
    "SINNERS Esports": "SINNERS",
    "Off Chops": "THUNDER dOWNUNDER",
    "Fnatic": "fnatic",
}
