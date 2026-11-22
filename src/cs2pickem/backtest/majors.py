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
    vrs_date: str | None  # VRS file published just before the event (YYYY_MM_DD); None = predates VRS
    hltv_ranking: str  # HLTV ranking archive path (YYYY/month/D) dated just before the event
    stages: list[StageSpec] = field(default_factory=list)
    note: str | None = None  # documented substitutions for missing point-in-time data

    @property
    def synthetic_vrs(self) -> bool:
        return self.vrs_date is None

    @property
    def vrs_day(self) -> str:
        """Snapshot day of the VRS table; without VRS it is a synthetic table built from the HLTV ranking."""
        return self.vrs_date.replace("_", "-") if self.vrs_date else self.hltv_day

    @property
    def hltv_day(self) -> str:
        from datetime import datetime

        return datetime.strptime(self.hltv_ranking, "%Y/%B/%d").date().isoformat()


def _stages(dates: list[tuple[str, str]], playoff_page: str = "Playoffs") -> list[StageSpec]:
    """Three Swiss stages (Stage_1..3) followed by the playoffs (2025+ Major format)."""
    names = [
        ("stage1", "Stage 1", "Stage_1"),
        ("stage2", "Stage 2", "Stage_2"),
        ("stage3", "Stage 3", "Stage_3"),
    ]
    out = [StageSpec(i, n, p, "swiss", s, e) for (i, n, p), (s, e) in zip(names, dates[:3], strict=True)]
    out.append(StageSpec("playoffs", "Playoffs", playoff_page, "playoffs", *dates[3]))
    return out


def _stages_2024(dates: list[tuple[str, str]], playoff_page: str) -> list[StageSpec]:
    """2024 Major format: Opening Stage + Elimination Stage (both 16-team Swiss), then the playoffs.
    ``playoff_page`` "" means the bracket lives on the Major's root page."""
    return [
        StageSpec("stage1", "Opening Stage", "Opening_Stage", "swiss", *dates[0]),
        StageSpec("stage2", "Elimination Stage", "Elimination_Stage", "swiss", *dates[1]),
        StageSpec("playoffs", "Playoffs", playoff_page, "playoffs", *dates[2]),
    ]


MAJORS: dict[str, MajorSpec] = {
    "copenhagen2024": MajorSpec(
        id="copenhagen2024",
        name="PGL Major Copenhagen 2024",
        liquipedia="PGL/2024/Copenhagen",
        start="2024-03-17",
        end="2024-03-31",
        map_pool=["Ancient", "Anubis", "Inferno", "Mirage", "Nuke", "Overpass", "Vertigo"],
        vrs_date=None,
        hltv_ranking="2024/march/11",
        stages=_stages_2024(
            [("2024-03-17", "2024-03-20"), ("2024-03-21", "2024-03-24"), ("2024-03-28", "2024-03-31")],
            playoff_page="",
        ),
        note=(
            "Predates the Valve Regional Standings (earliest file in the VRS repo: 2024-08-06). The VRS input "
            "is a synthetic table built from the HLTV ranking of 2024-03-11: rank = HLTV rank, points = "
            "215*ln(HLTV points) + 515 (fitted on the Budapest/Cologne snapshots, corr 0.93). No point-in-time "
            "match history exists, so BT runs on that prior plus earlier stages of the Major only."
        ),
    ),
    "shanghai2024": MajorSpec(
        id="shanghai2024",
        name="Perfect World Shanghai Major 2024",
        liquipedia="Perfect_World/Major/2024/Shanghai",
        start="2024-11-30",
        end="2024-12-15",
        map_pool=["Ancient", "Anubis", "Dust II", "Inferno", "Mirage", "Nuke", "Vertigo"],
        vrs_date="2024_11_25",
        hltv_ranking="2024/november/25",
        stages=_stages_2024(
            [("2024-11-30", "2024-12-03"), ("2024-12-05", "2024-12-08"), ("2024-12-12", "2024-12-15")],
            playoff_page="Playoff_Stage",
        ),
    ),
    "austin2025": MajorSpec(
        id="austin2025",
        name="BLAST.tv Austin Major 2025",
        liquipedia="BLAST/Major/2025/Austin",
        start="2025-06-03",
        end="2025-06-22",
        map_pool=["Ancient", "Anubis", "Dust II", "Inferno", "Mirage", "Nuke", "Train"],
        vrs_date="2025_06_02",
        hltv_ranking="2025/june/2",
        stages=_stages(
            [
                ("2025-06-03", "2025-06-06"),
                ("2025-06-07", "2025-06-10"),
                ("2025-06-12", "2025-06-15"),
                ("2025-06-19", "2025-06-22"),
            ]
        ),
    ),
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
    "AMKAL ESPORTS": "AMKAL",
    "Nemiga Gaming": "Nemiga",
}
