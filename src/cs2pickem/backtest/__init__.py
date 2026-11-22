"""Point-in-time backtests of the model on past Majors.

* :mod:`.prepare` – build configs, historical snapshots and actual results for a past Major (from
  Liquipedia + the VRS GitHub repo + archived HLTV rankings).
* :mod:`.run` – re-run the model stage by stage using only data available before each stage and grade it.
"""

from .majors import MAJORS, MajorSpec

__all__ = ["MAJORS", "MajorSpec"]
