from __future__ import annotations

import numpy as np


def logistic(ratings: list[float]) -> np.ndarray:
    """Elo-style P(i beats j) matrix."""
    r = np.asarray(ratings, float)
    return 1 / (1 + 10 ** ((r[None, :] - r[:, None]) / 400))
