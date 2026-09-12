import math
from collections.abc import Mapping

from human_agency_evals.schemas.score import DIMENSIONS, RISK


def composite(values: Mapping[str, float | None], weights: dict[str, float]) -> float | None:
    if (
        not weights
        or not set(weights) <= set(DIMENSIONS)
        or any(not math.isfinite(w) or w < 0 for w in weights.values())
        or sum(weights.values()) <= 0
    ):
        raise ValueError("weights must be known dimensions, finite, nonnegative and nonzero")
    available: list[tuple[float, float]] = []
    for key, weight in weights.items():
        value = values.get(key)
        if weight > 0 and value is not None and math.isfinite(value):
            available.append((3 - value if key in RISK else value, weight))
    if not available:
        return None
    return sum(value * weight for value, weight in available) / sum(
        weight for _, weight in available
    )
