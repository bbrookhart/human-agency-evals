import numpy as np
import pandas as pd


def agreement(a, b) -> dict:
    pairs = pd.DataFrame({"a": a, "b": b}).dropna()
    if pairs.empty:
        return {
            "n": 0,
            "percent_agreement": None,
            "kappa": None,
            "weighted_kappa": None,
            "confusion": [[0] * 4 for _ in range(4)],
        }
    if not pairs.isin([0, 1, 2, 3]).all().all():
        raise ValueError("ratings must be integers in 0..3")
    matrix = np.zeros((4, 4), dtype=int)
    for x, y in pairs.itertuples(index=False, name=None):
        matrix[int(x), int(y)] += 1
    observed = matrix / matrix.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))

    def kappa(weights):
        denominator = (weights * expected).sum()
        return None if denominator == 0 else float(1 - (weights * observed).sum() / denominator)

    return {
        "n": len(pairs),
        "percent_agreement": float(np.trace(observed)),
        "kappa": kappa(1 - np.eye(4)),
        "weighted_kappa": kappa((np.arange(4)[:, None] - np.arange(4)[None, :]) ** 2 / 9),
        "confusion": matrix.tolist(),
    }


def ordinal_alpha(ratings: pd.DataFrame) -> float | None:
    """Krippendorff interval-distance alpha for ordinal numeric codes; not ordinal-distance alpha."""
    coincidence = np.zeros((4, 4))
    for row in ratings.to_numpy():
        values = row[~pd.isna(row)]
        if len(values) < 2:
            continue
        if any(v not in (0, 1, 2, 3) for v in values):
            raise ValueError("invalid rating")
        for i, a in enumerate(values):
            for j, b in enumerate(values):
                if i != j:
                    coincidence[int(a), int(b)] += 1 / (len(values) - 1)
    n = coincidence.sum()
    if n <= 1:
        return None
    marginal = coincidence.sum(axis=0)
    distances = (np.arange(4)[:, None] - np.arange(4)[None, :]) ** 2
    observed = (coincidence * distances).sum() / n
    expected = (np.outer(marginal, marginal) * distances).sum() / (n * (n - 1))
    return None if expected == 0 else float(1 - observed / expected)
