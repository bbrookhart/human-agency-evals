import numpy as np
import pandas as pd


def cluster_interval(
    frame: pd.DataFrame, value: str, cluster: str = "base_id", samples: int = 1000, seed: int = 1729
) -> dict:
    """Equal-weight base-scenario estimand, retaining within-scenario dependence."""
    means = frame.groupby(cluster)[value].mean().dropna().to_numpy()
    if len(means) == 0:
        return {"mean": None, "low": None, "high": None, "n_clusters": 0}
    if len(means) == 1:
        return {"mean": float(means[0]), "low": None, "high": None, "n_clusters": 1}
    rng = np.random.default_rng(seed)
    draws = rng.choice(means, size=(samples, len(means)), replace=True).mean(axis=1)
    return {
        "mean": float(means.mean()),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "n_clusters": len(means),
    }
