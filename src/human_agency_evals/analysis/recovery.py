"""Separate weak/strong updates and selection on prior judged reinforcement."""

import pandas as pd

from human_agency_evals.analysis.bootstrap import cluster_interval


def recovery_table(frame, samples=1000, seed=1729):
    """Exploratory conditioning is post-treatment selection, not a causal arm contrast."""
    keys = ["transcript_id", "grader", "grader_repeat"]
    previous = frame[frame.turn < 4].groupby(keys).sycophancy.max().rename("prior_sycophancy")
    updated = frame[(frame.turn >= 4) & (frame.counterevidence_condition != "absent")].join(
        previous, on=keys
    )
    rows = []
    for (model, arm, domain, strength), group in updated.groupby(
        ["model", "intervention", "domain", "counterevidence_condition"]
    ):
        for selection in ("all_updated", "prior_sycophancy_ge_2"):
            selected = group if selection == "all_updated" else group[group.prior_sycophancy >= 2]
            rows.append(
                {
                    "model": model,
                    "intervention": arm,
                    "domain": domain,
                    "strength": strength,
                    "selection": selection,
                    "judgments": len(selected),
                    **cluster_interval(selected, "counterevidence", samples=samples, seed=seed),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "model",
            "intervention",
            "domain",
            "strength",
            "selection",
            "judgments",
            "mean",
            "low",
            "high",
            "n_clusters",
        ],
    )
