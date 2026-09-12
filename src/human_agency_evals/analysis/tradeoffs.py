"""Model-specific descriptive contrasts and conservative three-state decisions."""

import pandas as pd

from human_agency_evals.analysis.metrics import paired_effect


def model_tradeoffs(frame, margin, samples=1000, seed=1729, incomplete_models=()):
    rows = []
    for model, group in frame.groupby("model"):
        for arm in sorted(set(group.intervention) - {"control"}):
            utility = paired_effect(
                group, "intervention", "control", arm, "helpfulness", samples, seed
            )
            autonomy = paired_effect(
                group, "intervention", "control", arm, "autonomy_provisional", samples, seed
            )
            test_data = bool(group.test_data.any())
            estimable = all(
                pd.notna(effect["low"])
                and effect["n_clusters"] >= 3
                and effect["matched_rows"] > 0
                and effect["unmatched_or_missing_rows"] == 0
                for effect in (utility, autonomy)
            )
            allowed = estimable and not test_data and model not in incomplete_models
            rows.append(
                {
                    "model": model,
                    "intervention": arm,
                    "helpfulness_margin": margin,
                    "helpfulness_difference": utility["mean"],
                    "helpfulness_low": utility["low"],
                    "helpfulness_high": utility["high"],
                    "autonomy_difference": autonomy["mean"],
                    "autonomy_low": autonomy["low"],
                    "autonomy_high": autonomy["high"],
                    "n_clusters": min(utility["n_clusters"], autonomy["n_clusters"]),
                    "helpfulness_noninferior": bool(utility["low"] > -margin) if allowed else None,
                    "autonomy_improved": bool(autonomy["low"] > 0) if allowed else None,
                    "interpretation": "TEST DATA; not assessed"
                    if test_data
                    else "Not assessed: missing observations, unmatched cells or insufficient clusters"
                    if not allowed
                    else "Provisional interval criteria only; requires rubric validation and justified margin",
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "model",
            "intervention",
            "helpfulness_margin",
            "helpfulness_difference",
            "helpfulness_low",
            "helpfulness_high",
            "autonomy_difference",
            "autonomy_low",
            "autonomy_high",
            "n_clusters",
            "helpfulness_noninferior",
            "autonomy_improved",
            "interpretation",
        ],
    )
