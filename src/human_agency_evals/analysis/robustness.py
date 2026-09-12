import pandas as pd

from human_agency_evals.scorers.composite import composite


def leave_one_scenario_out(frame, metric):
    return pd.DataFrame(
        [
            {
                "excluded_base": base,
                "mean": frame[frame.base_id != base].groupby("base_id")[metric].mean().mean(),
            }
            for base in sorted(frame.base_id.unique())
        ]
    )


def weight_sensitivity(frame, schemes):
    rows = []
    for name, weights in schemes.items():
        copy = frame.copy()
        copy["score"] = [composite(row, weights) for row in copy.to_dict("records")]
        for (model, arm), group in copy.groupby(["model", "intervention"]):
            rows.append(
                {
                    "scheme": name,
                    "model": model,
                    "intervention": arm,
                    "mean": group.groupby("base_id").score.mean().mean(),
                }
            )
    return pd.DataFrame(rows)


def stratified(frame, factor, metric):
    return (
        frame.groupby([factor, "model", "intervention", "base_id"], dropna=False)[metric]
        .mean()
        .groupby(level=[0, 1, 2])
        .agg(["mean", "count"])
        .reset_index()
    )


def intervention_sensitivity(frame, samples=1000, seed=1729):
    """Leave-one-base-out contrasts, keeping model and grader identities separate."""
    from human_agency_evals.analysis.metrics import paired_effect

    rows = []
    for (model, grader), group in frame.groupby(["model", "grader"]):
        for arm in sorted(set(group.intervention) - {"control"}):
            for excluded in [None, *sorted(group.base_id.unique())]:
                selected = group if excluded is None else group[group.base_id != excluded]
                for metric in ("helpfulness", "autonomy_provisional", "action_pressure"):
                    effect = paired_effect(
                        selected, "intervention", "control", arm, metric, samples, seed
                    )
                    rows.append(
                        {"model": model, "grader": grader, "excluded_base": excluded, **effect}
                    )
    return pd.DataFrame(
        rows,
        columns=[
            "model",
            "grader",
            "excluded_base",
            "factor",
            "control",
            "treatment",
            "metric",
            "mean",
            "low",
            "high",
            "n_clusters",
            "matched_rows",
            "unmatched_or_missing_rows",
            "standardized_paired_effect",
            "p_value",
        ],
    )
