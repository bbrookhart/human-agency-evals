import numpy as np
import pandas as pd
from scipy import stats

from human_agency_evals.analysis.bootstrap import cluster_interval
from human_agency_evals.scorers.deterministic import conclusion_shift

MATCH = [
    "base_id",
    "model",
    "repetition",
    "turn",
    "grader",
    "grader_repeat",
    "position",
    "user_confidence",
    "validation_seeking",
    "counterevidence_condition",
    "intervention",
]


def paired_effect(frame, factor, control, treatment, metric, samples=1000, seed=1729):
    keys = [
        k
        for k in MATCH
        + ["wording_variant", "temperature", "emotional_pressure", "conversation_length"]
        if k != factor and k in frame.columns
    ]
    left = frame.loc[frame[factor] == control, keys + [metric]]
    right = frame.loc[frame[factor] == treatment, keys + [metric]]
    joined = left.merge(
        right, on=keys, suffixes=("_control", "_treatment"), validate="one_to_one"
    ).dropna(subset=[metric + "_control", metric + "_treatment"])
    joined["difference"] = joined[metric + "_treatment"] - joined[metric + "_control"]
    result = cluster_interval(joined, "difference", samples=samples, seed=seed)
    scenario_differences = joined.groupby("base_id").difference.mean()
    sd = scenario_differences.std(ddof=1)
    result.update(
        {
            "factor": factor,
            "control": control,
            "treatment": treatment,
            "metric": metric,
            "matched_rows": len(joined),
            "unmatched_or_missing_rows": len(left) + len(right) - 2 * len(joined),
            "standardized_paired_effect": float(scenario_differences.mean() / sd)
            if sd > 1e-12
            else None,
            "p_value": float(stats.ttest_1samp(scenario_differences, 0).pvalue)
            if len(scenario_differences) >= 3 and sd > 1e-12
            else None,
        }
    )
    return result


def bccs(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "pair_id",
        "base_id",
        "domain",
        "model",
        "intervention",
        "repetition",
        "turn",
        "grader",
        "grader_repeat",
    ]
    a = frame[frame.position == "A"][keys + ["conclusion"]]
    b = frame[frame.position == "B"][keys + ["conclusion"]]
    joined = a.merge(b, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    joined["bccs"] = [
        conclusion_shift(a, b)
        for a, b in zip(joined.conclusion_a, joined.conclusion_b, strict=True)
    ]
    joined["follows_both_positions"] = (joined.conclusion_a == "A") & (joined.conclusion_b == "B")
    return joined


def pareto(frame: pd.DataFrame, x="helpfulness", y="autonomy_provisional") -> pd.Series:
    values = frame[[x, y]].to_numpy()
    return pd.Series(
        [
            bool(
                np.isfinite(point).all()
                and not any(np.all(other >= point) and np.any(other > point) for other in values)
            )
            for point in values
        ],
        index=frame.index,
    )


def fdr(pvalues):
    from statsmodels.stats.multitest import multipletests

    values = np.asarray(pvalues, dtype=float)
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    if valid.any():
        result[valid] = multipletests(values[valid], method="fdr_bh")[1]
    return result
